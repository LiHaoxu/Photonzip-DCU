#include "dcu_ans.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "../dcu_workspace.h"

namespace mans::dcu::ans {
namespace {

constexpr std::uint32_t kSymbols = 256;
constexpr std::uint32_t kBlockBytes = 4096;
constexpr std::uint32_t kLanes = 32;
constexpr std::uint32_t kProbBits = 10;
constexpr std::uint32_t kProbTotal = 1u << kProbBits;
constexpr std::uint32_t kAlignmentWords = 8;
constexpr std::uint32_t kStateStart = 1u << 15;
constexpr std::uint32_t kStateMin = 1u << 15;
constexpr std::uint32_t kStateMask = (1u << 16) - 1;
constexpr std::uint32_t kStateCheckMul = 1u << 21;
constexpr std::uint32_t kMagicVersion = 0xd00d0001u;
constexpr std::uint32_t kHeaderBytes = 32;
constexpr std::uint32_t kWarpStateBytes = 128;
constexpr std::uint32_t kDirectoryEntryBytes = 8;

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

__device__ inline std::uint32_t lane32() {
    return static_cast<std::uint32_t>(threadIdx.x) & 31u;
}

__device__ inline std::uint32_t warp_ballot32(bool predicate) {
    // Each CTA contains exactly one logical 32-lane codec warp. On wave64
    // hardware the inactive upper half contributes no bits to this mask.
    return static_cast<std::uint32_t>(__ballot(predicate));
}

__device__ inline std::uint32_t popcount32(std::uint32_t value) {
    return __popc(value);
}

struct Header {
    std::uint32_t magic_version;
    std::uint32_t num_blocks;
    std::uint32_t uncompressed_bytes;
    std::uint32_t compressed_words;
    std::uint32_t options;
    std::uint32_t checksum;
    std::uint32_t unused0;
    std::uint32_t unused1;
};
static_assert(sizeof(Header) == kHeaderBytes, "ANS header size changed");

struct DirectoryEntry { std::uint32_t lengths; std::uint32_t offset_words; };
static_assert(sizeof(DirectoryEntry) == kDirectoryEntryBytes, "ANS directory entry size changed");

// [PhotonZip] Per-block shared-memory histogram, merged into the global one; the counts are
// identical to one global atomicAdd per byte, without the contention on 256 addresses. The
// grid scales with the input (grid-stride loop), so small inputs still use many CUs.
constexpr std::uint32_t kHistogramThreads = 256;
constexpr std::uint32_t kHistogramMaxBlocks = 1024;

inline std::uint32_t histogram_grid(std::uint32_t bytes) {
    const std::uint32_t wanted = (bytes + kHistogramThreads * 16 - 1) / (kHistogramThreads * 16);
    return std::max(1u, std::min(wanted, kHistogramMaxBlocks));
}

__global__ __launch_bounds__(kHistogramThreads) void histogram_kernel(const std::uint8_t* input, std::uint32_t bytes,
                                 std::uint32_t* histogram) {
    __shared__ std::uint32_t local[kSymbols];
    for (std::uint32_t s = threadIdx.x; s < kSymbols; s += blockDim.x) local[s] = 0;
    __syncthreads();
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += stride) {
        atomicAdd(local + input[i], 1u);
    }
    __syncthreads();
    for (std::uint32_t s = threadIdx.x; s < kSymbols; s += blockDim.x) {
        if (local[s] != 0) atomicAdd(histogram + s, local[s]);
    }
}

// [PhotonZip] Parallel replacement for the upstream normalisation, which ran on thread 0 only
// (insertion sort plus adjustment rounds, ~1.3 ms per call). It computes the same table:
//  - q[s] uses the same float expression;
//  - the descending order of the unique keys (q << 16) | s gives each symbol its rank as the
//    number of larger keys, i.e. the insertion-sort result;
//  - the positive adjustment adds ceil(diff / apply) to every rank whose symbol id is below
//    apply, which is what the upstream rounds add one at a time;
//  - the negative adjustment runs the same rounds, counting probabilities > 1 block-wide;
//  - cdf is the exclusive prefix sum over symbols; magic/shift use the same per-symbol formula.
__global__ __launch_bounds__(kSymbols) void normalize_kernel(const std::uint32_t* histogram,
                                 std::uint32_t bytes,
                                 std::uint16_t* probabilities,
                                 std::uint32_t* cdf,
                                 std::uint32_t* magic,
                                 std::uint32_t* shifts) {
    __shared__ std::uint32_t q[kSymbols];
    __shared__ std::uint32_t order[kSymbols];
    __shared__ std::uint32_t prefix[kSymbols];
    __shared__ std::int32_t total;
    __shared__ std::int32_t gt_one;
    const std::uint32_t tid = threadIdx.x;  // one thread per symbol, and per rank

    std::uint32_t value = static_cast<std::uint32_t>(
        static_cast<float>(kProbTotal) * (static_cast<float>(histogram[tid]) / static_cast<float>(bytes)));
    if (histogram[tid] && value == 0) value = 1;
    q[tid] = value;
    if (tid == 0) total = 0;
    __syncthreads();

    const std::uint32_t key = (q[tid] << 16) | tid;
    std::uint32_t rank = 0;
    for (std::uint32_t s = 0; s < kSymbols; ++s) rank += ((q[s] << 16) | s) > key ? 1u : 0u;
    order[rank] = tid;
    atomicAdd(&total, static_cast<std::int32_t>(value));
    __syncthreads();

    prefix[tid] = q[order[tid]];
    std::int32_t diff = static_cast<std::int32_t>(kProbTotal) - total;
    __syncthreads();
    if (diff > 0) {
        const std::int32_t apply = min(diff, static_cast<std::int32_t>(kSymbols));
        const std::int32_t rounds = (diff + apply - 1) / apply;
        if (order[tid] < static_cast<std::uint32_t>(apply)) prefix[tid] += static_cast<std::uint32_t>(rounds);
    } else {
        diff = -diff;
        while (diff > 0) {  // diff, gt_one and apply are uniform across the block
            if (tid == 0) gt_one = 0;
            __syncthreads();
            if (prefix[tid] > 1) atomicAdd(&gt_one, 1);
            __syncthreads();
            const std::int32_t count = gt_one;
            if (count == 0) break;  // unreachable: a total above kProbTotal needs a value > 1
            const std::int32_t apply = min(diff, count);
            const std::int32_t first = count - apply;
            if (static_cast<std::int32_t>(tid) >= first && static_cast<std::int32_t>(tid) < count) --prefix[tid];
            diff -= apply;
            __syncthreads();
        }
    }
    __syncthreads();
    q[order[tid]] = prefix[tid];
    __syncthreads();

    // Inclusive scan of q over the symbol index (prefix[] is free again).
    const std::uint32_t p = q[tid];
    prefix[tid] = p;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < kSymbols; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? prefix[tid - offset] : 0u;
        __syncthreads();
        prefix[tid] += add;
        __syncthreads();
    }
    const std::uint32_t running = prefix[tid] - p;
    probabilities[tid] = static_cast<std::uint16_t>(p);
    cdf[tid] = running;
    if (p == 0) {
        magic[tid] = 0;
        shifts[tid] = 0;
    } else {
        const std::uint32_t shift = 32u - static_cast<std::uint32_t>(__clz(p - 1));
        const std::uint64_t magic64 = ((std::uint64_t{1} << 32) *
            ((std::uint64_t{1} << shift) - p)) / p + 1;
        magic[tid] = static_cast<std::uint32_t>(magic64);
        shifts[tid] = shift;
    }
}

__device__ inline std::uint32_t encode_symbol(
    std::uint32_t& state, std::uint8_t symbol, std::uint32_t offset,
    std::uint16_t* output, const std::uint16_t* probabilities,
    const std::uint32_t* cdf, const std::uint32_t* magic,
    const std::uint32_t* shifts) {
    const std::uint32_t lane = lane32();
    const std::uint32_t pdf = probabilities[symbol];
    const bool write = state >= pdf * kStateCheckMul;
    const std::uint32_t ballot = warp_ballot32(write);
    const std::uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    const std::uint32_t rank = popcount32(ballot & lower);
    if (write) {
        output[offset + rank] = static_cast<std::uint16_t>(state & kStateMask);
        state >>= 16;
    }
    const std::uint32_t quotient = (__umulhi(state, magic[symbol]) + state) >> shifts[symbol];
    const std::uint32_t remainder = state - quotient * pdf;
    state = quotient * kProbTotal + remainder + cdf[symbol];
    return popcount32(ballot);
}

__global__ void encode_blocks_kernel(const std::uint8_t* input, std::uint32_t bytes,
                                     const std::uint16_t* probabilities,
                                     const std::uint32_t* cdf,
                                     const std::uint32_t* magic,
                                     const std::uint32_t* shifts,
                                     std::uint32_t stride_words,
                                     std::uint8_t* scratch,
                                     std::uint32_t* word_counts) {
    const std::uint32_t lane = lane32();
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t count_blocks = bytes / kBlockBytes + (bytes % kBlockBytes != 0);
    __shared__ std::uint16_t shared_prob[kSymbols];
    __shared__ std::uint32_t shared_cdf[kSymbols];
    __shared__ std::uint32_t shared_magic[kSymbols];
    __shared__ std::uint32_t shared_shifts[kSymbols];
    for (std::uint32_t symbol = lane; symbol < kSymbols; symbol += kLanes) {
        shared_prob[symbol] = probabilities[symbol];
        shared_cdf[symbol] = cdf[symbol];
        shared_magic[symbol] = magic[symbol];
        shared_shifts[symbol] = shifts[symbol];
    }
    __syncthreads();
    if (block >= count_blocks) return;
    const std::uint32_t start = block * kBlockBytes;
    const std::uint32_t size = min(kBlockBytes, bytes - start);

    std::uint32_t state = kStateStart;
    std::uint32_t offset = 0;
    const std::uint32_t limit = (size / (kLanes * 8)) * (kLanes * 8);
    for (std::uint32_t index = lane; index < limit; index += kLanes * 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            offset += encode_symbol(
                state, input[start + index + j * kLanes], offset,
                reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes),
                shared_prob, shared_cdf, shared_magic, shared_shifts);
        }
    }
    const std::uint32_t aligned = (size / kLanes) * kLanes;
    for (std::uint32_t index = limit + lane; index < aligned; index += kLanes) {
        offset += encode_symbol(
            state, input[start + index], offset,
            reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes),
            shared_prob, shared_cdf, shared_magic, shared_shifts);
    }
    if (aligned < size) {
        const bool valid = lane < size - aligned;
        const bool write = valid && state >= shared_prob[input[start + aligned + (valid ? lane : 0)]] * kStateCheckMul;
        const std::uint32_t ballot = warp_ballot32(write);
        const std::uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1u);
        const std::uint32_t rank = popcount32(ballot & lower);
        if (write) {
            reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes)[offset + rank] =
                static_cast<std::uint16_t>(state & kStateMask);
            state >>= 16;
        }
        if (valid) {
            const std::uint8_t symbol = input[start + aligned + lane];
            const std::uint32_t pdf = shared_prob[symbol];
            const std::uint32_t quotient = (__umulhi(state, shared_magic[symbol]) + state) >> shared_shifts[symbol];
            const std::uint32_t remainder = state - quotient * pdf;
            state = quotient * kProbTotal + remainder + shared_cdf[symbol];
        }
        offset += popcount32(ballot);
    }
    reinterpret_cast<std::uint32_t*>(scratch + block * stride_words * 2)[lane] = state;
    if (lane == 0) word_counts[block] = offset;
}

// [PhotonZip] The upstream coalesce_kernel ran on a single thread (<<<1, 1>>>) and copied the
// whole stream byte by byte, which dominated encode time (~2 s for 16 MB). It is split into
// layout_kernel (parallel prefix sum + header) and scatter_kernel (one CTA per ANS block).
// The stream layout is unchanged; the alignment padding after each block's words is now
// written as zeros instead of copying uninitialised scratch memory.
constexpr std::uint32_t kLayoutThreads = 1024;
constexpr std::uint32_t kScatterThreads = 256;

__device__ inline std::uint32_t padded_words(std::uint32_t words) {
    return (words + kAlignmentWords - 1) & ~(kAlignmentWords - 1);
}

__device__ inline std::uint32_t stream_overhead(std::uint32_t blocks) {
    return kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes +
           ((blocks + 1) & ~1u) * kDirectoryEntryBytes;
}

// DTK limits kernels without __launch_bounds__ to 256 threads per block.
__global__ __launch_bounds__(kLayoutThreads) void layout_kernel(const std::uint32_t* word_counts,
                              const std::uint16_t* probabilities,
                              std::uint32_t blocks, std::uint32_t input_bytes,
                              std::uint32_t* prefix, std::uint8_t* output,
                              std::uint32_t* output_size) {
    __shared__ std::uint32_t partial[kLayoutThreads];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t per_thread = (blocks + kLayoutThreads - 1) / kLayoutThreads;
    const std::uint32_t begin = min(tid * per_thread, blocks);
    const std::uint32_t end = min(begin + per_thread, blocks);

    std::uint32_t local = 0;
    for (std::uint32_t i = begin; i < end; ++i) local += padded_words(word_counts[i]);
    partial[tid] = local;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < kLayoutThreads; offset <<= 1) {  // inclusive scan
        const std::uint32_t add = tid >= offset ? partial[tid - offset] : 0u;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    std::uint32_t running = tid == 0 ? 0u : partial[tid - 1];
    for (std::uint32_t i = begin; i < end; ++i) {
        prefix[i] = running;
        running += padded_words(word_counts[i]);
    }

    if (tid < kSymbols) {
        reinterpret_cast<std::uint16_t*>(output + kHeaderBytes)[tid] = probabilities[tid];
    }
    if (tid == 0 && (blocks & 1u)) {
        // [PhotonZip] The directory is padded to an even number of entries. scatter_kernel
        // writes one entry per block, so the pad entry used to keep whatever the output buffer
        // held before: the stream then depended on the buffer's history (decoders skip it).
        auto* directory = reinterpret_cast<DirectoryEntry*>(
            output + kHeaderBytes + kSymbols * 2 + static_cast<std::size_t>(blocks) * kWarpStateBytes);
        directory[blocks] = DirectoryEntry{0u, 0u};
    }
    if (tid == kLayoutThreads - 1) {
        const std::uint32_t sum = partial[kLayoutThreads - 1];
        Header header{kMagicVersion, blocks, input_bytes, sum, kProbBits, 0, 0, 0};
        *reinterpret_cast<Header*>(output) = header;
        *output_size = stream_overhead(blocks) + sum * 2;
    }
}

__global__ __launch_bounds__(kScatterThreads) void scatter_kernel(const std::uint8_t* scratch,
                               const std::uint32_t* word_counts,
                               const std::uint32_t* prefix,
                               std::uint32_t blocks, std::uint32_t input_bytes,
                               std::uint32_t stride_words, std::uint8_t* output) {
    const std::uint32_t i = blockIdx.x;
    if (i >= blocks) return;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t words = word_counts[i];
    const std::uint8_t* source = scratch + static_cast<std::size_t>(i) * stride_words * 2;

    if (tid == 0) {
        auto* directory = reinterpret_cast<DirectoryEntry*>(
            output + kHeaderBytes + kSymbols * 2 + static_cast<std::size_t>(blocks) * kWarpStateBytes);
        const std::uint32_t block_bytes = i + 1 == blocks ? input_bytes - i * kBlockBytes : kBlockBytes;
        directory[i].lengths = (block_bytes << 16) | (words & 0xffffu);
        directory[i].offset_words = prefix[i];
    }
    if (tid < kLanes) {
        auto* states = reinterpret_cast<std::uint32_t*>(
            output + kHeaderBytes + kSymbols * 2 + static_cast<std::size_t>(i) * kWarpStateBytes);
        states[tid] = reinterpret_cast<const std::uint32_t*>(source)[tid];
    }
    const auto* src_words = reinterpret_cast<const std::uint16_t*>(source + kWarpStateBytes);
    auto* dst_words = reinterpret_cast<std::uint16_t*>(
        output + stream_overhead(blocks) + static_cast<std::size_t>(prefix[i]) * 2);
    const std::uint32_t padded = padded_words(words);
    for (std::uint32_t w = tid; w < padded; w += blockDim.x) {
        dst_words[w] = w < words ? src_words[w] : std::uint16_t{0};
    }
}

} // namespace

std::size_t get_max_compressed_size(std::size_t input_bytes) {
    if (input_bytes == 0) return kHeaderBytes;
    if (input_bytes > std::numeric_limits<std::uint32_t>::max()) return 0;
    const std::size_t blocks = input_bytes / kBlockBytes + (input_bytes % kBlockBytes != 0);
    constexpr std::size_t max_block_words =
        (kBlockBytes + kBlockBytes / 4 + 15) / 16 * 8;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (blocks > (max_size - kHeaderBytes - kSymbols * 2) /
                     (kWarpStateBytes + 2 * kDirectoryEntryBytes) ||
        blocks > max_size / max_block_words) {
        return 0;
    }
    const std::size_t directory_entries = (blocks + 1) & ~std::size_t{1};
    const std::size_t overhead = kHeaderBytes + kSymbols * 2 +
        blocks * kWarpStateBytes + directory_entries * kDirectoryEntryBytes;
    const std::size_t words = blocks * max_block_words;
    if (words > (max_size - overhead) / 2) return 0;
    return overhead + words * 2;
}

void compress_device(const std::uint8_t* input, std::size_t input_bytes,
                     std::uint8_t* output, std::size_t output_capacity,
                     std::size_t& output_bytes) {
    output_bytes = 0;
    if (!input || !output || input_bytes == 0 || input_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: invalid input or size");
    }
    const auto max_size = get_max_compressed_size(input_bytes);
    if (max_size == 0 || output_capacity < max_size) {
        throw std::runtime_error("DCU ANS: output buffer is too small");
    }
    if (max_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: compressed stream exceeds codec1 size limit");
    }
    const std::uint32_t bytes = static_cast<std::uint32_t>(input_bytes);
    const std::uint32_t blocks = bytes / kBlockBytes + (bytes % kBlockBytes != 0);
    if (blocks > std::numeric_limits<std::uint32_t>::max() / kWarpStateBytes) {
        throw std::runtime_error("DCU ANS: too many blocks");
    }
    const std::uint32_t stride_words = kLanes * 2 + (kBlockBytes + kBlockBytes / 4 + 15) / 16 * 8;
    // [PhotonZip] Scratch buffers come from a reusable workspace instead of nine hipMalloc calls.
    WorkspaceLease ws(kAnsPool);
    auto* histogram = ws->get<std::uint32_t>(0, kSymbols);
    auto* probabilities = ws->get<std::uint16_t>(1, kSymbols);
    auto* cdf = ws->get<std::uint32_t>(2, kSymbols);
    auto* magic = ws->get<std::uint32_t>(3, kSymbols);
    auto* shifts = ws->get<std::uint32_t>(4, kSymbols);
    auto* scratch = ws->get<std::uint8_t>(5, static_cast<std::size_t>(blocks) * stride_words * 2);
    auto* word_counts = ws->get<std::uint32_t>(6, blocks);
    auto* prefix = ws->get<std::uint32_t>(7, static_cast<std::size_t>(blocks) + 1);
    auto* size_device = ws->get<std::uint32_t>(8, 1);
    check_hip(hipMemsetAsync(histogram, 0, kSymbols * sizeof(std::uint32_t), nullptr), "DCU ANS histogram memset");
    histogram_kernel<<<histogram_grid(bytes), kHistogramThreads>>>(input, bytes, histogram);
    normalize_kernel<<<1, kSymbols>>>(histogram, bytes, probabilities, cdf, magic, shifts);
    encode_blocks_kernel<<<blocks, kLanes>>>(input, bytes, probabilities, cdf, magic, shifts,
        stride_words, scratch, word_counts);
    layout_kernel<<<1, kLayoutThreads>>>(word_counts, probabilities, blocks, bytes, prefix, output, size_device);
    scatter_kernel<<<blocks, kScatterThreads>>>(scratch, word_counts, prefix, blocks, bytes, stride_words, output);
    check_hip(hipGetLastError(), "DCU ANS encode launch");
    std::uint32_t actual = 0;
    check_hip(hipMemcpy(&actual, size_device, sizeof(actual), hipMemcpyDeviceToHost), "DCU ANS size D2H");
    if (actual > output_capacity) throw std::runtime_error("DCU ANS: encoded stream exceeds output capacity");
    output_bytes = actual;
}

// [PhotonZip] Upstream decode_kernel had lane 0 of *every* CTA validate the whole block
// directory (O(blocks^2): ~325 ms for 256 MiB) and rebuild the 1024-entry inverse table with an
// O(256) prefix per symbol. The same checks now run once in validate_kernel (parallel prefix
// sum over the directory), table_kernel builds the inverse table once, and decode_kernel only
// copies it to shared memory. Decoding itself is unchanged; a malformed stream still decodes
// nothing and reports error 1.
constexpr std::uint32_t kValidateThreads = 1024;

__device__ inline std::uint32_t padded_directory_words(std::uint32_t lengths) {
    return ((lengths & 0xffffu) + kAlignmentWords - 1) & ~(kAlignmentWords - 1);
}

__global__ __launch_bounds__(kValidateThreads) void validate_kernel(const std::uint8_t* input,
                                                                    std::uint32_t input_bytes,
                                                                    std::uint32_t output_bytes,
                                                                    std::uint32_t* error) {
    __shared__ std::uint32_t partial[kValidateThreads];
    __shared__ std::uint32_t bad;
    const std::uint32_t tid = threadIdx.x;
    const auto* header = reinterpret_cast<const Header*>(input);
    const std::uint32_t blocks = header->num_blocks;
    const std::uint64_t overhead = std::uint64_t{kHeaderBytes} + kSymbols * 2 +
        std::uint64_t{blocks} * kWarpStateBytes + std::uint64_t{(blocks + 1) & ~1u} * kDirectoryEntryBytes;
    if (tid == 0) {
        const auto* probabilities = reinterpret_cast<const std::uint16_t*>(input + kHeaderBytes);
        std::uint32_t total = 0;
        for (std::uint32_t i = 0; i < kSymbols; ++i) total += probabilities[i];
        bad = header->magic_version != kMagicVersion || header->options != kProbBits ||
              header->uncompressed_bytes != output_bytes || total != kProbTotal ||
              blocks != (output_bytes + kBlockBytes - 1) / kBlockBytes ||
              input_bytes < overhead ||
              overhead + std::uint64_t{header->compressed_words} * 2 > input_bytes;
    }
    __syncthreads();
    if (bad) {  // uniform; the directory may not even be inside the input
        if (tid == 0) atomicExch(error, 1u);
        return;
    }

    const auto* directory = reinterpret_cast<const DirectoryEntry*>(
        input + kHeaderBytes + kSymbols * 2 + std::size_t{blocks} * kWarpStateBytes);
    const std::uint32_t per_thread = (blocks + kValidateThreads - 1) / kValidateThreads;
    const std::uint32_t begin = min(tid * per_thread, blocks);
    const std::uint32_t end = min(begin + per_thread, blocks);
    std::uint32_t local = 0;
    for (std::uint32_t i = begin; i < end; ++i) {
        const std::uint32_t expected_size = (i + 1 == blocks) ? output_bytes - i * kBlockBytes : kBlockBytes;
        if ((directory[i].lengths >> 16) != expected_size) bad = 1;
        local += padded_directory_words(directory[i].lengths);
    }
    partial[tid] = local;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < kValidateThreads; offset <<= 1) {  // inclusive scan
        const std::uint32_t add = tid >= offset ? partial[tid - offset] : 0u;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    std::uint32_t expected_offset = tid == 0 ? 0u : partial[tid - 1];
    for (std::uint32_t i = begin; i < end; ++i) {
        if (directory[i].offset_words != expected_offset) bad = 1;
        expected_offset += padded_directory_words(directory[i].lengths);
    }
    if (tid == kValidateThreads - 1 && partial[tid] != header->compressed_words) bad = 1;
    __syncthreads();
    if (tid == 0 && bad) atomicExch(error, 1u);
}

__global__ __launch_bounds__(kSymbols) void table_kernel(const std::uint8_t* input,
                                                         std::uint32_t* inverse,
                                                         const std::uint32_t* error) {
    if (*error) return;
    __shared__ std::uint32_t scan[kSymbols];
    const std::uint32_t symbol = threadIdx.x;
    const auto* probabilities = reinterpret_cast<const std::uint16_t*>(input + kHeaderBytes);
    const std::uint32_t pdf = probabilities[symbol];
    scan[symbol] = pdf;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < kSymbols; offset <<= 1) {
        const std::uint32_t add = symbol >= offset ? scan[symbol - offset] : 0u;
        __syncthreads();
        scan[symbol] += add;
        __syncthreads();
    }
    const std::uint32_t cdf = scan[symbol] - pdf;
    for (std::uint32_t j = 0; j < pdf; ++j) {
        inverse[cdf + j] = symbol | (pdf << 8) | (j << 20);
    }
}

__global__ void decode_kernel(const std::uint8_t* input,
                              std::uint8_t* output,
                              const std::uint32_t* inverse_table,
                              const std::uint32_t* error) {
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const auto* header = reinterpret_cast<const Header*>(input);
    const std::uint32_t blocks = header->num_blocks;
    if (block >= blocks || *error) return;

    __shared__ std::uint32_t inverse[1u << kProbBits];
    for (std::uint32_t i = lane; i < (1u << kProbBits); i += kLanes) inverse[i] = inverse_table[i];
    __syncthreads();

    const auto* directory = reinterpret_cast<const DirectoryEntry*>(input + kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes);
    const auto* states = reinterpret_cast<const std::uint32_t*>(input + kHeaderBytes + kSymbols * 2 + block * kWarpStateBytes);
    const auto entry = directory[block];
    const std::uint32_t count = entry.lengths >> 16;
    std::uint32_t compressed_words = entry.lengths & 0xffffu;
    const auto* data = reinterpret_cast<const std::uint16_t*>(input +
        kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes +
        ((blocks + 1) & ~1u) * kDirectoryEntryBytes) + entry.offset_words;
    std::uint32_t state = states[lane];
    std::uint32_t remainder = count % kLanes;
    std::uint32_t output_offset = count - remainder;
    if (remainder != 0) {
        const bool valid = lane < remainder;
        std::uint32_t symbol = 0, pdf = 0, cdf = 0;
        const std::uint32_t packed = inverse[state & (kProbTotal - 1)];
        symbol = packed & 0xffu;
        pdf = (packed >> 8) & 0xfffu;
        cdf = packed >> 20;
        if (valid) state = pdf * (state >> kProbBits) + cdf;
        const bool read = valid && state < kStateMin;
        const std::uint32_t ballot = warp_ballot32(read);
        const std::uint32_t ge = 0xffffffffu << lane;
        const std::uint32_t rank = popcount32(ballot & ge);
        const std::uint32_t reads = popcount32(ballot);
        if (read) state = (state << 16) + data[compressed_words - rank];
        if (valid) output[block * kBlockBytes + output_offset + lane] = static_cast<std::uint8_t>(symbol);
        compressed_words -= reads;
    }
    while (output_offset > 0) {
        output_offset -= kLanes;
        const std::uint32_t packed = inverse[state & (kProbTotal - 1)];
        const std::uint32_t symbol = packed & 0xffu;
        const std::uint32_t pdf = (packed >> 8) & 0xfffu;
        const std::uint32_t cdf = packed >> 20;
        state = pdf * (state >> kProbBits) + cdf;
        const bool read = state < kStateMin;
        const std::uint32_t ballot = warp_ballot32(read);
        const std::uint32_t ge = 0xffffffffu << lane;
        const std::uint32_t rank = popcount32(ballot & ge);
        const std::uint32_t reads = popcount32(ballot);
        if (read) state = (state << 16) + data[compressed_words - rank];
        output[block * kBlockBytes + output_offset + lane] = static_cast<std::uint8_t>(symbol);
        compressed_words -= reads;
    }
}

void decompress_device(const std::uint8_t* input, std::size_t input_bytes,
                       std::uint8_t* output, std::size_t output_capacity,
                       std::size_t& output_bytes) {
    output_bytes = 0;
    if (!input || !output || input_bytes < kHeaderBytes ||
        input_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: invalid compressed input");
    }
    Header header{};
    check_hip(hipMemcpy(&header, input, sizeof(header), hipMemcpyDeviceToHost), "DCU ANS header D2H");
    if (header.magic_version != kMagicVersion || header.options != kProbBits ||
        header.uncompressed_bytes == 0 || header.num_blocks !=
            (header.uncompressed_bytes + kBlockBytes - 1) / kBlockBytes) {
        throw std::runtime_error("DCU ANS: invalid codec1 header");
    }
    const std::size_t overhead = kHeaderBytes + kSymbols * 2 +
        static_cast<std::size_t>(header.num_blocks) * kWarpStateBytes +
        ((header.num_blocks + 1u) & ~1u) * kDirectoryEntryBytes;
    const std::size_t required = overhead + static_cast<std::size_t>(header.compressed_words) * 2;
    if (required > input_bytes || required < overhead) throw std::runtime_error("DCU ANS: truncated or oversized stream");
    if (header.uncompressed_bytes > output_capacity) throw std::runtime_error("DCU ANS: output buffer is too small");
    WorkspaceLease ws(kAnsPool);
    auto* error = ws->get<std::uint32_t>(9, 1);
    auto* inverse = ws->get<std::uint32_t>(10, kProbTotal);
    check_hip(hipMemsetAsync(error, 0, sizeof(std::uint32_t), nullptr), "DCU ANS decode error initialize");
    validate_kernel<<<1, kValidateThreads>>>(input, static_cast<std::uint32_t>(input_bytes),
        header.uncompressed_bytes, error);
    table_kernel<<<1, kSymbols>>>(input, inverse, error);
    decode_kernel<<<header.num_blocks, kLanes>>>(input, output, inverse, error);
    check_hip(hipGetLastError(), "DCU ANS decode launch");
    std::uint32_t host_error = 0;
    check_hip(hipMemcpy(&host_error, error, sizeof(host_error), hipMemcpyDeviceToHost), "DCU ANS decode validation result");
    if (host_error) throw std::runtime_error(std::string("DCU ANS: invalid codec1 payload (") + std::to_string(host_error) + ")");
    output_bytes = header.uncompressed_bytes;
}

} // namespace mans::dcu::ans
