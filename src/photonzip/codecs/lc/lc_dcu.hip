// DCU (HIP) implementation of the fixed LC pipeline DIFFMS_2 + BIT_2 + RZE_2.
//
// The kernels are adapted from the standalone compressor/decompressor that the upstream LC
// framework generates for this pipeline (HIPversion/generate_standalone_GPU_compressor_
// decompressor.py "" "DIFFMS_2 BIT_2 RZE_2"). Changes against the generated code:
//   - the dynamic work counter is passed in as a pointer instead of a module-level global,
//     so independent calls do not share state;
//   - a chunk size mismatch in the decoder sets an error flag instead of trapping;
//   - __funnelshift_r is replaced by a portable shift, since not every DTK release
//     provides the intrinsic.
// The chunk layout, component order and header are unchanged, so the bitstream is
// byte-for-byte compatible with the host pipeline in lc_pipeline.hpp.

// As in the generated code: the component asserts are compile-time invariants and must not
// end up in device code.
#ifndef NDEBUG
#define NDEBUG
#endif

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "photonzip/codecs/lc/lc_dcu.hpp"

namespace photonzip {
namespace lc {
namespace dcu {
namespace {

static const int CS = 1024 * 16;  // chunk size in bytes, must match lc_pipeline.hpp
static const int TPB = 512;       // threads per block, power of 2 and at least 128

// DCUs are wave64 devices. The device pass of hipcc defines the wavefront size; the host
// pass never instantiates the kernels, so its value there does not matter.
#if defined(PHOTONZIP_LC_WS)
#define WS PHOTONZIP_LC_WS
#elif defined(__AMDGCN_WAVEFRONT_SIZE)
#define WS __AMDGCN_WAVEFRONT_SIZE
#elif defined(__AMDGCN_WAVEFRONT_SIZE__)
#define WS __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__HIP_DEVICE_COMPILE__)
#define WS 64
#else
#define WS 32
#endif

// The LC component headers expect byte/CS/TPB/WS to be visible and define `static inline`
// device helpers, so including them inside this namespace keeps them local to this file.
#include "lossless/lc/include/macros.h"
#include "lossless/lc/include/sum_reduction.h"
#include "lossless/lc/include/prefix_sum.h"
#include "lossless/lc/components/d_DIFFMS_2.h"
#include "lossless/lc/components/d_BIT_2.h"
#include "lossless/lc/components/d_RZE_2.h"

// Equivalent of CUDA's __funnelshift_r for shift amounts in [0, 31].
static inline __device__ int funnelshift_r(const int lo, const int hi, const int shift)
{
  const unsigned long long v =
      ((unsigned long long)(unsigned int)hi << 32) | (unsigned long long)(unsigned int)lo;
  return (int)(unsigned int)(v >> (shift & 31));
}

// copy (len) bytes from shared memory (source) to global memory (destination)
// source must be word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}

// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must be word aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}

template <typename T>
__device__ inline T atomicRead(T* const addr)
{
  return *((volatile T*)addr);
}

template <typename T>
__device__ inline void atomicWrite(T* const addr, const T val)
{
  *((volatile T*)addr) = val;
}

// Decoupled look-back: publishes this chunk's compressed size and waits for the running
// offset of all earlier chunks.
static inline __device__ void propagate_carry(const int value, const long long chunkID, long long* const __restrict__ fullcarry, long long* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    atomicWrite(&fullcarry[chunkID], (chunkID == 0) ? (long long)value : (long long)-value);
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const long long cidm1ml = chunkID - 1 - lane;
      long long val = -1;
      // The generated code has __syncwarp() here; LC maps it to a no-op on AMD/DCU, where
      // a wavefront executes in lockstep, so it is omitted to avoid depending on its
      // declaration in the host compilation pass.
      do {
        if (cidm1ml >= 0) {
          val = atomicRead(&fullcarry[cidm1ml]);
        }
      } while ((__any(val == 0)) || (__all(val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot(val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot(val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      long long partc = (lane < pos) ? -val : 0;
      partc += __shfl_xor(partc, 1);
      partc += __shfl_xor(partc, 2);
      partc += __shfl_xor(partc, 4);
      partc += __shfl_xor(partc, 8);
      partc += __shfl_xor(partc, 16);
#if defined(WS) && (WS == 64)
      partc += __shfl_xor(partc, 32);
#endif
      if (lane == pos) {
        const long long fullc = partc + val;
        atomicWrite(&fullcarry[chunkID], fullc + value);
        *s_fullc = fullc;
      }
    }
  }
}

__global__ __launch_bounds__(TPB)
void d_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry, unsigned long long* const __restrict__ chunk_counter)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = (long long)atomicAdd(chunk_counter, 1ULL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)((insize - base < (long long)CS) ? (insize - base) : (long long)CS);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_DIFFMS_2(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_BIT_2(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_RZE_2(csize, in, out, temp);
      __syncthreads();
    }

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}

__global__ __launch_bounds__(TPB)
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, unsigned long long* const __restrict__ chunk_counter, int* const __restrict__ error)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // input header
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = (long long)atomicAdd(chunk_counter, 1ULL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)((outsize - base < (long long)CS) ? (outsize - base) : (long long)CS);
    if (csize < osize) {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_iRZE_2(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iBIT_2(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iDIFFMS_2(csize, in, out, temp);
      __syncthreads();
    }

    // csize is identical in all threads of the block, so this branch is uniform
    if (csize != osize) {
      if (tid == 0) atomicExch(error, 1);
      continue;
    }

    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);
}

// =========================================================
// Host side
// =========================================================

void check_hip(hipError_t status, const char* what)
{
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("LC DCU: ") + what + ": " + hipGetErrorString(status));
  }
}

long long max_encoded_size(long long insize)
{
  const long long chunks = (insize + CS - 1) / CS;
  return 2 * static_cast<long long>(sizeof(long long)) +
         chunks * static_cast<long long>(sizeof(unsigned short)) + chunks * CS;
}

// Grow-only device allocation. It is never released: the cached buffers live until the
// process exits, because freeing them from a static destructor can run after the HIP
// runtime has already been torn down (HDF5 unloads plugins at exit).
struct DeviceScratch {
  void* ptr = nullptr;
  size_t bytes = 0;

  void* reserve(size_t want, const char* what)
  {
    if (want > bytes) {
      if (ptr != nullptr) {
        check_hip(hipFree(ptr), what);
        ptr = nullptr;
        bytes = 0;
      }
      check_hip(hipMalloc(&ptr, want), what);
      bytes = want;
    }
    return ptr;
  }
};

// Scratch buffers for one call. Workspaces are pooled per device and leased for the duration
// of a call, so concurrent host threads (one per DCU, or several per DCU) never share one,
// and a buffer is never used on a device it was not allocated on.
struct Workspace {
  DeviceScratch control;  // counter, size and error words
  DeviceScratch carry;    // per-chunk carries for the encoder
  DeviceScratch input;    // staging buffers for the host wrappers
  DeviceScratch output;
  int encode_blocks = 0;  // grid sizes for this device, computed on first use
  int decode_blocks = 0;
};

class WorkspaceLease {
 public:
  WorkspaceLease()
  {
    check_hip(hipGetDevice(&device_), "hipGetDevice");
    std::lock_guard<std::mutex> lock(mutex());
    auto& free_list = pool()[device_];
    if (free_list.empty()) {
      ws_ = new Workspace();  // intentionally leaked, see DeviceScratch
    } else {
      ws_ = free_list.back();
      free_list.pop_back();
    }
  }
  ~WorkspaceLease()
  {
    std::lock_guard<std::mutex> lock(mutex());
    pool()[device_].push_back(ws_);
  }
  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;
  Workspace& operator*() const { return *ws_; }

 private:
  static std::mutex& mutex()
  {
    static std::mutex m;
    return m;
  }
  static std::map<int, std::vector<Workspace*>>& pool()
  {
    static auto* p = new std::map<int, std::vector<Workspace*>>();
    return *p;
  }
  int device_ = 0;
  Workspace* ws_ = nullptr;
};

struct Control {
  unsigned long long counter;
  long long size;
  int error;
};

int grid_size(const void* kernel)
{
  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");
  hipDeviceProp_t prop;
  check_hip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
  int per_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, TPB, 0) != hipSuccess || per_cu < 1) {
    per_cu = (prop.maxThreadsPerMultiProcessor / TPB > 0) ? prop.maxThreadsPerMultiProcessor / TPB : 1;
  }
  return prop.multiProcessorCount * per_cu;
}

// All work goes to the calling thread's default stream; with -fgpu-default-stream=per-thread
// (set by the build) threads on the same DCU run concurrently.
long long encode_device_ws(Workspace& ws, const byte* d_input, long long insize, byte* d_output)
{
  if (insize < 0) {
    throw std::runtime_error("LC DCU: negative input size.");
  }
  if (insize == 0) {
    // The kernel only writes the header when it processes a chunk.
    const long long header = 0;
    check_hip(hipMemcpy(d_output, &header, sizeof(header), hipMemcpyHostToDevice), "header H2D");
    return static_cast<long long>(sizeof(long long));
  }

  const long long chunks = (insize + CS - 1) / CS;
  auto* control = static_cast<Control*>(ws.control.reserve(sizeof(Control), "hipMalloc control"));
  auto* carry = static_cast<long long*>(ws.carry.reserve(chunks * sizeof(long long), "hipMalloc carry"));
  check_hip(hipMemsetAsync(control, 0, sizeof(Control), nullptr), "hipMemset control");
  check_hip(hipMemsetAsync(carry, 0, chunks * sizeof(long long), nullptr), "hipMemset carry");

  if (ws.encode_blocks == 0) ws.encode_blocks = grid_size(reinterpret_cast<const void*>(&d_encode));
  d_encode<<<ws.encode_blocks, TPB>>>(d_input, insize, d_output, &control->size, carry, &control->counter);
  check_hip(hipGetLastError(), "d_encode launch");

  long long outsize = 0;
  check_hip(hipMemcpy(&outsize, &control->size, sizeof(outsize), hipMemcpyDeviceToHost), "size D2H");
  if (outsize <= 0 || outsize > max_encoded_size(insize)) {
    throw std::runtime_error("LC DCU: encoder produced an invalid size.");
  }
  return outsize;
}

long long decode_device_ws(Workspace& ws, const byte* d_input, byte* d_output, long long out_capacity)
{
  long long outsize = 0;
  check_hip(hipMemcpy(&outsize, d_input, sizeof(outsize), hipMemcpyDeviceToHost), "header D2H");
  if (outsize < 0) {
    throw std::runtime_error("LC DCU: compressed payload reports a negative size.");
  }
  if (outsize > out_capacity) {
    throw std::runtime_error("LC DCU: output buffer is too small.");
  }
  if (outsize == 0) {
    return 0;
  }

  auto* control = static_cast<Control*>(ws.control.reserve(sizeof(Control), "hipMalloc control"));
  check_hip(hipMemsetAsync(control, 0, sizeof(Control), nullptr), "hipMemset control");

  if (ws.decode_blocks == 0) ws.decode_blocks = grid_size(reinterpret_cast<const void*>(&d_decode));
  d_decode<<<ws.decode_blocks, TPB>>>(d_input, d_output, &control->counter, &control->error);
  check_hip(hipGetLastError(), "d_decode launch");

  int error = 0;
  check_hip(hipMemcpy(&error, &control->error, sizeof(error), hipMemcpyDeviceToHost), "error D2H");
  if (error != 0) {
    throw std::runtime_error("LC DCU: decode chunk size mismatch (corrupt payload).");
  }
  return outsize;
}

}  // namespace

bool device_available()
{
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

long long encode_device(const byte* d_input, long long insize, byte* d_output)
{
  WorkspaceLease ws;
  return encode_device_ws(*ws, d_input, insize, d_output);
}

long long decode_device(const byte* d_input, byte* d_output, long long out_capacity)
{
  WorkspaceLease ws;
  return decode_device_ws(*ws, d_input, d_output, out_capacity);
}

long long decoded_size_device(const byte* d_input)
{
  long long outsize = 0;
  check_hip(hipMemcpy(&outsize, d_input, sizeof(outsize), hipMemcpyDeviceToHost), "header D2H");
  return outsize;
}

long long encode_host(const byte* input, long long insize, byte* output)
{
  WorkspaceLease lease;
  Workspace& ws = *lease;
  const long long capacity = max_encoded_size(insize);
  auto* d_in = static_cast<byte*>(ws.input.reserve(insize > 0 ? insize : 1, "hipMalloc input"));
  auto* d_out = static_cast<byte*>(ws.output.reserve(capacity, "hipMalloc output"));
  if (insize > 0) {
    check_hip(hipMemcpy(d_in, input, insize, hipMemcpyHostToDevice), "input H2D");
  }
  const long long outsize = encode_device_ws(ws, d_in, insize, d_out);
  check_hip(hipMemcpy(output, d_out, outsize, hipMemcpyDeviceToHost), "output D2H");
  return outsize;
}

long long decode_host(const byte* input, long long insize, byte* output, long long out_capacity)
{
  if (insize < static_cast<long long>(sizeof(long long))) {
    throw std::runtime_error("LC DCU: compressed payload is missing its header.");
  }
  long long outsize = 0;
  std::memcpy(&outsize, input, sizeof(outsize));
  if (outsize < 0 || outsize > out_capacity) {
    throw std::runtime_error("LC DCU: output buffer is too small.");
  }

  WorkspaceLease lease;
  Workspace& ws = *lease;
  auto* d_in = static_cast<byte*>(ws.input.reserve(insize, "hipMalloc input"));
  auto* d_out = static_cast<byte*>(ws.output.reserve(outsize > 0 ? outsize : 1, "hipMalloc output"));
  check_hip(hipMemcpy(d_in, input, insize, hipMemcpyHostToDevice), "input H2D");
  const long long produced = decode_device_ws(ws, d_in, d_out, outsize);
  if (produced > 0) {
    check_hip(hipMemcpy(output, d_out, produced, hipMemcpyDeviceToHost), "output D2H");
  }
  return produced;
}

}  // namespace dcu
}  // namespace lc
}  // namespace photonzip
