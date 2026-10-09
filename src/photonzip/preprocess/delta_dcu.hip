#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "photonzip/preprocess/delta.hpp"

namespace photonzip {
namespace delta {
namespace {

constexpr unsigned int kThreads = 256;

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("delta DCU: ") + what + ": " + hipGetErrorString(status));
  }
}

// kVec consecutive elements of a frame, loaded and stored with one 2*kVec-byte access.
template <int kVec>
struct alignas(2 * kVec) Pack {
  std::uint16_t v[kVec];
};

// Each thread owns kVec consecutive elements of a frame (grid-stride over the frame) and walks
// all frames in order; the previous frame lives in registers, so in == out is safe and every
// frame access is coalesced. With prev, frame 0 is coded relative to it as well.
template <bool kOffset, int kVec>
__global__ __launch_bounds__(kThreads) void encode_kernel(const std::uint16_t* in, std::uint16_t* out,
                                                          std::size_t frames, std::size_t frame_elems,
                                                          const std::uint16_t* prev_frame,
                                                          unsigned long long* clipped) {
  __shared__ unsigned long long block_clipped;
  if (threadIdx.x == 0) block_clipped = 0;
  __syncthreads();
  unsigned long long local = 0;
  const std::size_t groups = frame_elems / kVec;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < groups; g += stride) {
    std::size_t first = 0;
    Pack<kVec> prev;
    if (prev_frame != nullptr) {
      prev = *reinterpret_cast<const Pack<kVec>*>(prev_frame + g * kVec);
    } else {
      prev = *reinterpret_cast<const Pack<kVec>*>(in + g * kVec);
      *reinterpret_cast<Pack<kVec>*>(out + g * kVec) = prev;
      first = 1;
    }
    for (std::size_t f = first; f < frames; ++f) {
      const std::size_t idx = f * frame_elems + g * kVec;
      const Pack<kVec> cur = *reinterpret_cast<const Pack<kVec>*>(in + idx);
      Pack<kVec> stored;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        int d = static_cast<int>(cur.v[k]) - static_cast<int>(prev.v[k]);
        if (d < -32768 || d > 32767) {
          ++local;
          d = d < 0 ? -32768 : 32767;
        }
        stored.v[k] = kOffset ? static_cast<std::uint16_t>(d + 32768) : static_cast<std::uint16_t>(d);
      }
      *reinterpret_cast<Pack<kVec>*>(out + idx) = stored;
      prev = cur;
    }
  }
  if (local != 0) atomicAdd(&block_clipped, local);
  __syncthreads();
  if (threadIdx.x == 0 && block_clipped != 0) atomicAdd(clipped, block_clipped);
}

// kWrite: store every restored frame (decode); otherwise only the running sum of the last frame
// goes to `out` (frame_sum, with prev_frame == nullptr meaning a zero frame).
template <bool kOffset, int kVec, bool kWrite>
__global__ __launch_bounds__(kThreads) void decode_kernel(const std::uint16_t* in, std::uint16_t* out,
                                                          std::size_t frames, std::size_t frame_elems,
                                                          const std::uint16_t* prev_frame) {
  const std::size_t groups = frame_elems / kVec;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < groups; g += stride) {
    std::size_t first = 0;
    Pack<kVec> acc{};
    if (prev_frame != nullptr) {
      acc = *reinterpret_cast<const Pack<kVec>*>(prev_frame + g * kVec);
    } else if (kWrite) {
      acc = *reinterpret_cast<const Pack<kVec>*>(in + g * kVec);
      *reinterpret_cast<Pack<kVec>*>(out + g * kVec) = acc;
      first = 1;
    }
    for (std::size_t f = first; f < frames; ++f) {
      const std::size_t idx = f * frame_elems + g * kVec;
      const Pack<kVec> d = *reinterpret_cast<const Pack<kVec>*>(in + idx);
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        acc.v[k] = static_cast<std::uint16_t>(acc.v[k] + d.v[k] + (kOffset ? 32768u : 0u));  // modulo 2**16
      }
      if (kWrite) *reinterpret_cast<Pack<kVec>*>(out + idx) = acc;
    }
    if (!kWrite) *reinterpret_cast<Pack<kVec>*>(out + g * kVec) = acc;
  }
}

// Widest access the layout allows: frames must split into whole packs and every pointer must be
// aligned to the pack size (hipMalloc memory always is).
int vector_width(std::initializer_list<const void*> pointers, std::size_t frame_elems) {
  for (int vec : {8, 4}) {
    const auto align = static_cast<std::uintptr_t>(2 * vec);
    bool aligned = frame_elems % vec == 0;
    for (const void* p : pointers) aligned &= reinterpret_cast<std::uintptr_t>(p) % align == 0;
    if (aligned) return vec;
  }
  return 1;
}

unsigned int grid_for(std::size_t frame_elems, int vec) {
  const std::size_t groups = frame_elems / static_cast<std::size_t>(vec);
  const std::size_t blocks = (groups + kThreads - 1) / kThreads;
  return static_cast<unsigned int>(std::max<std::size_t>(1, std::min<std::size_t>(blocks, 65535u * 16u)));
}

template <bool kOffset>
void launch_encode(const std::uint16_t* in, std::uint16_t* out, std::size_t frames, std::size_t frame_elems,
                   const std::uint16_t* prev, unsigned long long* clipped) {
  const int vec = vector_width({in, out, prev}, frame_elems);
  const unsigned int grid = grid_for(frame_elems, vec);
  if (vec == 8) {
    encode_kernel<kOffset, 8><<<grid, kThreads>>>(in, out, frames, frame_elems, prev, clipped);
  } else if (vec == 4) {
    encode_kernel<kOffset, 4><<<grid, kThreads>>>(in, out, frames, frame_elems, prev, clipped);
  } else {
    encode_kernel<kOffset, 1><<<grid, kThreads>>>(in, out, frames, frame_elems, prev, clipped);
  }
}

template <bool kOffset, bool kWrite>
void launch_decode(const std::uint16_t* in, std::uint16_t* out, std::size_t frames, std::size_t frame_elems,
                   const std::uint16_t* prev) {
  const int vec = vector_width({in, out, prev}, frame_elems);
  const unsigned int grid = grid_for(frame_elems, vec);
  if (vec == 8) {
    decode_kernel<kOffset, 8, kWrite><<<grid, kThreads>>>(in, out, frames, frame_elems, prev);
  } else if (vec == 4) {
    decode_kernel<kOffset, 4, kWrite><<<grid, kThreads>>>(in, out, frames, frame_elems, prev);
  } else {
    decode_kernel<kOffset, 1, kWrite><<<grid, kThreads>>>(in, out, frames, frame_elems, prev);
  }
}

// Device words for the clipped count, leased per call so concurrent host threads on one DCU
// never share one, and reused so short-lived threads do not hipMalloc (which can synchronise
// the device) on every call. Never freed: the pool lives until process exit.
class CounterLease {
public:
  CounterLease() {
    check_hip(hipGetDevice(&device_), "hipGetDevice");
    {
      std::lock_guard<std::mutex> lock(mutex());
      auto& free_list = pool()[device_];
      if (!free_list.empty()) {
        ptr_ = free_list.back();
        free_list.pop_back();
      }
    }
    if (ptr_ == nullptr) {
      check_hip(hipMalloc(reinterpret_cast<void**>(&ptr_), sizeof(*ptr_)), "hipMalloc counter");
    }
  }
  ~CounterLease() {
    std::lock_guard<std::mutex> lock(mutex());
    pool()[device_].push_back(ptr_);
  }
  CounterLease(const CounterLease&) = delete;
  CounterLease& operator=(const CounterLease&) = delete;

  unsigned long long* get() const { return ptr_; }

private:
  static std::mutex& mutex() {
    static std::mutex m;
    return m;
  }
  static std::map<int, std::vector<unsigned long long*>>& pool() {
    static auto* p = new std::map<int, std::vector<unsigned long long*>>();
    return *p;
  }

  int device_ = 0;
  unsigned long long* ptr_ = nullptr;
};

}  // namespace

std::uint64_t encode_device(const std::uint16_t* d_in, std::uint16_t* d_out, std::size_t frames,
                            std::size_t frame_elems, Encoding encoding, const std::uint16_t* d_prev) {
  if (frames == 0 || frame_elems == 0) return 0;
  CounterLease lease;
  unsigned long long* counter = lease.get();
  check_hip(hipMemsetAsync(counter, 0, sizeof(*counter), nullptr), "hipMemset counter");
  if (encoding == Encoding::kOffset) {
    launch_encode<true>(d_in, d_out, frames, frame_elems, d_prev, counter);
  } else {
    launch_encode<false>(d_in, d_out, frames, frame_elems, d_prev, counter);
  }
  check_hip(hipGetLastError(), "encode launch");
  unsigned long long clipped = 0;
  check_hip(hipMemcpy(&clipped, counter, sizeof(clipped), hipMemcpyDeviceToHost), "counter D2H");
  return clipped;
}

void decode_device(const std::uint16_t* d_in, std::uint16_t* d_out, std::size_t frames,
                   std::size_t frame_elems, Encoding encoding, const std::uint16_t* d_prev) {
  if (frames == 0 || frame_elems == 0) return;
  if (encoding == Encoding::kOffset) {
    launch_decode<true, true>(d_in, d_out, frames, frame_elems, d_prev);
  } else {
    launch_decode<false, true>(d_in, d_out, frames, frame_elems, d_prev);
  }
  check_hip(hipGetLastError(), "decode launch");
  check_hip(hipStreamSynchronize(nullptr), "decode synchronize");
}

void frame_sum_device(const std::uint16_t* d_in, std::uint16_t* d_sum, std::size_t frames,
                      std::size_t frame_elems, Encoding encoding) {
  if (frame_elems == 0) return;
  if (encoding == Encoding::kOffset) {
    launch_decode<true, false>(d_in, d_sum, frames, frame_elems, nullptr);
  } else {
    launch_decode<false, false>(d_in, d_sum, frames, frame_elems, nullptr);
  }
  check_hip(hipGetLastError(), "frame sum launch");
  check_hip(hipStreamSynchronize(nullptr), "frame sum synchronize");
}

}  // namespace delta
}  // namespace photonzip
