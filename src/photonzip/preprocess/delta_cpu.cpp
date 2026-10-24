#include "photonzip/preprocess/delta.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace photonzip {
namespace delta {
namespace {

inline std::uint16_t store(int difference, Encoding encoding, std::uint64_t& clipped) {
  if (difference < -32768 || difference > 32767) {
    ++clipped;
    difference = difference < 0 ? -32768 : 32767;
  }
  return encoding == Encoding::kOffset ? static_cast<std::uint16_t>(difference + 32768)
                                       : static_cast<std::uint16_t>(difference);  // two's complement bits
}

// This thread's share [begin, end) of the elements of a frame.
std::pair<std::size_t, std::size_t> thread_range(std::size_t frame_elems) {
#ifdef _OPENMP
  const auto threads = static_cast<std::size_t>(omp_get_num_threads());
  const auto thread = static_cast<std::size_t>(omp_get_thread_num());
#else
  const std::size_t threads = 1, thread = 0;
#endif
  const std::size_t share = (frame_elems + threads - 1) / threads;
  const std::size_t begin = std::min(frame_elems, thread * share);
  return {begin, std::min(frame_elems, begin + share)};
}

}  // namespace

// Each thread owns a range of columns (element positions within a frame) and walks every
// frame for it: one parallel region whatever the shape, and the frame-to-frame dependency of
// the decoder stays inside a thread.
std::uint64_t encode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                          std::size_t frame_elems, Encoding encoding) {
  if (frames == 0 || frame_elems == 0) return 0;
  std::uint64_t clipped = 0;
#pragma omp parallel reduction(+ : clipped)
  {
    const auto [begin, end] = thread_range(frame_elems);
    // Last frame first, so in-place encoding still reads the original previous frame.
    for (std::size_t f = frames; f-- > 1;) {
      const std::uint16_t* cur = in + f * frame_elems;
      const std::uint16_t* prev = cur - frame_elems;
      std::uint16_t* dst = out + f * frame_elems;
      for (std::size_t i = begin; i < end; ++i) {
        dst[i] = store(static_cast<int>(cur[i]) - static_cast<int>(prev[i]), encoding, clipped);
      }
    }
    if (out != in) {
      for (std::size_t i = begin; i < end; ++i) out[i] = in[i];
    }
  }
  return clipped;
}

void decode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                 std::size_t frame_elems, Encoding encoding) {
  if (frames == 0 || frame_elems == 0) return;
  const std::uint16_t offset = encoding == Encoding::kOffset ? 32768 : 0;
#pragma omp parallel
  {
    const auto [begin, end] = thread_range(frame_elems);
    if (out != in) {
      for (std::size_t i = begin; i < end; ++i) out[i] = in[i];
    }
    for (std::size_t f = 1; f < frames; ++f) {
      const std::uint16_t* d = in + f * frame_elems;
      const std::uint16_t* prev = out + (f - 1) * frame_elems;
      std::uint16_t* dst = out + f * frame_elems;
      for (std::size_t i = begin; i < end; ++i) {
        dst[i] = static_cast<std::uint16_t>(prev[i] + d[i] + offset);  // modulo 2**16
      }
    }
  }
}

}  // namespace delta
}  // namespace photonzip
