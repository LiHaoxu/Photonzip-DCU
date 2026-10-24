#pragma once

// Inter-frame delta along the slowest axis of a uint16 array of `frames` frames with
// `frame_elems` elements each (C order). Frame 0 is stored unchanged; frame i stores
// clip(x[i] - x[i-1], -32768, 32767) as a uint16:
//   kInt16  - two's complement bit pattern (-1 -> 65535), used with LC;
//   kOffset - difference + 32768 (-1 -> 32767), keeps the numeric order MANS's ADM needs.
// The bytes are identical to photonzip.preprocess.apply_delta / invert_delta. Decoding is a
// running sum modulo 2**16, so it inverts the encoding exactly when nothing was clipped.
//
// All functions accept in == out (in-place). The encoders return the number of clipped
// elements; a non-zero count means the encoding is lossy.

#include <cstddef>
#include <cstdint>

namespace photonzip {
namespace delta {

enum class Encoding : std::uint32_t {
  kNone = 0,
  kInt16 = 1,
  kOffset = 2,
};

// OpenMP over the elements of a frame.
std::uint64_t encode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                          std::size_t frame_elems, Encoding encoding);
void decode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                 std::size_t frame_elems, Encoding encoding);

}  // namespace delta
}  // namespace photonzip
