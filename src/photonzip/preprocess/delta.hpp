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

// CPU (OpenMP over the elements of a frame).
std::uint64_t encode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                          std::size_t frame_elems, Encoding encoding);
void decode_host(const std::uint16_t* in, std::uint16_t* out, std::size_t frames,
                 std::size_t frame_elems, Encoding encoding);

#ifdef PHOTONZIP_ENABLE_DCU
// DCU: device pointers, synchronous on the calling thread's default stream. Each DCU thread
// owns up to 8 consecutive elements of a frame and walks all frames, so the input is read and
// the output written once, coalesced. The parallelism is the frame size: 1-D data (one element
// per frame) runs on a single thread and belongs on the CPU.
//
// Chunks of a longer stack: with d_prev, the frames are a chunk that follows the frame d_prev,
// and frame 0 is coded relative to it, so every chunk gets exactly the bytes of the delta of the
// whole stack. encode_device takes the original previous frame; decode_device the restored one.
// Without d_prev, frame 0 is stored / taken as is.
std::uint64_t encode_device(const std::uint16_t* d_in, std::uint16_t* d_out, std::size_t frames,
                            std::size_t frame_elems, Encoding encoding,
                            const std::uint16_t* d_prev = nullptr);
void decode_device(const std::uint16_t* d_in, std::uint16_t* d_out, std::size_t frames,
                   std::size_t frame_elems, Encoding encoding, const std::uint16_t* d_prev = nullptr);

// d_sum[i] = sum over the frames of (d_in[f][i] + offset), modulo 2**16: what decoding a chunk
// of stored differences adds to the frame before it (restored last frame = previous + d_sum).
// Lets chunks be decoded in parallel: sum each chunk, chain the sums on one frame per chunk,
// then decode every chunk with its d_prev.
void frame_sum_device(const std::uint16_t* d_in, std::uint16_t* d_sum, std::size_t frames,
                      std::size_t frame_elems, Encoding encoding);
#endif

}  // namespace delta
}  // namespace photonzip
