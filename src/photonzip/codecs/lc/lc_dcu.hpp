#pragma once

// DCU (HIP) implementation of the fixed LC pipeline DIFFMS_2 + BIT_2 + RZE_2.
//
// The kernels come from the upstream LC framework's HIP code generator
// (HIPversion/generate_standalone_GPU_compressor_decompressor.py) and produce the same
// bitstream as the host pipeline in lc_pipeline.hpp, so data compressed on the DCU can be
// decompressed on the CPU and vice versa.
//
// This header has no HIP dependency so it can be included from plain C++ translation units.
// All functions are synchronous and throw std::runtime_error on failure.

namespace photonzip {
namespace lc {
namespace dcu {

using byte = unsigned char;

// True when the HIP runtime reports at least one usable device.
bool device_available();

// Device-resident entry points. `d_output` must hold max_encoded_size(insize) bytes for
// encoding, and `out_capacity` bytes (at least the decoded size) for decoding.
long long encode_device(const byte* d_input, long long insize, byte* d_output);
long long decode_device(const byte* d_input, byte* d_output, long long out_capacity);

// Reads the decoded size from the header of a device-resident payload.
long long decoded_size_device(const byte* d_input);

// Host-resident wrappers that stage the data through cached device buffers.
long long encode_host(const byte* input, long long insize, byte* output);
long long decode_host(const byte* input, long long insize, byte* output, long long out_capacity);

}  // namespace dcu
}  // namespace lc
}  // namespace photonzip
