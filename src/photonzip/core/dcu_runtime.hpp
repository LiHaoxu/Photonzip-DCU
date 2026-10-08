#pragma once

// Thin HIP runtime helpers for the DCU backend. Only available when PhotonZip is built with
// PHOTONZIP_ENABLE_DCU; the header itself has no HIP dependency.

#include <cstddef>

#include "photonzip/core/codec_types.hpp"

namespace photonzip {
namespace dcu {

// True when the HIP runtime reports at least one usable device.
bool available();

// Device allocation owned by the returned Buffer (memory_kind == MemoryKind::kDcu).
Buffer make_device_buffer(std::size_t size);

void copy_device_to_host(void* dst, const void* src, std::size_t nbytes);
void copy_host_to_device(void* dst, const void* src, std::size_t nbytes);

}  // namespace dcu
}  // namespace photonzip
