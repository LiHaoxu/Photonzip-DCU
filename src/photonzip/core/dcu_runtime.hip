#include "photonzip/core/dcu_runtime.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace photonzip {
namespace dcu {
namespace {

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("DCU: ") + what + ": " + hipGetErrorString(status));
  }
}

}  // namespace

bool available() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

Buffer make_device_buffer(std::size_t size) {
  Buffer buffer;
  buffer.size = size;
  buffer.memory_kind = MemoryKind::kDcu;
  if (size == 0) {
    return buffer;
  }
  void* ptr = nullptr;
  check_hip(hipMalloc(&ptr, size), "hipMalloc");
  buffer.data = std::shared_ptr<void>(ptr, [](void* p) { (void)hipFree(p); });
  return buffer;
}

void copy_device_to_host(void* dst, const void* src, std::size_t nbytes) {
  if (nbytes != 0) {
    check_hip(hipMemcpy(dst, src, nbytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
  }
}

void copy_host_to_device(void* dst, const void* src, std::size_t nbytes) {
  if (nbytes != 0) {
    check_hip(hipMemcpy(dst, src, nbytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
  }
}

}  // namespace dcu
}  // namespace photonzip
