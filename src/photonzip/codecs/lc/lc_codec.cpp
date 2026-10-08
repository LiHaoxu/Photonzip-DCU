#include "photonzip/codecs/lc/lc_codec.hpp"

#include <cstddef>
#include <stdexcept>

#include "lc_pipeline.hpp"

#ifdef PHOTONZIP_ENABLE_DCU
#include "photonzip/codecs/lc/lc_dcu.hpp"
#include "photonzip/core/dcu_runtime.hpp"
#endif

namespace photonzip {
namespace {

void ensure_host(const InputBuffer& input) {
  if (input.memory_kind != MemoryKind::kHost) {
    throw std::runtime_error("LC CPU backend only supports host buffers.");
  }
}

void ensure_supported_backend(const CodecOptions& options) {
  if (options.backend == Backend::kCpu) {
    return;
  }
#ifdef PHOTONZIP_ENABLE_DCU
  if (options.backend == Backend::kDcu) {
    return;
  }
  throw std::runtime_error("LC codec supports the CPU and DCU backends.");
#else
  throw std::runtime_error("LC codec only supports the CPU backend (built without DCU support).");
#endif
}

long long read_header(const InputBuffer& input) {
  if (input.data == nullptr || input.size < sizeof(long long)) {
    throw std::runtime_error("LC compressed payload is missing its header.");
  }
#ifdef PHOTONZIP_ENABLE_DCU
  if (input.memory_kind == MemoryKind::kDcu) {
    return lc::dcu::decoded_size_device(static_cast<const lc::byte*>(input.data));
  }
#endif
  if (input.memory_kind != MemoryKind::kHost) {
    throw std::runtime_error("Unsupported input memory kind for LC.");
  }
  return lc::decoded_size(static_cast<const lc::byte*>(input.data));
}

#ifdef PHOTONZIP_ENABLE_DCU
// DCU backend: host input yields a host payload, device input yields a device payload.
Buffer compress_dcu(const InputBuffer& input, std::size_t capacity) {
  const auto* src = static_cast<const lc::byte*>(input.data);
  const auto insize = static_cast<long long>(input.size);
  if (input.memory_kind == MemoryKind::kDcu) {
    Buffer output = dcu::make_device_buffer(capacity);
    output.size = static_cast<std::size_t>(lc::dcu::encode_device(src, insize, output.mutable_bytes()));
    return output;
  }
  if (input.memory_kind == MemoryKind::kHost) {
    Buffer output = make_host_buffer(capacity);
    output.size = static_cast<std::size_t>(lc::dcu::encode_host(src, insize, output.mutable_bytes()));
    return output;
  }
  throw std::runtime_error("Unsupported input memory kind for the LC DCU backend.");
}

Buffer decompress_dcu(const InputBuffer& input, std::size_t decoded_bytes) {
  const auto* src = static_cast<const lc::byte*>(input.data);
  const auto capacity = static_cast<long long>(decoded_bytes);
  if (input.memory_kind == MemoryKind::kDcu) {
    Buffer output = dcu::make_device_buffer(decoded_bytes);
    output.size = static_cast<std::size_t>(lc::dcu::decode_device(src, output.mutable_bytes(), capacity));
    return output;
  }
  if (input.memory_kind == MemoryKind::kHost) {
    Buffer output = make_host_buffer(decoded_bytes);
    output.size = static_cast<std::size_t>(lc::dcu::decode_host(
        src, static_cast<long long>(input.size), output.mutable_bytes(), capacity));
    return output;
  }
  throw std::runtime_error("Unsupported input memory kind for the LC DCU backend.");
}
#endif

}  // namespace

CodecVTable make_lc_codec() {
  CodecVTable codec;
  codec.name = "lc";

  codec.max_compress_size = [](std::size_t input_nbytes, const CodecOptions&) -> std::size_t {
    return static_cast<std::size_t>(lc::max_encoded_size(static_cast<long long>(input_nbytes)));
  };

  codec.query_decompressed_size = [](const void* data, std::size_t nbytes,
                                     const CodecOptions&) -> std::size_t {
    if (data == nullptr || nbytes < sizeof(long long)) {
      throw std::runtime_error("LC compressed payload is missing its header.");
    }
    const long long decoded = lc::decoded_size(static_cast<const lc::byte*>(data));
    if (decoded < 0) {
      throw std::runtime_error("LC compressed payload reports a negative size.");
    }
    return static_cast<std::size_t>(decoded);
  };

  codec.compress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_supported_backend(options);
    const std::size_t capacity = codec.max_compress_size(input.size, options);

#ifdef PHOTONZIP_ENABLE_DCU
    if (options.backend == Backend::kDcu) {
      Buffer output = compress_dcu(input, capacity);
      if (output.size == 0) {
        throw std::runtime_error("LC compression produced an empty payload.");
      }
      return output;
    }
#endif

    ensure_host(input);
    Buffer output = make_host_buffer(capacity);
    long long encoded_size = 0;
    lc::h_encode(static_cast<const lc::byte*>(input.data),
                 static_cast<long long>(input.size),
                 output.mutable_bytes(),
                 encoded_size);
    if (encoded_size <= 0) {
      throw std::runtime_error("LC compression produced an empty payload.");
    }
    output.size = static_cast<std::size_t>(encoded_size);
    return output;
  };

  codec.decompress = [](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_supported_backend(options);
    const long long decoded = read_header(input);
    if (decoded < 0) {
      throw std::runtime_error("LC compressed payload reports a negative size.");
    }
    const auto decoded_bytes = static_cast<std::size_t>(decoded);

#ifdef PHOTONZIP_ENABLE_DCU
    if (options.backend == Backend::kDcu) {
      return decompress_dcu(input, decoded_bytes);
    }
#endif

    ensure_host(input);
    Buffer output = make_host_buffer(decoded_bytes);
    long long produced = 0;
    lc::h_decode(static_cast<const lc::byte*>(input.data), output.mutable_bytes(), produced);
    output.size = static_cast<std::size_t>(produced);
    return output;
  };

  codec.invoke = nullptr;
  return codec;
}

}  // namespace photonzip
