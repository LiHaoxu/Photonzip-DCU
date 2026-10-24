#include "photonzip/codecs/lc/lc_codec.hpp"

#include <cstddef>
#include <stdexcept>

#include "lc_pipeline.hpp"

namespace photonzip {
namespace {

void ensure_host(const InputBuffer& input) {
  if (input.memory_kind != MemoryKind::kHost) {
    throw std::runtime_error("LC codec only supports host (CPU) buffers.");
  }
}

void ensure_cpu_backend(const CodecOptions& options) {
  if (options.backend != Backend::kCpu) {
    throw std::runtime_error("LC codec only supports the CPU backend.");
  }
}

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
    ensure_cpu_backend(options);
    ensure_host(input);

    Buffer output = make_host_buffer(codec.max_compress_size(input.size, options));
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

  codec.decompress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_cpu_backend(options);
    ensure_host(input);

    const std::size_t decoded_bytes = codec.query_decompressed_size(input.data, input.size, options);
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
