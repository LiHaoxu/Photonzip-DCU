#include "photonzip/codecs/pfpl/pfpl_codec.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "pfpl_pipeline.hpp"

namespace photonzip {
namespace {

enum PfplCodecParamIndex : std::size_t {
  kMode = 0,
  kErrorBoundBits = 1,
  kThresholdBits = 2,
};

struct PfplParams {
  float errorbound = 0.0f;
  float threshold = std::numeric_limits<float>::infinity();
};

float bits_to_float(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t codec_param_or(const CodecOptions& options,
                             std::size_t index,
                             std::uint32_t fallback) {
  return index < options.codec_params.size() ? options.codec_params[index] : fallback;
}

void ensure_host(const InputBuffer& input) {
  if (input.memory_kind != MemoryKind::kHost) {
    throw std::runtime_error("PFPL codec only supports host (CPU) buffers.");
  }
}

void ensure_cpu_backend(const CodecOptions& options) {
  if (options.backend != Backend::kCpu) {
    throw std::runtime_error("PFPL codec only supports the CPU backend.");
  }
}

void ensure_float32(const CodecOptions& options) {
  if (options.dtype != DataType::kFloat32) {
    throw std::runtime_error("PFPL codec only supports float32 data.");
  }
}

PfplParams to_pfpl_params(const CodecOptions& options) {
  const auto mode = codec_param_or(options, kMode, static_cast<std::uint32_t>(pfpl::Mode::kRel));
  if (mode != static_cast<std::uint32_t>(pfpl::Mode::kRel)) {
    throw std::runtime_error("Unsupported PFPL mode: " + std::to_string(mode) +
                             " (only REL is implemented).");
  }

  PfplParams params;
  params.errorbound = bits_to_float(codec_param_or(options, kErrorBoundBits, 0));
  if (!(params.errorbound > 0.0f) || !std::isfinite(params.errorbound)) {
    throw std::runtime_error("PFPL requires a positive, finite relative error bound.");
  }

  const auto threshold_bits = codec_param_or(
      options, kThresholdBits,
      [] {
        const float infinity = std::numeric_limits<float>::infinity();
        std::uint32_t bits = 0;
        std::memcpy(&bits, &infinity, sizeof(bits));
        return bits;
      }());
  params.threshold = bits_to_float(threshold_bits);
  if (!(params.threshold >= std::numeric_limits<float>::min())) {
    throw std::runtime_error("PFPL requires the threshold to be a positive, normal float.");
  }
  return params;
}

}  // namespace

CodecVTable make_pfpl_codec() {
  CodecVTable codec;
  codec.name = "pfpl";

  codec.max_compress_size = [](std::size_t input_nbytes, const CodecOptions&) -> std::size_t {
    return static_cast<std::size_t>(pfpl::max_encoded_size(static_cast<long long>(input_nbytes)));
  };

  codec.query_decompressed_size = [](const void* data, std::size_t nbytes,
                                     const CodecOptions&) -> std::size_t {
    if (data == nullptr || nbytes < static_cast<std::size_t>(pfpl::kHeaderBytes)) {
      throw std::runtime_error("PFPL compressed payload is missing its header.");
    }
    const long long decoded = pfpl::decoded_size(static_cast<const pfpl::byte*>(data));
    if (decoded < 0) {
      throw std::runtime_error("PFPL compressed payload reports a negative size.");
    }
    return static_cast<std::size_t>(decoded);
  };

  codec.compress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_cpu_backend(options);
    ensure_host(input);
    ensure_float32(options);
    const PfplParams params = to_pfpl_params(options);

    Buffer output = make_host_buffer(codec.max_compress_size(input.size, options));
    long long encoded_size = 0;
    pfpl::h_encode(static_cast<const pfpl::byte*>(input.data),
                   static_cast<long long>(input.size),
                   output.mutable_bytes(),
                   encoded_size,
                   params.errorbound,
                   params.threshold);
    if (encoded_size <= 0) {
      throw std::runtime_error("PFPL compression produced an empty payload.");
    }
    output.size = static_cast<std::size_t>(encoded_size);
    return output;
  };

  // Decompression needs no parameters: the error bound rides along in the payload header.
  codec.decompress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_cpu_backend(options);
    ensure_host(input);

    const std::size_t decoded_bytes = codec.query_decompressed_size(input.data, input.size, options);
    Buffer output = make_host_buffer(decoded_bytes);
    long long produced = 0;
    pfpl::h_decode(static_cast<const pfpl::byte*>(input.data), output.mutable_bytes(), produced);
    output.size = static_cast<std::size_t>(produced);
    return output;
  };

  codec.invoke = nullptr;
  return codec;
}

}  // namespace photonzip
