#include "photonzip/codecs/mans/mans_codec.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "dcu/mans_dcu.h"
#include "mans_defs.h"
#include "photonzip/core/dcu_runtime.hpp"

namespace photonzip {
namespace {

std::size_t bytes_per_element(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt16:
      return sizeof(std::uint16_t);
    case DataType::kUInt32:
      return sizeof(std::uint32_t);
    default:
      throw std::runtime_error("MANS supports uint16 and uint32 data only.");
  }
}

void ensure_dcu_backend(const CodecOptions& options) {
  if (options.backend != Backend::kDcu) {
    throw std::runtime_error("MANS codec only supports the DCU backend.");
  }
}

// MANS indexes elements as x + y * nx + z * nx * ny, so nx must be the fastest-varying
// dimension. PhotonZip tensors are C-contiguous, so that is the last axis: nx = shape[-1],
// ny = shape[-2], nz = shape[-3]. (Passing shape[0] as nx transposes the ADM tiles; it stays
// lossless but costs ratio, e.g. 1.96 -> 1.48 on an 8x512x512 volume chunk.)
mans::MansParams to_mans_params(const CodecOptions& options, std::size_t element_count) {
  if (options.shape.empty() || options.shape.size() > 3) {
    throw std::runtime_error("MANS requires a tensor with 1 to 3 dimensions.");
  }
  std::size_t product = 1;
  for (const auto dim : options.shape) {
    if (dim == 0) {
      throw std::runtime_error("MANS requires positive shape values.");
    }
    product *= static_cast<std::size_t>(dim);
  }
  if (element_count != 0 && product != element_count) {
    throw std::runtime_error("MANS requires the shape product to match the element count.");
  }

  mans::MansParams params;
  params.backend = mans::Backend::DCU;
  params.dtype = options.dtype == DataType::kUInt32 ? mans::DataType::U32 : mans::DataType::U16;
  params.mode = mans::Mode::P;  // the DCU backend implements P-mode only
  const std::size_t rank = options.shape.size();
  params.dims = static_cast<std::uint32_t>(rank);
  params.nx = options.shape[rank - 1];
  params.ny = rank >= 2 ? options.shape[rank - 2] : 0;
  params.nz = rank >= 3 ? options.shape[rank - 3] : 0;
  return params;
}

std::size_t element_count_for(std::size_t nbytes, const CodecOptions& options) {
  const std::size_t width = bytes_per_element(options.dtype);
  if (nbytes % width != 0) {
    throw std::runtime_error("MANS input size is not a multiple of the element size.");
  }
  return nbytes / width;
}

std::size_t decompressed_bytes(const InputBuffer& input, const mans::MansParams& params) {
  if (input.size <= mans::kMansHeaderBytes) {
    throw std::runtime_error("MANS compressed payload is missing its header or body.");
  }
  if (input.memory_kind == MemoryKind::kHost) {
    return mans::dcu::get_exact_decompress_bytes(input.data, input.size, params);
  }
  if (input.memory_kind == MemoryKind::kDcu) {
    // get_exact_decompress_bytes only parses the header, but it checks that a body follows.
    std::uint8_t header[mans::kMansHeaderBytes + 1] = {};
    dcu::copy_device_to_host(header, input.data, mans::kMansHeaderBytes);
    return mans::dcu::get_exact_decompress_bytes(header, input.size, params);
  }
  throw std::runtime_error("Unsupported input memory kind for MANS.");
}

}  // namespace

CodecVTable make_mans_codec() {
  CodecVTable codec;
  codec.name = "mans";

  codec.max_compress_size = [](std::size_t input_nbytes, const CodecOptions& options) -> std::size_t {
    const std::size_t n = element_count_for(input_nbytes, options);
    return mans::dcu::get_max_compress_bytes(n, to_mans_params(options, n));
  };

  codec.query_decompressed_size = [](const void* data, std::size_t nbytes,
                                     const CodecOptions& options) -> std::size_t {
    return mans::dcu::get_exact_decompress_bytes(data, nbytes, to_mans_params(options, 0));
  };

  codec.compress = [](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_dcu_backend(options);
    const std::size_t n = element_count_for(input.size, options);
    const mans::MansParams params = to_mans_params(options, n);
    const std::size_t capacity = mans::dcu::get_max_compress_bytes(n, params);

    Buffer output;
    if (input.memory_kind == MemoryKind::kDcu) {
      output = dcu::make_device_buffer(capacity);
      std::size_t out_size = capacity;
      mans::dcu::compress_internal_device(input.data, n, params, output.mutable_bytes(), out_size);
      output.size = out_size;
    } else if (input.memory_kind == MemoryKind::kHost) {
      output = make_host_buffer(capacity);
      std::size_t out_size = capacity;
      mans::dcu::compress_internal(input.data, n, params, output.mutable_bytes(), out_size);
      output.size = out_size;
    } else {
      throw std::runtime_error("Unsupported input memory kind for MANS.");
    }
    if (output.size == 0) {
      throw std::runtime_error("MANS compression produced an empty payload.");
    }
    return output;
  };

  codec.decompress = [](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    ensure_dcu_backend(options);
    // The stream header carries the geometry, so only dtype and backend matter here.
    mans::MansParams params;
    params.backend = mans::Backend::DCU;
    params.dtype = options.dtype == DataType::kUInt32 ? mans::DataType::U32 : mans::DataType::U16;
    params.mode = mans::Mode::P;

    const std::size_t raw_bytes = decompressed_bytes(input, params);
    Buffer output;
    std::size_t out_size = raw_bytes;
    if (input.memory_kind == MemoryKind::kDcu) {
      output = dcu::make_device_buffer(raw_bytes);
      mans::dcu::decompress_internal_device(input.data, input.size, params, output.mutable_bytes(), out_size);
    } else {
      output = make_host_buffer(raw_bytes);
      mans::dcu::decompress_internal(input.data, input.size, params, output.mutable_bytes(), out_size);
    }
    if (out_size != raw_bytes) {
      throw std::runtime_error("MANS decompression size mismatch.");
    }
    output.size = out_size;
    return output;
  };

  codec.invoke = nullptr;
  return codec;
}

}  // namespace photonzip
