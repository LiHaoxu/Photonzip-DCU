#include <H5PLextern.h>
#include <hdf5.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "dcu/mans_dcu.h"
#include "mans_defs.h"
#include "photonzip_h5z_ids.h"

// =========================================================
// H5Z-MANS: lossless MANS (ADM + ANS, P-mode) on the DCU
// =========================================================
//
// Accepts uint16 and uint32 datasets. The geometry passed to ADM is taken from the HDF5
// chunk shape in set_local, following the upstream H5Z-MANS plugin:
//   1D chunk  -> nx
//   2D chunk  -> nx, ny
//   3D+ chunk -> nx, ny, nz = product of the remaining chunk dims
//
// Public parameters (compression_opts): none, or a single mode word that must be 0 (P-mode);
// the DCU backend does not implement R-mode. set_local expands them to the upstream layout
// [MansParams words..., chunk_elements], storing backend=CPU and mode=P so that the upstream
// CPU plugin can read the dataset. The filter itself always runs on the DCU.

namespace {

constexpr std::size_t kMansParamsWords = sizeof(mans::MansParams) / sizeof(unsigned int);
constexpr std::uint32_t kDefaultAdmThreads = 32;  // only meaningful to the CPU plugin

bool derive_dtype(hid_t type_id, std::uint32_t& dtype, std::string& error) {
  if (H5Tget_class(type_id) != H5T_INTEGER) {
    error = "datatype is not INTEGER";
    return false;
  }
  if (H5Tget_sign(type_id) != H5T_SGN_NONE) {
    error = "datatype must be unsigned";
    return false;
  }
  const size_t size = H5Tget_size(type_id);
  if (size == 2) {
    dtype = mans::DataType::U16;
    return true;
  }
  if (size == 4) {
    dtype = mans::DataType::U32;
    return true;
  }
  error = "only 2-byte (uint16) or 4-byte (uint32) integers are supported";
  return false;
}

bool apply_chunk_dims(const std::vector<hsize_t>& chunk_dims, mans::MansParams& params,
                      std::string& error) {
  const auto u32_max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
  auto to_u32 = [&](std::uint64_t v, const char* name, std::uint32_t& out) -> bool {
    if (v == 0 || v > u32_max) {
      error = std::string("invalid ") + name + " in chunk dims";
      return false;
    }
    out = static_cast<std::uint32_t>(v);
    return true;
  };

  if (chunk_dims.empty()) {
    error = "chunk dims are empty";
    return false;
  }
  params.dims = static_cast<std::uint32_t>(chunk_dims.size() < 3 ? chunk_dims.size() : 3);
  params.nx = params.ny = params.nz = 0;
  if (!to_u32(chunk_dims[0], "nx", params.nx)) {
    return false;
  }
  if (chunk_dims.size() >= 2 && !to_u32(chunk_dims[1], "ny", params.ny)) {
    return false;
  }
  if (chunk_dims.size() >= 3) {
    std::uint64_t merged_z = 1;
    for (std::size_t i = 2; i < chunk_dims.size(); ++i) {
      const auto d = static_cast<std::uint64_t>(chunk_dims[i]);
      if (d == 0 || merged_z > u32_max / d) {
        error = "invalid nz in chunk dims";
        return false;
      }
      merged_z *= d;
    }
    if (!to_u32(merged_z, "nz", params.nz)) {
      return false;
    }
  }
  return true;
}

std::size_t geometry_elements(const mans::MansParams& p) {
  std::size_t n = p.nx;
  if (p.dims >= 2) n *= p.ny;
  if (p.dims >= 3) n *= p.nz;
  return n;
}

htri_t H5Z_can_apply_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  (void)dcpl_id;
  (void)space_id;
  std::uint32_t dtype = 0;
  std::string error;
  if (!derive_dtype(type_id, dtype, error)) {
    std::cerr << "[H5Z-MANS Error] " << error << ".\n";
    return 0;
  }
  return 1;
}

herr_t H5Z_set_local_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  if (H5Pget_layout(dcpl_id) != H5D_CHUNKED) {
    std::cerr << "[H5Z-MANS Error] The dataset must be chunked.\n";
    return -1;
  }

  unsigned int flags = 0;
  size_t cd_nelmts = 8;
  unsigned int cd_values[8] = {};
  if (H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_PHOTONZIP_MANS_ID, &flags, &cd_nelmts, cd_values,
                           0, nullptr, nullptr) < 0) {
    return -1;
  }
  // Accept the public form (no value or a single mode word) and an already expanded set.
  if (cd_nelmts >= 1 && cd_values[0] != mans::Mode::P && cd_nelmts < kMansParamsWords) {
    std::cerr << "[H5Z-MANS Error] Only P-mode (compression_opts=(0,)) is supported on the DCU.\n";
    return -1;
  }

  mans::MansParams params{};
  params.backend = mans::Backend::CPU;  // stored for compatibility with the upstream plugin
  params.mode = mans::Mode::P;
  params.adm_compress_thread = kDefaultAdmThreads;
  params.adm_decompress_thread = kDefaultAdmThreads;
  std::string error;
  if (!derive_dtype(type_id, params.dtype, error)) {
    std::cerr << "[H5Z-MANS Error] " << error << ".\n";
    return -1;
  }

  const int ndims = H5Sget_simple_extent_ndims(space_id);
  if (ndims <= 0) {
    return -1;
  }
  std::vector<hsize_t> chunk_dims(static_cast<std::size_t>(ndims), 0);
  if (H5Pget_chunk(dcpl_id, ndims, chunk_dims.data()) < 0) {
    return -1;
  }
  if (!apply_chunk_dims(chunk_dims, params, error)) {
    std::cerr << "[H5Z-MANS Error] " << error << ".\n";
    return -1;
  }

  std::vector<unsigned int> values(kMansParamsWords + 1, 0);
  std::memcpy(values.data(), &params, sizeof(params));
  values[kMansParamsWords] = static_cast<unsigned int>(geometry_elements(params));
  if (H5Pmodify_filter(dcpl_id, H5Z_FILTER_PHOTONZIP_MANS_ID, flags, values.size(),
                       values.data()) < 0) {
    return -1;
  }
  return 0;
}

size_t H5Z_filter_mans(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
                       size_t nbytes, size_t* buf_size, void** buf) {
  if (!buf || !*buf || nbytes == 0) {
    return 0;
  }
  if (cd_nelmts < kMansParamsWords) {
    std::cerr << "[H5Z-MANS Error] Expected " << kMansParamsWords << " filter parameters, got "
              << cd_nelmts << ".\n";
    return 0;
  }
  mans::MansParams params{};
  std::memcpy(&params, cd_values, sizeof(params));
  if (params.mode != mans::Mode::P) {
    std::cerr << "[H5Z-MANS Error] Dataset uses MANS R-mode, which the DCU backend cannot decode.\n";
    return 0;
  }
  params.backend = mans::Backend::DCU;
  const std::size_t elem_size = params.dtype == mans::DataType::U32 ? 4 : 2;

  void* dst_buf = nullptr;
  try {
    size_t dst_capacity = 0;
    size_t produced = 0;

    if (flags & H5Z_FLAG_REVERSE) {
      dst_capacity = mans::dcu::get_exact_decompress_bytes(*buf, nbytes, params);
      if (dst_capacity == 0 || dst_capacity % elem_size != 0) {
        std::cerr << "[H5Z-MANS Error] Invalid decompressed size " << dst_capacity << ".\n";
        return 0;
      }
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
      produced = dst_capacity;
      mans::dcu::decompress_internal(*buf, nbytes, params, static_cast<std::uint8_t*>(dst_buf),
                                     produced);
    } else {
      if (nbytes % elem_size != 0) {
        std::cerr << "[H5Z-MANS Error] Chunk size " << nbytes
                  << " is not a multiple of the element size.\n";
        return 0;
      }
      const std::size_t n = nbytes / elem_size;
      if (n != geometry_elements(params)) {
        // Partial chunk: fall back to a 1D layout so the header records the exact geometry.
        params.dims = 1;
        params.nx = static_cast<std::uint32_t>(n);
        params.ny = params.nz = 0;
      }
      dst_capacity = mans::dcu::get_max_compress_bytes(n, params);
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
      produced = dst_capacity;
      mans::dcu::compress_internal(*buf, n, params, static_cast<std::uint8_t*>(dst_buf), produced);
    }

    if (produced == 0 || produced > dst_capacity) {
      std::cerr << "[H5Z-MANS Error] Codec produced " << produced << " bytes for a "
                << dst_capacity << " byte buffer.\n";
      std::free(dst_buf);
      return 0;
    }

    std::free(*buf);
    *buf = dst_buf;
    *buf_size = dst_capacity;
    return produced;
  } catch (const std::exception& e) {
    std::cerr << "[H5Z-MANS Error] " << e.what() << "\n";
    std::free(dst_buf);
    return 0;
  } catch (...) {
    std::cerr << "[H5Z-MANS Error] Unknown exception.\n";
    std::free(dst_buf);
    return 0;
  }
}

}  // namespace

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_MANS_CLASS[1] = {{
    H5Z_CLASS_T_VERS,
    H5Z_FILTER_PHOTONZIP_MANS_ID,
    1,
    1,
    "H5Z-MANS",
    H5Z_can_apply_mans,
    H5Z_set_local_mans,
    H5Z_filter_mans,
}};

H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }
const void* H5PLget_plugin_info(void) { return H5Z_MANS_CLASS; }
