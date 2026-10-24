#include <H5PLextern.h>
#include <hdf5.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>

#include "photonzip/codecs/pfpl/pfpl_pipeline.hpp"
#include "photonzip_h5z_ids.h"

// =========================================================
// H5Z-PFPL: error-bounded lossy PFPL pipeline QREL_4 + DIFFNB_4 + BIT_4 + RZE_1
// =========================================================
//
// The quantiser works on the raw float32 bit patterns, so the filter only accepts 4-byte
// little-endian floats. Parameters (cd_values):
//
//   [0] mode, 0 = REL (relative error bound); the only mode implemented so far
//   [1] error bound, as the bit pattern of a float
//   [2] threshold, as the bit pattern of a float; optional, defaults to +infinity
//
// The mode has to be recorded here because the payload header only carries the decoded size
// and the error bound, so a file written today stays readable when further modes are added.

namespace {

using photonzip::pfpl::byte;

constexpr size_t kModeIndex = 0;
constexpr size_t kErrorBoundIndex = 1;
constexpr size_t kThresholdIndex = 2;
constexpr size_t kRequiredParams = 2;

float bits_to_float(unsigned int bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool parse_params(size_t cd_nelmts, const unsigned int cd_values[], float& errorbound,
                  float& threshold) {
  if (!cd_values || cd_nelmts < kRequiredParams) {
    std::cerr << "[H5Z-PFPL Error] Filter parameter count (" << cd_nelmts << ") must be at least "
              << kRequiredParams << " (mode, error bound).\n";
    return false;
  }
  const unsigned int mode = cd_values[kModeIndex];
  if (mode != static_cast<unsigned int>(photonzip::pfpl::Mode::kRel)) {
    std::cerr << "[H5Z-PFPL Error] Unsupported mode " << mode << " (only REL is implemented).\n";
    return false;
  }

  errorbound = bits_to_float(cd_values[kErrorBoundIndex]);
  if (!(errorbound > 0.0f) || !std::isfinite(errorbound)) {
    std::cerr << "[H5Z-PFPL Error] The relative error bound must be positive and finite.\n";
    return false;
  }

  threshold = (cd_nelmts > kThresholdIndex) ? bits_to_float(cd_values[kThresholdIndex])
                                            : std::numeric_limits<float>::infinity();
  if (!(threshold >= std::numeric_limits<float>::min())) {
    std::cerr << "[H5Z-PFPL Error] The threshold must be a positive, normal float.\n";
    return false;
  }
  return true;
}

htri_t H5Z_can_apply_pfpl(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  (void)dcpl_id;
  (void)space_id;

  if (H5Tget_class(type_id) != H5T_FLOAT) {
    std::cerr << "[H5Z-PFPL Error] Datatype is not FLOAT.\n";
    return 0;
  }
  if (H5Tget_size(type_id) != 4) {
    std::cerr << "[H5Z-PFPL Error] Only single-precision floats are supported.\n";
    return 0;
  }
  if (H5Tget_order(type_id) != H5T_ORDER_LE) {
    std::cerr << "[H5Z-PFPL Error] Only little-endian floats are supported.\n";
    return 0;
  }
  return 1;
}

herr_t H5Z_set_local_pfpl(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  (void)type_id;
  (void)space_id;

  unsigned int flags = 0;
  size_t cd_nelmts = 8;
  unsigned int cd_values[8] = {0};
  if (H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_PHOTONZIP_PFPL_ID, &flags, &cd_nelmts, cd_values,
                           0, nullptr, nullptr) < 0) {
    std::cerr << "[H5Z-PFPL Error] Failed to read the filter parameters.\n";
    return -1;
  }

  float errorbound = 0.0f;
  float threshold = 0.0f;
  return parse_params(cd_nelmts, cd_values, errorbound, threshold) ? 0 : -1;
}

size_t H5Z_filter_pfpl(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
                       size_t nbytes, size_t* buf_size, void** buf) {
  if (!buf || !*buf || nbytes == 0) {
    return 0;
  }

  float errorbound = 0.0f;
  float threshold = 0.0f;
  if (!parse_params(cd_nelmts, cd_values, errorbound, threshold)) {
    return 0;
  }

  void* dst_buf = nullptr;
  try {
    size_t dst_capacity = 0;
    long long produced = 0;

    if (flags & H5Z_FLAG_REVERSE) {
      if (nbytes < static_cast<size_t>(photonzip::pfpl::kHeaderBytes)) {
        std::cerr << "[H5Z-PFPL Error] Compressed chunk is missing its header.\n";
        return 0;
      }
      const long long decoded = photonzip::pfpl::decoded_size(static_cast<const byte*>(*buf));
      if (decoded <= 0) {
        std::cerr << "[H5Z-PFPL Error] Compressed chunk reports a non-positive decoded size.\n";
        return 0;
      }
      dst_capacity = static_cast<size_t>(decoded);
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
      photonzip::pfpl::h_decode(static_cast<const byte*>(*buf), static_cast<byte*>(dst_buf),
                               produced);
    } else {
      if (nbytes % sizeof(float) != 0) {
        std::cerr << "[H5Z-PFPL Error] Chunk size (" << nbytes
                  << ") is not a multiple of 4 bytes.\n";
        return 0;
      }
      dst_capacity = static_cast<size_t>(
          photonzip::pfpl::max_encoded_size(static_cast<long long>(nbytes)));
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
      photonzip::pfpl::h_encode(static_cast<const byte*>(*buf), static_cast<long long>(nbytes),
                                static_cast<byte*>(dst_buf), produced, errorbound, threshold);
    }

    if (produced <= 0 || static_cast<size_t>(produced) > dst_capacity) {
      std::cerr << "[H5Z-PFPL Error] Codec produced " << produced << " bytes for a "
                << dst_capacity << " byte buffer.\n";
      std::free(dst_buf);
      return 0;
    }

    std::free(*buf);
    *buf = dst_buf;
    *buf_size = dst_capacity;
    return static_cast<size_t>(produced);
  } catch (const std::exception& e) {
    std::cerr << "[H5Z-PFPL Error] " << e.what() << "\n";
    std::free(dst_buf);
    return 0;
  } catch (...) {
    std::cerr << "[H5Z-PFPL Error] Unknown exception.\n";
    std::free(dst_buf);
    return 0;
  }
}

}  // namespace

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_PFPL_CLASS[1] = {{
    H5Z_CLASS_T_VERS,
    H5Z_FILTER_PHOTONZIP_PFPL_ID,
    1,
    1,
    "H5Z-PFPL",
    H5Z_can_apply_pfpl,
    H5Z_set_local_pfpl,
    H5Z_filter_pfpl,
}};

H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }
const void* H5PLget_plugin_info(void) { return H5Z_PFPL_CLASS; }
