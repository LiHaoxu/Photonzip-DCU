#include <H5PLextern.h>
#include <hdf5.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "photonzip/codecs/lc/lc_pipeline.hpp"
#include "photonzip_h5z_ids.h"

#ifdef PHOTONZIP_ENABLE_DCU
#include "photonzip/codecs/lc/lc_dcu.hpp"
#endif

// =========================================================
// H5Z-LC: lossless LC pipeline DIFFMS_2 + BIT_2 + RZE_2
// =========================================================
//
// The pipeline components operate on 2-byte words, so the filter only accepts 16-bit
// unsigned data. It takes no parameters: the decoded size travels in the payload header,
// which makes the compressed chunks self-describing.
//
// When built with PHOTONZIP_ENABLE_DCU the chunks are (de)compressed on the DCU. The CPU and
// DCU pipelines produce the same bitstream, so the backend is a runtime choice that does not
// affect the file. PHOTONZIP_LC_BACKEND selects it: "dcu", "cpu" or "auto" (default: DCU when
// a device is present, CPU otherwise).

namespace {

using photonzip::lc::byte;

bool use_dcu() {
#ifdef PHOTONZIP_ENABLE_DCU
  static const bool enabled = []() {
    const char* env = std::getenv("PHOTONZIP_LC_BACKEND");
    const std::string choice = (env && env[0] != '\0') ? env : "auto";
    if (choice == "cpu") {
      return false;
    }
    const bool available = photonzip::lc::dcu::device_available();
    if (choice == "dcu" && !available) {
      std::cerr << "[H5Z-LC Warning] PHOTONZIP_LC_BACKEND=dcu but no DCU is available; using the CPU.\n";
    } else if (choice != "dcu" && choice != "auto") {
      std::cerr << "[H5Z-LC Warning] Unknown PHOTONZIP_LC_BACKEND=" << choice << "; using auto.\n";
    }
    return available;
  }();
  return enabled;
#else
  return false;
#endif
}

htri_t H5Z_can_apply_lc(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  (void)dcpl_id;
  (void)space_id;

  if (H5Tget_class(type_id) != H5T_INTEGER) {
    std::cerr << "[H5Z-LC Error] Datatype is not INTEGER.\n";
    return 0;
  }
  if (H5Tget_sign(type_id) != H5T_SGN_NONE) {
    std::cerr << "[H5Z-LC Error] Datatype must be unsigned.\n";
    return 0;
  }
  if (H5Tget_size(type_id) != 2) {
    std::cerr << "[H5Z-LC Error] The LC pipeline is specialised for 16-bit words.\n";
    return 0;
  }
  return 1;
}

herr_t H5Z_set_local_lc(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
  (void)dcpl_id;
  (void)type_id;
  (void)space_id;
  return 0;
}

size_t H5Z_filter_lc(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
                     size_t nbytes, size_t* buf_size, void** buf) {
  (void)cd_nelmts;
  (void)cd_values;

  if (!buf || !*buf || nbytes == 0) {
    return 0;
  }

  void* dst_buf = nullptr;
  try {
    size_t dst_capacity = 0;
    long long produced = 0;

    if (flags & H5Z_FLAG_REVERSE) {
      const long long decoded = photonzip::lc::decoded_size(static_cast<const byte*>(*buf));
      if (decoded <= 0) {
        std::cerr << "[H5Z-LC Error] Compressed chunk reports a non-positive decoded size.\n";
        return 0;
      }
      dst_capacity = static_cast<size_t>(decoded);
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
#ifdef PHOTONZIP_ENABLE_DCU
      if (use_dcu()) {
        produced = photonzip::lc::dcu::decode_host(static_cast<const byte*>(*buf),
                                                   static_cast<long long>(nbytes),
                                                   static_cast<byte*>(dst_buf), decoded);
      } else
#endif
      {
        photonzip::lc::h_decode(static_cast<const byte*>(*buf), static_cast<byte*>(dst_buf), produced);
      }
    } else {
      dst_capacity = static_cast<size_t>(
          photonzip::lc::max_encoded_size(static_cast<long long>(nbytes)));
      dst_buf = std::malloc(dst_capacity);
      if (!dst_buf) {
        return 0;
      }
#ifdef PHOTONZIP_ENABLE_DCU
      if (use_dcu()) {
        produced = photonzip::lc::dcu::encode_host(static_cast<const byte*>(*buf),
                                                   static_cast<long long>(nbytes),
                                                   static_cast<byte*>(dst_buf));
      } else
#endif
      {
        photonzip::lc::h_encode(static_cast<const byte*>(*buf), static_cast<long long>(nbytes),
                                static_cast<byte*>(dst_buf), produced);
      }
    }

    if (produced <= 0 || static_cast<size_t>(produced) > dst_capacity) {
      std::cerr << "[H5Z-LC Error] Codec produced " << produced << " bytes for a "
                << dst_capacity << " byte buffer.\n";
      std::free(dst_buf);
      return 0;
    }

    std::free(*buf);
    *buf = dst_buf;
    *buf_size = dst_capacity;
    return static_cast<size_t>(produced);
  } catch (const std::exception& e) {
    std::cerr << "[H5Z-LC Error] " << e.what() << "\n";
    std::free(dst_buf);
    return 0;
  } catch (...) {
    std::cerr << "[H5Z-LC Error] Unknown exception.\n";
    std::free(dst_buf);
    return 0;
  }
}

}  // namespace

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_LC_CLASS[1] = {{
    H5Z_CLASS_T_VERS,
    H5Z_FILTER_PHOTONZIP_LC_ID,
    1,
    1,
    "H5Z-LC",
    H5Z_can_apply_lc,
    H5Z_set_local_lc,
    H5Z_filter_lc,
}};

H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }
const void* H5PLget_plugin_info(void) { return H5Z_LC_CLASS; }
