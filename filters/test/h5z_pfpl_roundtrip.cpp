// Round-trip test for the H5Z-PFPL filter: writes a chunked float32 dataset through the
// filter, reads it back and checks that every sample honours the relative error bound.

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "photonzip_h5z_ids.h"

namespace {

constexpr hsize_t kDims[3] = {64, 128, 128};
constexpr hsize_t kChunk[3] = {8, 128, 128};
constexpr float kErrorBound = 1e-3f;

unsigned int float_bits(float value) {
  unsigned int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// A smooth positive field spanning a few decades, plus the special values the quantiser has
// dedicated paths for.
std::vector<float> make_data() {
  std::vector<float> data(kDims[0] * kDims[1] * kDims[2]);
  std::size_t index = 0;
  for (hsize_t z = 0; z < kDims[0]; ++z) {
    for (hsize_t y = 0; y < kDims[1]; ++y) {
      for (hsize_t x = 0; x < kDims[2]; ++x) {
        const double phase = 0.05 * static_cast<double>(x) + 0.03 * static_cast<double>(y) +
                             0.11 * static_cast<double>(z);
        data[index++] = static_cast<float>(std::exp(0.001 * static_cast<double>(z)) *
                                           (100.0 + 90.0 * std::sin(phase)));
      }
    }
  }
  data[0] = 0.0f;
  data[1] = -data[2];
  return data;
}

}  // namespace

int main() {
  setenv("HDF5_PLUGIN_PATH", PHOTONZIP_H5Z_PLUGIN_DIR, 1);

  const char* const path = "h5z_pfpl_roundtrip.h5";
  const std::vector<float> original = make_data();
  const hsize_t element_count = kDims[0] * kDims[1] * kDims[2];

  if (H5Zfilter_avail(H5Z_FILTER_PHOTONZIP_PFPL_ID) <= 0) {
    std::fprintf(stderr, "H5Z-PFPL is not available from %s\n", PHOTONZIP_H5Z_PLUGIN_DIR);
    return 1;
  }

  const unsigned int cd_values[2] = {0u /* REL */, float_bits(kErrorBound)};

  const hid_t space = H5Screate_simple(3, kDims, nullptr);
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl, 3, kChunk);
  if (H5Pset_filter(dcpl, H5Z_FILTER_PHOTONZIP_PFPL_ID, H5Z_FLAG_MANDATORY, 2, cd_values) < 0) {
    std::fprintf(stderr, "H5Pset_filter failed\n");
    return 1;
  }

  const hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  const hid_t dset = H5Dcreate2(file, "data", H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl,
                                H5P_DEFAULT);
  if (dset < 0) {
    std::fprintf(stderr, "H5Dcreate2 failed\n");
    return 1;
  }
  if (H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, original.data()) < 0) {
    std::fprintf(stderr, "H5Dwrite failed\n");
    return 1;
  }

  const hsize_t stored_bytes = H5Dget_storage_size(dset);
  H5Dclose(dset);
  H5Pclose(dcpl);
  H5Sclose(space);
  H5Fclose(file);

  std::vector<float> restored(element_count, 0.0f);
  const hid_t rfile = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t rdset = H5Dopen2(rfile, "data", H5P_DEFAULT);
  if (H5Dread(rdset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, restored.data()) < 0) {
    std::fprintf(stderr, "H5Dread failed\n");
    return 1;
  }
  H5Dclose(rdset);
  H5Fclose(rfile);

  double worst_relative = 0.0;
  for (std::size_t i = 0; i < original.size(); ++i) {
    const double orig = original[i];
    const double diff = std::fabs(orig - static_cast<double>(restored[i]));
    const double relative = (orig == 0.0) ? (diff == 0.0 ? 0.0 : HUGE_VAL) : diff / std::fabs(orig);
    if (relative > worst_relative) {
      worst_relative = relative;
    }
  }
  // PFPL only accepts a bin when the reconstruction lands inside orig * (1 + eb) evaluated
  // in float32. Both the sum and the product round, so the bound it actually enforces can
  // sit up to about one float32 epsilon above the requested one.
  const double realised_bound = kErrorBound + 2.0 * 1.1920929e-7;
  if (!(worst_relative <= realised_bound)) {
    std::fprintf(stderr, "H5Z-PFPL violated the error bound: %.6g > %.6g\n", worst_relative,
                 realised_bound);
    return 1;
  }

  const double raw_bytes = static_cast<double>(original.size() * sizeof(float));
  std::printf("H5Z-PFPL round trip ok: %zu samples, %.0f -> %llu bytes, ratio %.2fx, "
              "worst relative error %.3g (bound %.3g)\n",
              original.size(), raw_bytes, static_cast<unsigned long long>(stored_bytes),
              raw_bytes / static_cast<double>(stored_bytes), worst_relative,
              static_cast<double>(kErrorBound));
  std::remove(path);
  return 0;
}
