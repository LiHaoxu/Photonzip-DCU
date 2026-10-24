// Round-trip test for the H5Z-LC filter: writes a chunked uint16 dataset through the filter,
// reads it back and checks that the LC pipeline reproduced every sample exactly.

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "photonzip_h5z_ids.h"

namespace {

constexpr hsize_t kDims[3] = {64, 128, 128};
constexpr hsize_t kChunk[3] = {8, 128, 128};

// A smooth field with a little noise: enough structure for the delta stage to pay off without
// making the test depend on a particular compression ratio.
std::vector<unsigned short> make_data() {
  std::vector<unsigned short> data(kDims[0] * kDims[1] * kDims[2]);
  std::size_t index = 0;
  for (hsize_t z = 0; z < kDims[0]; ++z) {
    for (hsize_t y = 0; y < kDims[1]; ++y) {
      for (hsize_t x = 0; x < kDims[2]; ++x) {
        const unsigned value = static_cast<unsigned>(1000 + 7 * x + 3 * y + 11 * z +
                                                     ((x * y + z) % 17));
        data[index++] = static_cast<unsigned short>(value);
      }
    }
  }
  return data;
}

}  // namespace

int main() {
  setenv("HDF5_PLUGIN_PATH", PHOTONZIP_H5Z_PLUGIN_DIR, 1);

  const char* const path = "h5z_lc_roundtrip.h5";
  const std::vector<unsigned short> original = make_data();
  const hsize_t element_count = kDims[0] * kDims[1] * kDims[2];

  if (H5Zfilter_avail(H5Z_FILTER_PHOTONZIP_LC_ID) <= 0) {
    std::fprintf(stderr, "H5Z-LC is not available from %s\n", PHOTONZIP_H5Z_PLUGIN_DIR);
    return 1;
  }

  const hid_t space = H5Screate_simple(3, kDims, nullptr);
  const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl, 3, kChunk);
  if (H5Pset_filter(dcpl, H5Z_FILTER_PHOTONZIP_LC_ID, H5Z_FLAG_MANDATORY, 0, nullptr) < 0) {
    std::fprintf(stderr, "H5Pset_filter failed\n");
    return 1;
  }

  const hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  const hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_USHORT, space, H5P_DEFAULT, dcpl,
                                H5P_DEFAULT);
  if (dset < 0) {
    std::fprintf(stderr, "H5Dcreate2 failed\n");
    return 1;
  }
  if (H5Dwrite(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, original.data()) < 0) {
    std::fprintf(stderr, "H5Dwrite failed\n");
    return 1;
  }

  const hsize_t stored_bytes = H5Dget_storage_size(dset);
  H5Dclose(dset);
  H5Pclose(dcpl);
  H5Sclose(space);
  H5Fclose(file);

  std::vector<unsigned short> restored(element_count, 0);
  const hid_t rfile = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t rdset = H5Dopen2(rfile, "data", H5P_DEFAULT);
  if (H5Dread(rdset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, restored.data()) < 0) {
    std::fprintf(stderr, "H5Dread failed\n");
    return 1;
  }
  H5Dclose(rdset);
  H5Fclose(rfile);

  if (std::memcmp(original.data(), restored.data(), original.size() * sizeof(unsigned short)) != 0) {
    std::fprintf(stderr, "H5Z-LC round trip is not lossless\n");
    return 1;
  }

  const double raw_bytes = static_cast<double>(original.size() * sizeof(unsigned short));
  std::printf("H5Z-LC round trip ok: %zu samples, %.0f -> %llu bytes, ratio %.2fx\n",
              original.size(), raw_bytes, static_cast<unsigned long long>(stored_bytes),
              raw_bytes / static_cast<double>(stored_bytes));
  std::remove(path);
  return 0;
}
