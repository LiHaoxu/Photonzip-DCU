#pragma once

// HDF5 filter IDs for the PhotonZip codecs.
//
// LC and PFPL live in the 32768-65535 range that HDF5 reserves for local/internal use, and
// they avoid the IDs the vendored MANS plugin already claims (32032, 32769 and 32770).
#define H5Z_FILTER_PHOTONZIP_LC_ID 32771
#define H5Z_FILTER_PHOTONZIP_PFPL_ID 32772

// MANS reuses the ID of the upstream H5Z-MANS plugin, and stores the same cd_values
// (MansParams + chunk element count, written with backend=CPU and mode=P), so datasets
// written by the DCU plugin can be read by the upstream CPU plugin and vice versa for
// P-mode data. Do not put both plugins on the same HDF5_PLUGIN_PATH.
#define H5Z_FILTER_PHOTONZIP_MANS_ID 32032
