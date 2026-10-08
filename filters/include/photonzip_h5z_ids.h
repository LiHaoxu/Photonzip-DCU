#pragma once

// HDF5 filter IDs for the PhotonZip codecs.
//
// LC uses an ID in the 32768-65535 range that HDF5 reserves for local/internal use.
#define H5Z_FILTER_PHOTONZIP_LC_ID 32771

// MANS reuses the ID of the upstream H5Z-MANS plugin, and stores the same cd_values
// (MansParams + chunk element count, written with backend=CPU and mode=P), so datasets
// written by the DCU plugin can be read by the upstream CPU plugin and vice versa for
// P-mode data. Do not put both plugins on the same HDF5_PLUGIN_PATH.
#define H5Z_FILTER_PHOTONZIP_MANS_ID 32032
