#pragma once

// HDF5 filter IDs for the PhotonZip codecs.
//
// Both live in the 32768-65535 range that HDF5 reserves for local/internal use, and they
// avoid the IDs the vendored MANS plugin already claims (32032, 32769 and 32770).
#define H5Z_FILTER_PHOTONZIP_LC_ID 32771
#define H5Z_FILTER_PHOTONZIP_PFPL_ID 32772
