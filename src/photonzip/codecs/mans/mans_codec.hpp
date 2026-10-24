#pragma once

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

// MANS codec (ADM + ANS). Lossless, uint16/uint32, 1-3D, P- and R-mode.
//
// Backends: CPU (autotunable ADM threads) and CUDA when the vendored MANS is built with its
// NVIDIA backend (MANS_ENABLE_NV). Every stream ([MansHeader][payload]) carries its geometry;
// upstream MANS checks that P-mode streams decode across the CPU and CUDA backends.
CodecVTable make_mans_codec();

}  // namespace photonzip
