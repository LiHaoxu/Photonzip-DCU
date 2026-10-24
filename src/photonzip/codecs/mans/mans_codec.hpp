#pragma once

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

// MANS codec (ADM + ANS). Lossless, uint16/uint32, 1-3D.
//
// Backends: CPU (P- and R-mode, autotunable ADM threads), CUDA when the vendored MANS is
// built with its NVIDIA backend (MANS_ENABLE_NV), and DCU when built with
// PHOTONZIP_ENABLE_DCU (P-mode only). Every backend writes a complete MANS stream
// ([MansHeader][payload]) that carries its geometry; upstream MANS checks that P-mode streams
// decode across the CPU, CUDA and DCU backends.
CodecVTable make_mans_codec();

}  // namespace photonzip
