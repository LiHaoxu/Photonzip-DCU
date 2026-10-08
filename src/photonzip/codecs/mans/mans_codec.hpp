#pragma once

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

// MANS codec (ADM + ANS, P-mode) on the DCU backend. Lossless, uint16/uint32, 1-3D.
// Only built with PHOTONZIP_ENABLE_DCU. The stream is a complete MANS stream
// ([MansHeader][ANS payload], codec=1, mode=P) that the upstream MANS CPU backend can decode.
CodecVTable make_mans_codec();

}  // namespace photonzip
