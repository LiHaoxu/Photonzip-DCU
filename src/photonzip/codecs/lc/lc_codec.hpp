#pragma once

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

// LC pipeline codec with the fixed component chain DIFFMS_2 + BIT_2 + RZE_2.
// Lossless. CPU backend always; DCU backend when built with PHOTONZIP_ENABLE_DCU. Both
// backends produce the same bitstream.
CodecVTable make_lc_codec();

}  // namespace photonzip
