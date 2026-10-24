#pragma once

#include "photonzip/core/codec_types.hpp"

namespace photonzip {

// PFPL pipeline codec with the fixed component chain QREL_4 + DIFFNB_4 + BIT_4 + RZE_1.
// Error-bounded lossy, float32 only, CPU-only.
CodecVTable make_pfpl_codec();

}  // namespace photonzip
