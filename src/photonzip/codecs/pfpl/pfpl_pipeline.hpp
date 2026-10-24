#pragma once

// Self-contained host implementation of the PFPL pipeline QREL_4 + DIFFNB_4 + BIT_4 + RZE_1,
// i.e. the single-precision relative-error-bounded (REL) lossy compressor.
//
// The chunk-based bitstream format is byte-for-byte compatible with the standalone CPU
// compressor/decompressor from the upstream PFPL repository (src/f32_rel_comp_cpu.cpp and
// src/f32_rel_decomp_cpu.cpp), so a payload produced here can be decoded by the reference
// tool and vice versa.
//
// The quantiser below (h_QREL_4 / h_iQREL_4 and the log2/pow2 approximations they use) is
// derived from those upstream translation units, which are BSD 3-Clause licensed:
// Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez,
// Benila Jerald, Yiqian Liu, and Martin Burtscher. See 3rdparty/lossy/pfpl/LICENSE.
//
// Deviations from upstream, none of which change the bitstream:
//   - bit reinterpretation goes through memcpy instead of pointer casts, and the shifts that
//     would overflow a signed int are done on the unsigned type;
//   - the 4 header bytes that pad the error bound are zeroed rather than left uninitialised,
//     so a given input always encodes to the same bytes;
//   - empty inputs and a decoded chunk size mismatch are handled instead of being undefined.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace photonzip {
namespace pfpl {

using byte = unsigned char;

// Pipeline constants, identical to the upstream PFPL CPU framework.
static const int CS = 1024 * 16;  // chunk size in bytes, must be a multiple of 8

// The PFPL component headers expect the symbols above (byte/CS) to already be visible. They
// define `static inline` helpers, so including them inside this namespace keeps them out of
// the global scope.
#include "lossy/pfpl/components/h_DIFFNB_4.h"
#include "lossy/pfpl/components/h_BIT_4.h"
#include "lossy/pfpl/components/h_RZE_1.h"

namespace detail {

constexpr int kMantissaBits = 23;
constexpr std::int32_t kSignExpoMask = static_cast<std::int32_t>(0xff800000u);  // ~0 << 23
constexpr std::int32_t kMantissaMask = 0x007fffff;                              // ~kSignExpoMask
constexpr std::int32_t kMaxBin = (1 << (kMantissaBits - 2)) - 1;  // 2 bits left for the signs

inline std::int32_t bits_of(const float value) {
  std::int32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float float_of(const std::int32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline float log2approx(const float orig_f) {
  const std::int32_t orig_i = bits_of(orig_f);
  const int expo = (orig_i >> kMantissaBits) & 0xff;
  const float frac_f = float_of((127 << kMantissaBits) | (orig_i & kMantissaMask));
  return frac_f + static_cast<float>(expo - 128);  // - bias - 1
}

inline float pow2approx(const float log_f) {
  const float biased = log_f + 127.0f;
  const int expo = static_cast<int>(biased);
  const float frac_f = biased - static_cast<float>(expo - 1);
  return float_of((expo << kMantissaBits) | (bits_of(frac_f) & kMantissaMask));
}

// Quantises each float into a bin index whose reconstruction stays within the relative error
// bound. Values that cannot be represented that way (too large a bin, at or above the
// threshold, or a reconstruction that falls outside the bound) are passed through verbatim.
inline void h_QREL_4(int& csize, byte in[CS], byte out[CS], const float errorbound,
                     const float threshold) {
  const int size = csize / static_cast<int>(sizeof(float));

  const float log2eb = 2.0f * log2approx(1.0f + errorbound);
  const float inv_log2eb = 1.0f / log2eb;

  std::int32_t* const data_in_i = reinterpret_cast<std::int32_t*>(in);
  std::int32_t* const data_out_i = reinterpret_cast<std::int32_t*>(out);

  for (int i = 0; i < size; i++) {
    const std::int32_t orig_i = data_in_i[i];
    const std::int32_t abs_orig_i = orig_i & 0x7fffffff;
    const float abs_orig_f = float_of(abs_orig_i);
    std::int32_t output = orig_i;
    const int expo = (orig_i >> kMantissaBits) & 0xff;
    if (expo == 0) {          // zero or de-normal
      if (abs_orig_i == 0) {  // zero
        output = kSignExpoMask | 1;
      }
    } else if (expo == 0xff) {  // INF or NaN
      if (((orig_i & kSignExpoMask) == kSignExpoMask) && ((orig_i & kMantissaMask) != 0)) {
        output = abs_orig_i;  // turn a negative NaN into a positive one
      }
    } else {  // normal value
      const float scaled = log2approx(abs_orig_f) * inv_log2eb;
      std::int32_t bin = static_cast<std::int32_t>(std::roundf(scaled));
      const float abs_recon_f = pow2approx(static_cast<float>(bin) * log2eb);
      const float lower = abs_orig_f / (1.0f + errorbound);
      const float upper = abs_orig_f * (1.0f + errorbound);
      if (!((bin >= kMaxBin) || (bin <= -kMaxBin) || (abs_orig_f >= threshold) ||
            (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0.0f) ||
            !std::isfinite(abs_recon_f))) {
        bin = static_cast<std::int32_t>(static_cast<std::uint32_t>(bin) << 1) ^ (bin >> 31);
        bin = static_cast<std::int32_t>(static_cast<std::uint32_t>(bin + 1) << 1);
        if (orig_i < 0) {
          bin |= 1;  // carry the sign
        }
        // 'sign' and 'exponent' all ones with a non-zero 'mantissa' looks like a negative NaN,
        // which is exactly the pattern the pass-through branch above cannot produce.
        output = kSignExpoMask | bin;
      }
    }
    data_out_i[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(output) ^
                                              static_cast<std::uint32_t>(kSignExpoMask)) -
                    1;
  }
}

inline void h_iQREL_4(int& csize, byte in[CS], byte out[CS], const float errorbound) {
  const int size = csize / static_cast<int>(sizeof(float));

  const float log2eb = 2.0f * log2approx(1.0f + errorbound);

  std::int32_t* const data_in_i = reinterpret_cast<std::int32_t*>(in);
  std::int32_t* const data_out_i = reinterpret_cast<std::int32_t*>(out);

  for (int i = 0; i < size; i++) {
    std::int32_t val = static_cast<std::int32_t>(
        (static_cast<std::uint32_t>(data_in_i[i]) + 1u) ^ static_cast<std::uint32_t>(kSignExpoMask));
    if (((val & kSignExpoMask) == kSignExpoMask) && ((val & kMantissaMask) != 0)) {  // encoded
      if (val == (kSignExpoMask | 1)) {
        val = 0;
      } else {
        const std::int32_t dec = ((val & kMantissaMask) >> 1) - 1;
        const std::int32_t bin = (dec >> 1) ^ -(dec & 1);  // TCMS decoding
        const float abs_recon_f = pow2approx(static_cast<float>(bin) * log2eb);
        val = bits_of((val & 1) ? -abs_recon_f : abs_recon_f);
      }
    }
    data_out_i[i] = val;
  }
}

}  // namespace detail

// Quantiser selection. The mode is not recorded in the bitstream, so it has to travel in the
// codec parameters (and in the HDF5 filter's cd_values) for the writer and reader to agree.
enum class Mode : std::uint32_t {
  kRel = 0,  // |orig - recon| <= errorbound * |orig|
};

// Number of bytes the header occupies: the decoded size plus the error bound.
static const long long kHeaderBytes = 2 * static_cast<long long>(sizeof(long long));

// Upper bound on the encoded size for a given input size.
inline long long max_encoded_size(long long insize) {
  const long long chunks = (insize + CS - 1) / CS;  // round up
  return kHeaderBytes + chunks * static_cast<long long>(sizeof(unsigned short)) + chunks * CS;
}

// Largest input the chunk table and offsets can address.
inline long long max_input_size() { return static_cast<long long>(std::numeric_limits<int>::max()); }

// Reads the original (decoded) size stored in the PFPL header.
inline long long decoded_size(const byte* const input) {
  long long value = 0;
  std::memcpy(&value, input, sizeof(value));
  return value;
}

// Reads the error bound the payload was encoded with. The decoder needs no other parameter.
inline float stored_error_bound(const byte* const input) {
  float value = 0.0f;
  std::memcpy(&value, input + sizeof(long long), sizeof(value));
  return value;
}

inline void h_encode(const byte* const input, const long long insize, byte* const output,
                     long long& outsize, const float errorbound, const float threshold) {
  if (insize > max_input_size()) {
    throw std::runtime_error("PFPL input exceeds the addressable chunk range.");
  }
  if (insize % static_cast<long long>(sizeof(float)) != 0) {
    throw std::runtime_error("PFPL requires the input size to be a multiple of 4 bytes.");
  }

  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = reinterpret_cast<long long*>(output);
  unsigned short* const size_out = reinterpret_cast<unsigned short*>(&head_out[2]);
  byte* const data_out = reinterpret_cast<byte*>(&size_out[chunks]);
  long long* const carry = new long long[chunks == 0 ? 1 : chunks];
  if (chunks > 0) {
    std::memset(carry, 0, chunks * sizeof(long long));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    long long chunk1[CS / sizeof(long long)];
    long long chunk2[CS / sizeof(long long)];
    byte* in = reinterpret_cast<byte*>(chunk1);
    byte* out = reinterpret_cast<byte*>(chunk2);
    const long long base = chunkID * CS;
    const int osize = static_cast<int>(std::min(static_cast<long long>(CS), insize - base));
    std::memcpy(out, &input[base], osize);

    int csize = osize;
    bool good = true;
    if (good) {
      std::swap(in, out);
      detail::h_QREL_4(csize, in, out, errorbound, threshold);
    }
    if (good) {
      std::swap(in, out);
      good = h_DIFFNB_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      good = h_BIT_4(csize, in, out);
    }
    if (good) {
      std::swap(in, out);
      good = h_RZE_1(csize, in, out);
    }

    long long offs = 0LL;
    if (chunkID > 0) {
      do {
        #pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
      #pragma omp flush
    }
    if (good && (csize < osize)) {
      #pragma omp atomic write
      carry[chunkID] = (offs + static_cast<long long>(csize));
      size_out[chunkID] = static_cast<unsigned short>(csize);
      std::memcpy(&data_out[offs], out, csize);
    } else {
      #pragma omp atomic write
      carry[chunkID] = (offs + static_cast<long long>(osize));
      size_out[chunkID] = static_cast<unsigned short>(osize);
      std::memcpy(&data_out[offs], &input[base], osize);
    }
  }

  head_out[0] = insize;
  head_out[1] = 0;  // zero the error bound word first so its upper half is reproducible
  std::memcpy(&head_out[1], &errorbound, sizeof(errorbound));
  outsize = (chunks == 0) ? (data_out - output) : (&data_out[carry[chunks - 1]] - output);
  delete[] carry;
}

inline void h_decode(const byte* const input, byte* const output, long long& outsize) {
  const long long* const head_in = reinterpret_cast<const long long*>(input);
  long long out_total = 0;
  std::memcpy(&out_total, head_in, sizeof(out_total));
  outsize = out_total;
  const float errorbound = stored_error_bound(input);

  const long long chunks = (out_total + CS - 1) / CS;  // round up
  const unsigned short* const size_in = reinterpret_cast<const unsigned short*>(&head_in[2]);
  const byte* const data_in = reinterpret_cast<const byte*>(&size_in[chunks]);
  long long* const start = new long long[chunks == 0 ? 1 : chunks];

  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += static_cast<long long>(size_in[chunkID]);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    long long chunk1[CS / sizeof(long long)];
    long long chunk2[CS / sizeof(long long)];
    byte* in = reinterpret_cast<byte*>(chunk1);
    byte* out = reinterpret_cast<byte*>(chunk2);
    const long long base = chunkID * CS;
    const int osize = static_cast<int>(std::min(static_cast<long long>(CS), out_total - base));
    int csize = size_in[chunkID];
    if (csize == osize) {
      std::memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      std::memcpy(out, &data_in[start[chunkID]], csize);

      std::swap(in, out);
      h_iRZE_1(csize, in, out);
      std::swap(in, out);
      h_iBIT_4(csize, in, out);
      std::swap(in, out);
      h_iDIFFNB_4(csize, in, out);
      std::swap(in, out);
      detail::h_iQREL_4(csize, in, out, errorbound);

      if (csize != osize) {
        delete[] start;
        throw std::runtime_error("PFPL decode chunk size mismatch.");
      }
      std::memcpy(&output[base], out, csize);
    }
  }

  delete[] start;
}

}  // namespace pfpl
}  // namespace photonzip
