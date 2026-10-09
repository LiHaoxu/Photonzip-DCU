# Vendored MANS (DCU subset)

Source: https://github.com/hpdps-group/MANS
Commit: 32793bc4c029f21d8093e6f498b06041998bc067 (2026-10-06)

Only the files needed by the standalone DCU backend are vendored. They are unmodified
except for the patch listed below:

- `mans_defs.h`, `mans_utils.h` (stream header and helpers)
- `dcu/` (HIP ADM + CPU-compatible codec=1 P-mode ANS)
- `LICENSE`

The DCU backend emits codec=1 P-mode MANS streams, which upstream verifies to be
interoperable with the CPU backend (`tests/mans_cpu_dcu_cross_test.cpp`).

## Local patches

All changes are marked `[PhotonZip]` in the sources. They are performance fixes
plus zero-filling of bytes upstream left uninitialised: on a 19-case corpus (1D/2D/3D, uint16/uint32, 1 element to 256 MiB, skewed,
constant, sparse and mixed data) the patched encoder produces streams that are
byte-identical to the unmodified upstream DCU encoder, the upstream MANS
CPU/DCU test suite passes, and the upstream CPU decoder reads the streams.

Times are for a 16 MiB uint16 2D input on a Hygon BW (gfx936) unless noted.

`dcu/ans/dcu_ans.hip`
- `coalesce_kernel` assembled the stream on a single GPU thread (`<<<1, 1>>>`):
  ~2 s. Replaced by `layout_kernel` (parallel prefix sum) and `scatter_kernel`
  (one CTA per ANS block).
- `normalize_kernel` ran on thread 0 only (insertion sort + adjustment rounds,
  ~1.3 ms per call). Now 256 threads compute the identical table.
- `decode_kernel` had lane 0 of every CTA validate the whole block directory
  (O(blocks^2): 325 ms for 256 MiB) and rebuild the inverse table per CTA.
  The checks now run once in `validate_kernel`, the table is built once in
  `table_kernel`; a malformed stream still decodes nothing and reports error 1.
- `histogram_kernel` counts in shared memory with a grid scaled to the input.
- The block directory is padded to an even number of 8-byte entries; with an odd
  block count the pad entry was never written and kept whatever the output buffer
  held, so the same input could give different streams (only those 8 bytes;
  decoders skip them). `layout_kernel` now writes it as zeros.

`dcu/adm/mapping_uint16.hip`, `dcu/adm/adm_kernel.cuh`
- ADM encode copied lengths, flags, centers, codes and the 32 B-per-lane signal
  scratch to the host, packed the payload there (~1 M small `memcpy` calls) and
  copied it back: ~15 ms. The payload is now packed on the device
  (`offsets_kernel`, `pack_meta_kernel`, `pack_signals_kernel`); only the 8-byte
  signal total is read back. Codes of RAW blocks are zeroed (upstream left
  uninitialised memory there; decoders ignore those bytes).
- `flat_index` (two 64-bit divisions per element) is replaced by an incremental
  `TileCursor` in encode and decode.
- Full-depth 3D tiles (nz >= 16) are staged through shared memory so the tile is
  read and the codes/output are written in coalesced row order (`kStaged`).

`dcu/mans_dcu.cpp`, `dcu/dcu_workspace.h` (new)
- Scratch memory comes from per-device, per-thread-safe workspaces instead of
  ~20 `hipMalloc`/`hipFree` pairs per call.

All kernels with more than 256 threads per block declare `__launch_bounds__`;
DTK otherwise limits kernels to 256 threads per block.

Result (device-resident, compress / decompress): 16 MiB 2D 0.01 / 5.8 GB/s
(unmodified upstream) -> 13.8 / 26.8 GB/s; 256 MiB 3D volume 1.5 / 0.9 GB/s
(upstream + the coalesce fix only) -> 12.8 / 21.5 GB/s.
