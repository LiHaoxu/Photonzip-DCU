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

## Local patch

`dcu/ans/dcu_ans.hip` (marked `[PhotonZip]`):

- `coalesce_kernel` ran on a single GPU thread (`<<<1, 1>>>`) and copied the whole
  stream byte by byte, which took ~2 s for a 16 MB input on a gfx936 DCU. It is
  replaced by `layout_kernel` (parallel prefix sum, header) and `scatter_kernel`
  (one CTA per ANS block). ANS encode of the same input now takes ~2 ms.
- `histogram_kernel` counts in shared memory before merging into the global
  histogram instead of issuing one global atomic per byte.
- All kernels with more than 256 threads per block declare `__launch_bounds__`;
  DTK otherwise limits kernels to 256 threads per block.

The stream format is unchanged; on the test inputs the patched encoder produces
byte-identical streams. Alignment padding is written as zeros instead of being
copied from uninitialised scratch memory.
