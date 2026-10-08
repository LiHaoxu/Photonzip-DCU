# Vendored MANS (DCU subset)

Source: https://github.com/hpdps-group/MANS
Commit: 32793bc4c029f21d8093e6f498b06041998bc067 (2026-10-06)

Only the files needed by the standalone DCU backend are vendored, unmodified:

- `mans_defs.h`, `mans_utils.h` (stream header and helpers)
- `dcu/` (HIP ADM + CPU-compatible codec=1 P-mode ANS)
- `LICENSE`

The DCU backend emits codec=1 P-mode MANS streams, which upstream verifies to be
interoperable with the CPU backend (`tests/mans_cpu_dcu_cross_test.cpp`).
