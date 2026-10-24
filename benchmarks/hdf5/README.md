# PhotonZip HDF5 tests

PhotonZip builds two additional HDF5 plugins:

- `libH5Z-LC.so` (`uint16`, lossless)
- `libH5Z-PFPL.so` (`float32`, PFPL REL, lossy)

They are written to `build/bin/plugins` together with the vendored MANS plugin.

## Correctness tests

```bash
export HDF5_PLUGIN_PATH="$PWD/build/bin/plugins"
build/bin/filters/h5z_lc_roundtrip
build/bin/filters/h5z_pfpl_roundtrip
python3 -m pytest -q tests/python/test_hdf5_filters.py
```

The equivalent h5py examples are:

```bash
python3 examples/hdf5/h5z_lc_roundtrip.py
python3 examples/hdf5/h5z_pfpl_roundtrip.py
```

## MPI performance test

`mpi_h5_bench.py` is the benchmark previously used on
`/data/lhx/test_data/part`. It uses one HDF5 file per MPI rank and reports all
throughput relative to the original `uint16` source bytes.

```bash
RANKS_LIST="1 8 16 32" \
CASES="raw,lc,mans_r,pfpl_rel" \
TIFF_DIR=/data/lhx/test_data/part \
bash benchmarks/hdf5/run_sweep.sh
```

Use `EXTRA_ARGS="--cold --codec-only"` for cold-read and codec-only timings.
The previously measured 32-rank PFPL REL error-bound sweep is retained in
`results/pfpl_rel_32r.csv`.

To reproduce only the PFPL REL `1e-1` point with 32 ranks:

```bash
export HDF5_PLUGIN_PATH="$PWD/build/bin/plugins"
RANKS_LIST=32 \
CASES=pfpl_rel \
TIFF_DIR=/data/lhx/test_data/part \
EXTRA_ARGS="--pfpl-eb 1e-1" \
bash benchmarks/hdf5/run_sweep.sh
```
# PhotonZip HDF5 tests

This directory contains the MPI performance test already used for the tomography
benchmark. It compares:

- raw HDF5 (`uint16`)
- H5Z-LC (`uint16`, lossless)
- H5Z-MANS p/r (`uint16`, lossless)
- H5Z-PFPL REL (`float32`, lossy)

All throughput columns are normalized to the original `uint16` source bytes.

Build PhotonZip and its filters:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPHOTONZIP_MANS_TARGET_PLATFORM=cpu \
  -DHDF5_PREFER_PARALLEL=OFF
cmake --build build -j
```

Run a small correctness/performance smoke test:

```bash
export HDF5_PLUGIN_PATH="$PWD/build/bin/plugins"
mpirun -n 1 python3 benchmarks/hdf5/mpi_h5_bench.py \
  --cases raw,lc,pfpl_rel \
  --mib-per-rank 4 \
  --chunk-frames 1 \
  --repeats 1 \
  --verify
```

Run the 1/8/16/32-rank sweep:

```bash
bash benchmarks/hdf5/run_sweep.sh
```

For `/data/lhx/test_data/part`, pass:

```bash
EXTRA_ARGS="--tiff-dir /data/lhx/test_data/part --verify --codec-only" \
  bash benchmarks/hdf5/run_sweep.sh
```

Two additional HDF5 correctness test sets are provided:

- C++: `filters/test/h5z_lc_roundtrip.cpp` and
  `filters/test/h5z_pfpl_roundtrip.cpp`
- h5py: `tests/python/test_hdf5_filters.py`

The previously measured 32-rank PFPL REL error-bound sweep is preserved in
`results/pfpl_rel_32r.csv`.
