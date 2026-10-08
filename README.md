# PhotonZip

PhotonZip is a tensor-first Python wrapper around lossless codecs for detector
and tomography data. It ships two codecs:

**PhotonZip-LC** (`codec="lc"`)

- Fixed [LC framework](https://github.com/burtscher/LC-framework) pipeline
  `DIFFMS_2 BIT_2 RZE_2`, applied independently to 16 KiB chunks.
- Lossless; no tunable parameters. Runs on the CPU (OpenMP) and, in DCU builds,
  on Hygon DCUs (HIP kernels from LC's HIP code generator).
- One bitstream for both backends, compatible with the LC framework's standalone
  compressors: compress on the DCU, decompress on the CPU, or the other way round.
- HDF5 filter plugin `H5Z-LC` (filter ID `32771`).

**PhotonZip-MANS** (`codec="mans"`, DCU builds only)

- [MANS](https://github.com/hpdps-group/MANS) (ADM mapping + ANS) in P-mode on the
  DCU, for uint16/uint32 tensors with 1-3 dimensions.
- Emits complete MANS streams that the upstream MANS CPU backend can decode.
- HDF5 filter plugin `H5Z-MANS` (filter ID `32032`, same as upstream H5Z-MANS).

Both codecs can be combined with the optional global inter-slice delta
preprocessing (`photonzip.preprocess`).

> **Status:** the CPU path is tested, and its LC output is byte-identical to the
> LC framework's standalone CPU compressor. The DCU sources compile for gfx906
> (wave64) with HIP/Clang but have not been run on DCU hardware yet; use
> `tests/test_dcu.py` to validate a build.

The Python layer is DLPack-based; `compress(...)` and `decompress(...)` return
a `PhotonZipArray`, which converts to NumPy with `np.from_dlpack(...)`.

## Layout

```
3rdparty/lossless/lc/      LC framework components (host h_* and device d_*, header-only)
3rdparty/lossless/MANS/    MANS DCU backend (vendored subset, see VENDORED.md)
src/photonzip/core/        codec registry, buffer types, DLPack header, HIP helpers
src/photonzip/codecs/lc/   LC pipeline (CPU), LC DCU kernels, codec registration
src/photonzip/codecs/mans/ MANS codec (DCU)
bindings/python/           pybind11 extension module photonzip._native
python/photonzip/          Python package (API, LC codec, preprocessing)
filters/                   HDF5 filter plugins H5Z-LC and H5Z-MANS
examples/photonzip_cli.py  command-line tool with the .pzc container format
tests/test_dcu.py          DCU correctness and throughput checks
```

## Install

Requirements: a C++17 compiler with OpenMP, CMake >= 3.20, Python >= 3.9,
NumPy, and (for the HDF5 filter) the HDF5 C library and headers.

```bash
python3 -m pip install .
```

Without HDF5 installed, skip the filter plugin:

```bash
python3 -m pip install . -Ccmake.define.PHOTONZIP_BUILD_HDF5_FILTERS=OFF
```

### DCU build

Requirements: Hygon DTK (`hipcc`), CMake >= 3.21. Set the architecture of your
DCU (`rocminfo | grep gfx`). Configure flags are forwarded through scikit-build:

```bash
python3 -m pip install . \
  -Ccmake.define.PHOTONZIP_ENABLE_DCU=ON \
  -Ccmake.define.PHOTONZIP_DCU_ARCH=gfx936 \
  -Ccmake.define.PHOTONZIP_DCU_HIP_COMPILER=$ROCM_PATH/bin/hipcc

# filters only
cmake -S . -B build -DPHOTONZIP_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release \
  -DPHOTONZIP_ENABLE_DCU=ON -DPHOTONZIP_DCU_ARCH=gfx936
cmake --build build -j
# -> build/bin/plugins/libH5Z-LC.so, build/bin/plugins/libH5Z-MANS.so

python3 tests/test_dcu.py --plugins build/bin/plugins
```

If the DTK compiler cannot find the host C++ headers (`fatal error: 'cmath' file
not found`), pass them via `PHOTONZIP_DCU_HIP_FLAGS`, e.g.
`-DPHOTONZIP_DCU_HIP_FLAGS="-isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11"`.

To build the HDF5 plugin on its own:

```bash
cmake -S . -B build -DPHOTONZIP_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/bin/plugins/libH5Z-LC.so
```

## Python API

```python
import numpy as np
import photonzip
from photonzip.preprocess import apply_delta, invert_delta

volume = np.fromfile("scan.u2", dtype=np.uint16).reshape(127, 127, 127)

packed = photonzip.compress(volume, codec="lc", backend="cpu")
restored = np.from_dlpack(photonzip.decompress(packed, backend="cpu"))
assert np.array_equal(restored, volume)

# DCU builds: same LC payload on the DCU, plus MANS.
packed = photonzip.compress(volume, codec="lc", backend="dcu")
packed = photonzip.compress(volume, codec="mans", backend="dcu")
restored = np.from_dlpack(photonzip.decompress(packed, backend="dcu"))

# With inter-slice delta preprocessing (uint16 only).
delta, meta = apply_delta(volume)
packed = photonzip.compress(delta, codec="lc", backend="cpu")
restored = invert_delta(np.from_dlpack(photonzip.decompress(packed, backend="cpu")), meta)
assert meta["lossless"] and np.array_equal(restored, volume)

payload = packed.to_bytes()
```

To decompress raw payload bytes, rebuild the array with
`photonzip._native.compressed_from_bytes("lc", payload, "uint16", shape, "cpu")`
and pass it to `photonzip.decompress(...)`.

### Delta preprocessing

`apply_delta` stores slice 0 as-is (bit-cast to int16) and every following
slice as `clip(x[i] - x[i-1], -32768, 32767)`, returned as a uint16 bit-view.
The result is lossless iff no difference overflows int16; the returned metadata
reports `overflow_pixels` and `lossless`. `invert_delta` restores the volume
with a cumulative sum modulo 2^16.

## CLI

```bash
# compress
python3 examples/photonzip_cli.py --mode compress \
  --input scan.u2 --output scan.pzc --dims 127 127 127 --dtype uint16 --delta

# decompress (shape, dtype and preprocessing are read from the container)
python3 examples/photonzip_cli.py --mode decompress --input scan.pzc --output scan.out.u2

# compress + decompress in memory, report throughput, ratio and is_equal
python3 examples/photonzip_cli.py --mode roundtrip \
  --input scan.u2 --output scan.pzc --dims 127 127 127 --dtype uint16 --delta
```

Compression levels select the codec and are kept for codecs added later:

| Option | Values | Current behaviour |
|---|---|---|
| `--quality-level` | `lossless` (default), `high`, `low` | only `lossless` is wired; `high`/`low` (error bound 1.0/8.0) raise `NotImplementedError` |
| `--throughput-level` | `high`, `low` | `high` selects `lc` |
| `--ratio-level` | `high`, `low` | falls back to the default lossless codec (`lc`) |
| `--codec` | `lc`, `mans` | forces a codec, overriding the levels (`mans` needs `--backend dcu`) |
| `--backend` | `cpu` (default), `dcu` | execution backend |

Legacy aliases `--level`/`--fidelity`, `--speed` and `--ratio` are accepted.

The `.pzc` container is `b"PZC1"`, a little-endian uint64 metadata length,
UTF-8 JSON metadata (`codec`, `backend`, `dtype`, `shape`, `codec_params`,
`preprocess`), then the LC payload.

Tensors that already live on the DCU (e.g. PyTorch on DTK, DLPack device
`kDLROCM`) are compressed in place and the results stay on the device.

## HDF5 filters

`H5Z-LC` takes no `compression_opts`; the decoded size travels in the payload
header. It accepts uint16 datasets. In DCU builds it runs on the DCU when one is
present; `PHOTONZIP_LC_BACKEND=cpu|dcu|auto` overrides that. The stored chunks
are the same either way.

`H5Z-MANS` (DCU builds) accepts uint16/uint32 datasets and takes the ADM geometry
from the chunk shape. Pass `compression_opts=(0,)` (P-mode) or nothing. It
stores the same parameters as the upstream MANS plugin, so the upstream CPU
plugin can read the data; do not put both plugins on one `HDF5_PLUGIN_PATH`.

```bash
export HDF5_PLUGIN_PATH=$PWD/build/bin/plugins
```

```python
import h5py
import photonzip.codec.lc as lc

with h5py.File("scan.h5", "w") as f:
    f.create_dataset("data", data=volume, chunks=(16, 127, 127),
                     compression=lc.H5Z_FILTER_LC_ID)

import photonzip.codec.mans as mans  # DCU builds
with h5py.File("scan_mans.h5", "w") as f:
    f.create_dataset("data", data=volume, chunks=(16, 127, 127),
                     compression=mans.H5Z_FILTER_MANS_ID, compression_opts=(0,))
```

## LC payload format

```
int64          decoded size in bytes
uint16[n]      compressed size of each 16 KiB chunk (n = ceil(size / 16384))
bytes          concatenated chunk payloads
```

## License

BSD 3-Clause (see `LICENSE`). The LC framework components under
`3rdparty/lossless/lc` carry their own BSD 3-Clause notice.
