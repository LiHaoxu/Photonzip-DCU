# PhotonZip (LC)

PhotonZip is a tensor-first Python wrapper around a lossless codec for detector
and tomography data. This distribution ships a single codec, **PhotonZip-LC**:

- Fixed [LC framework](https://github.com/burtscher/LC-framework) pipeline
  `DIFFMS_2 BIT_2 RZE_2`, applied independently to 16 KiB chunks.
- Lossless, CPU-only, OpenMP-parallel; no tunable parameters.
- Bitstream compatible with the LC framework's standalone CPU compressor.
- Optional global inter-slice delta preprocessing (`photonzip.preprocess`).
- HDF5 filter plugin `H5Z-LC` (filter ID `32771`).

The Python layer is DLPack-based; `compress(...)` and `decompress(...)` return
a `PhotonZipArray`, which converts to NumPy with `np.from_dlpack(...)`.

## Layout

```
3rdparty/lossless/lc/      LC framework components (header-only)
src/photonzip/core/        codec registry, buffer types, DLPack header
src/photonzip/codecs/lc/   LC pipeline and codec registration
bindings/python/           pybind11 extension module photonzip._native
python/photonzip/          Python package (API, LC codec, preprocessing)
filters/                   HDF5 filter plugin H5Z-LC
examples/photonzip_cli.py  command-line tool with the .pzc container format
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
| `--codec` | `lc` | forces a codec, overriding the levels |

Legacy aliases `--level`/`--fidelity`, `--speed` and `--ratio` are accepted.

The `.pzc` container is `b"PZC1"`, a little-endian uint64 metadata length,
UTF-8 JSON metadata (`codec`, `backend`, `dtype`, `shape`, `codec_params`,
`preprocess`), then the LC payload.

## HDF5 filter

The plugin takes no `compression_opts`; the decoded size travels in the payload
header. It accepts uint16 datasets.

```bash
export HDF5_PLUGIN_PATH=$PWD/build/bin/plugins
```

```python
import h5py
import photonzip.codec.lc as lc

with h5py.File("scan.h5", "w") as f:
    f.create_dataset("data", data=volume, chunks=(16, 127, 127),
                     compression=lc.H5Z_FILTER_LC_ID)
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
