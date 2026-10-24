# 📦 PhotonZip

PhotonZip is a tensor-first Python wrapper around vendored lossless and
error-bounded lossy codecs for detector and tomography data, on the CPU, NVIDIA
GPUs (CUDA) and Hygon DCUs.

This repository is [hpdps-group/PhotonZip](https://github.com/hpdps-group/PhotonZip)
plus a DCU backend, the LC and PFPL codecs, HDF5 filters and global delta
preprocessing. Upstream is merged as a git remote (`git merge upstream/main`).

| Codec | Data | CPU | CUDA | DCU | HDF5 filter |
|---|---|---|---|---|---|
| **MANS** (`codec="mans"`) | uint16/uint32, 1-3D | P- and R-mode, autotuned threads | P- and R-mode | P-mode | `H5Z-MANS` (32032) |
| **LC** (`codec="lc"`) | uint16 | yes | - | yes | `H5Z-LC` (32771) |
| **PFPL** (`codec="pfpl"`) | float32, lossy (relative error bound) | yes | - | - | `H5Z-PFPL` (32772) |

**MANS** ([hpdps-group/MANS](https://github.com/hpdps-group/MANS), ADM mapping + ANS)

- CPU and CUDA paths from upstream PhotonZip, exposed through `photonzip.compress(...)`,
  `photonzip.decompress(...)` and `photonzip.codec.mans.autotune(...)`.
- DCU path in DCU builds (P-mode). The streams of all backends carry their geometry;
  P-mode streams decode on any backend.

**LC** ([LC framework](https://github.com/burtscher/LC-framework) pipeline `DIFFMS_2 BIT_2 RZE_2`)

- Applied independently to 16 KiB chunks. Lossless, no tunable parameters.
- CPU (OpenMP) and, in DCU builds, DCU (HIP kernels from LC's HIP code generator).
- One bitstream for both backends, compatible with the LC framework's standalone
  compressors: compress on the DCU, decompress on the CPU, or the other way round.

**PFPL** ([PFPL](https://github.com/burtscher/PFPL) REL pipeline `QREL_4 DIFFNB_4 BIT_4 RZE_1`)

- Error-bounded lossy for float32: every sample is restored within `error_bound`
  of its own magnitude; samples at or above `threshold` are stored exactly.
- CPU only. The bitstream matches the standalone PFPL CPU compressor.

The lossless codecs can be combined with the optional global inter-slice delta
preprocessing (`photonzip.preprocess`), which runs on the CPU or, in DCU builds,
on the DCU, where its output can stay in device memory for the codec.

The Python layer is DLPack-based; `compress(...)` and `decompress(...)` return
a `PhotonZipArray`, which converts to NumPy with `np.from_dlpack(...)`.

> **Status**
>
> - CPU + CUDA: built on an RTX 5090 host (GCC 11, CUDA 12.8, HDF5 1.10.7) and in the
>   NGC PyTorch 25.06 container (CUDA 12.9, PyTorch 2.8). `tests/python/test_codecs.py`
>   passes (MANS CPU/CUDA, LC, PFPL), MANS P-mode streams decode across the CPU and CUDA
>   backends, and the C++ and h5py HDF5 filter tests pass, except
>   `test_lc_filter_declines_float32` on HDF5 1.10.7, which keeps a declined optional
>   filter in the dataset's filter list (the chunks are stored raw and read back).
> - DCU: tested on a Hygon BW DCU (gfx936, 80 CUs) with DTK 26.04, and with DTK 25.04.1 +
>   PyTorch 2.5.1 for device tensors. LC payloads from the DCU are byte-identical to the LC
>   framework's official standalone CPU and GPU compressors, and both official decoders
>   read them. MANS streams from the DCU decode with the MANS CPU backend and vice versa
>   (P-mode); the upstream MANS DCU/CPU test suite passes. HDF5 files written with the DCU
>   filters read back with the CPU paths (`PHOTONZIP_LC_BACKEND=cpu`, the CPU H5Z-MANS)
>   and vice versa.

## Layout

```
3rdparty/lossless/lc/      LC framework components (host h_* and device d_*, header-only)
3rdparty/lossless/MANS/    MANS (CPU, CUDA, DCU backends, H5Z-MANS), vendored, see VENDORED.md
3rdparty/lossy/pfpl/       PFPL components (host, header-only)
src/photonzip/core/        codec registry, buffer types, DLPack header, HIP helpers
src/photonzip/codecs/lc/   LC pipeline (CPU), LC DCU kernels, codec registration
src/photonzip/codecs/mans/ MANS codec (CPU, CUDA, DCU)
src/photonzip/codecs/pfpl/ PFPL REL pipeline (CPU) and codec registration
src/photonzip/preprocess/  inter-slice delta: CPU (OpenMP) and DCU (HIP) kernels
bindings/python/           pybind11 extension module photonzip._native
python/photonzip/          Python package (API, codecs, preprocessing)
filters/                   HDF5 filter plugins H5Z-LC, H5Z-PFPL, (DCU builds) H5Z-MANS; C++ tests
examples/photonzip_cli.py  command-line tool with the .pzc container format
examples/MANS/             MANS examples (autotune, CPU/CUDA round trips, HDF5)
examples/hdf5/             H5Z-LC and H5Z-PFPL round trips with h5py
tests/python/              pytest suite (codecs on CPU/CUDA, HDF5 filters via h5py)
tests/test_dcu.py          DCU correctness checks (Python API, device tensors, HDF5)
benchmarks/dcu_bench.cpp   DCU throughput benchmark (device-resident, host chunks, multi-DCU)
benchmarks/hdf5/           MPI/HDF5 filter benchmarks (tomography pipeline, PFPL REL sweep)
scripts/package_source.py  source-only ZIP of the repository
testdata/u2/               sample uint16 volumes (127^3)
```

## 🔧 Install

Requirements: a C++17 compiler with OpenMP, CMake >= 3.20, Python >= 3.9, NumPy,
and (for the HDF5 filters) the HDF5 C library with its high-level library, plus MPI
for the MANS HDF5 tools. MANS is vendored, so no submodules are needed.

```bash
git clone https://github.com/LiHaoxu/Photonzip-DCU.git
cd Photonzip-DCU
python3 -m pip install .            # or: python3 -m pip install -e . --no-deps
```

Without HDF5 installed, skip the filter plugins:

```bash
python3 -m pip install . -Ccmake.define.PHOTONZIP_BUILD_HDF5_FILTERS=OFF
```

The CUDA path of MANS is built when CMake finds a CUDA compiler. Force a platform
with `-DPHOTONZIP_MANS_TARGET_PLATFORM=cpu` or `cpu_nv`; `photonzip.available_backends()`
lists what a build has. For the CUDA examples and tests, use an environment with
PyTorch, e.g.

```bash
conda create -n photonzip python=3.12 pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -c conda-forge
conda activate photonzip
```

To build the native module and plugins with CMake only:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -j
python3 -m pytest -q tests/python/test_codecs.py     # uses build/bindings/python
```

### DCU build

Requirements: Hygon DTK (`hipcc`), CMake >= 3.21. Load the DTK first
(`module load compiler/dtk/...` or `source $DTK/env.sh`) and set the
architecture of your DCU (`rocminfo | grep gfx`; e.g. BW is `gfx936`).
Configure flags are forwarded through scikit-build:

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
build/bin/photonzip_dcu_bench data.u2 --dims 8192 1024   # needs a DCU; C-order shape
build/bin/photonzip_dcu_bench data.u2 --dims 8192 1024 --chunk-rows 512 --parallel 2 4
# delta on the DCU in front of the codecs (frames of 2048*2048 elements)
build/bin/photonzip_dcu_bench scan.u2 --dims 100 2048 2048 --delta offset --frame 4194304
# chunks of 10 frames kept in DCU memory, 4 threads on one DCU: kernels only, no PCIe
build/bin/photonzip_dcu_bench scan.u2 --dims 100 2048 2048 --chunk-rows 10 --resident --host-only --parallel 1 4
```

A DCU build also contains the MANS CPU backend, so DCU streams can be checked
against the CPU in one process (`tests/test_dcu.py` does). Its `H5Z-MANS` is the DCU
plugin from `filters/` (the vendored CPU/CUDA plugin is not built).

For the HDF5 filters, use a serial, shared HDF5 and make sure h5py links the
same `libhdf5` as the plugins (e.g. `HDF5_DIR=... pip install --no-binary=h5py h5py`);
two HDF5 copies in one process break filter callbacks.

If the DTK compiler cannot find the host C++ headers (`fatal error: 'cmath' file
not found`), pass them via `PHOTONZIP_DCU_HIP_FLAGS`, e.g.
`-DPHOTONZIP_DCU_HIP_FLAGS="-isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11"`.

## 🧪 Usage

### CLI

```bash
# CPU, high throughput (LC)
python examples/photonzip_cli.py --mode compress \
  --input testdata/u2/sfc-gi/sfc-gi_127x127x127_4096kB.u2 --output /tmp/photonzip_cpu.pzc \
  --dims 127 127 127 --dtype uint16 --backend cpu --quality-level lossless --throughput-level high

# CPU, high compression ratio (MANS R-mode)
python examples/photonzip_cli.py --mode compress \
  --input testdata/u2/sfc-gi/sfc-gi_127x127x127_4096kB.u2 --output /tmp/photonzip_cpu.pzc \
  --dims 127 127 127 --dtype uint16 --backend cpu --quality-level lossless --ratio-level high

# decompress (shape, dtype, codec and preprocessing are read from the container)
python examples/photonzip_cli.py --mode decompress \
  --input /tmp/photonzip_cpu.pzc --output /tmp/photonzip_cpu_restored.u2 --backend cpu

# CUDA round trip (MANS on the GPU; needs PyTorch)
python examples/photonzip_cli.py --mode roundtrip \
  --input testdata/u2/sfc-gi/sfc-gi_127x127x127_4096kB.u2 --output /tmp/photonzip_cuda.pzc \
  --dims 127 127 127 --dtype uint16 --backend cuda

# DCU round trip with inter-slice delta (MANS on the DCU, delta on the DCU)
python examples/photonzip_cli.py --mode roundtrip \
  --input scan.u2 --output scan.pzc --dims 100 2048 2048 --dtype uint16 --codec mans --backend dcu --delta
```

The CLI prints throughput for the requested mode (`compress`: compression throughput
and ratio; `decompress`: decompression throughput; `roundtrip`: both, the ratio and an
equality check).

| Option | Values | Behaviour |
|---|---|---|
| `--quality-level` | `lossless` (default), `high`, `low` | only `lossless` is wired; `high`/`low` (error bound 1.0/8.0) are rejected |
| `--throughput-level` | `high`, `low` | `high` selects `lc` |
| `--ratio-level` | `high`, `low` | `high` selects MANS R-mode (P-mode on the DCU, with a warning) |
| `--codec` | `mans`, `lc` | forces a codec, overriding the levels; default `mans` |
| `--backend` | `cpu`, `cuda`, `dcu` | compress: `dcu` for MANS in DCU builds, else `cpu`; decompress: the backend recorded in the file if the build has it, else `cpu`. LC runs on `cpu`/`dcu` |
| `--no-warmup` | | include the one-time GPU/DCU start-up (~50 ms on a DCU) in the timings |
| `--delta` | | inter-slice delta before compression (uint16), see "Delta preprocessing" |
| `--delta-backend` | `cpu`, `dcu`, `numpy` | where `--delta` and its inverse run; default `dcu` when the codec runs on the DCU and the data has 2+ dimensions (the delta output then stays in DCU memory), else `cpu`. The container is the same either way |

For MANS on `cpu`/`cuda`, the CLI reuses `build/best_threads.csv` if it exists;
otherwise it runs MANS autotune once and writes that CSV. The DCU needs no thread
table.

`--dims` is the C-order shape (slowest dimension first, 1 to 3 values). Legacy
aliases `--level`/`--fidelity`, `--speed` and `--ratio` are accepted. Invalid
arguments exit with status 2, runtime errors with status 1. `--delta` stores the
differences with the `offset` encoding for MANS and `int16` for LC.

Throughputs are raw bytes / time. `compress` and `decompress` time the codec
alone, including the copy of the compressed stream between device and host (with
`--backend cuda` the raw input is copied to the GPU before timing starts, as
upstream). With `--delta` the CLI also prints `delta` / `inverse_delta` (seconds)
and `compress_with_delta` / `decompress_with_delta`, which include them. With the
DCU delta the codec reads its input from (or writes its output to) DCU memory, so
the host <-> DCU copy of the raw data is counted in the delta time.

The `.pzc` container is `b"PZC1"`, a little-endian uint64 metadata length,
UTF-8 JSON metadata (`codec`, `backend`, `dtype`, `shape`, `codec_params`,
`preprocess`), then the codec payload.

More examples are in [`examples`](./examples) (MANS autotune, CPU/CUDA round trips,
HDF5).

### Python API

```python
import numpy as np
import photonzip
import photonzip.codec.mans as mans
from photonzip.preprocess import apply_delta, invert_delta

volume = np.fromfile("scan.u2", dtype=np.uint16).reshape(127, 127, 127)
photonzip.available_backends()        # e.g. ['cpu', 'cuda'] or ['cpu', 'dcu']

# MANS on the CPU with an autotuned thread table
table = mans.autotune(mans.MansAutotuneOptions()).thread_table
packed = photonzip.compress(volume, codec="mans", backend="cpu",
                            codec_options=mans.MansOptions(mode="r", thread_table=table))
restored = np.from_dlpack(photonzip.decompress(packed, backend="cpu"))
assert np.array_equal(restored, volume)

# MANS on CUDA (input on the host or as a CUDA tensor; the result stays on the GPU)
packed = photonzip.compress(volume, codec="mans", backend="cuda")
restored = photonzip.decompress(packed, backend="cuda")      # .to_host() copies it back

# LC on the CPU, and in DCU builds the same LC payload on the DCU, plus MANS
packed = photonzip.compress(volume, codec="lc", backend="cpu")
packed = photonzip.compress(volume, codec="lc", backend="dcu")
packed = photonzip.compress(volume, codec="mans", backend="dcu")
restored = np.from_dlpack(photonzip.decompress(packed, backend="dcu"))

# With inter-slice delta preprocessing (uint16 only)
delta, meta = apply_delta(volume)
packed = photonzip.compress(delta, codec="lc", backend="cpu")
restored = invert_delta(np.from_dlpack(photonzip.decompress(packed, backend="cpu")), meta)
assert meta["lossless"] and np.array_equal(restored, volume)

payload = packed.to_bytes()

# DCU builds: delta on the DCU, its output left in DCU memory and compressed there, so the
# raw volume crosses PCIe once; only the compressed stream comes back.
delta, meta = apply_delta(volume, encoding="offset", backend="dcu", output_on_dcu=True)
packed = photonzip.compress(delta, codec="mans", backend="dcu").to_host()
restored = photonzip.decompress(packed.to_device(), backend="dcu")       # stays on the DCU
restored = invert_delta(restored, meta, output_on_dcu=False)              # NumPy array
```

PFPL compresses float32 within a relative error bound (CPU only):

```python
from photonzip.codec.pfpl import PfplOptions

x = np.asarray(..., dtype=np.float32)
packed = photonzip.compress(x, codec="pfpl", codec_options=PfplOptions(error_bound=1e-3))
restored = np.from_dlpack(photonzip.decompress(packed))
# |x - restored| <= ~error_bound * |x|; PfplOptions(threshold=t) keeps |x| >= t exact
```

`photonzip.compress(...)` defaults to `codec="mans"`. `MansOptions(mode="p"|"r")`
selects the MANS pipeline (default `"p"`); on the CPU the ADM thread counts come
from `adm_compress_thread`/`adm_decompress_thread`, a `thread_table`, a
`thread_csv_path` or `$MANS_THREAD_CSV`, else 32/32 with a one-time warning.

`PhotonZipArray.to_host()` copies an array (compressed or not) from DCU or CUDA
memory to the host, `to_device()` copies it to DCU memory, `.device` says where
it is (`"cpu"`, `"cuda"` or `"dcu"`), and `.reshape(shape)` gives an uncompressed
array another shape without a copy, e.g. to pick the MANS geometry of a
DCU-resident array.

To decompress raw payload bytes, rebuild the array with
`photonzip._native.compressed_from_bytes("lc", payload, "uint16", shape, "cpu")`
and pass it to `photonzip.decompress(...)`.

Tensors that already live on the GPU or DCU (PyTorch/CuPy, DLPack `kDLCUDA` or
`kDLROCM`) are compressed in place and the results stay on the device.

### Delta preprocessing

`apply_delta(array, encoding="int16", backend="auto", output_on_dcu=None)` stores
slice 0 (along axis 0) as-is and every following slice as
`clip(x[i] - x[i-1], -32768, 32767)` in a uint16 array. The result is lossless
iff no difference overflows int16; the returned metadata reports
`overflow_pixels`, `lossless` and the `encoding`. `invert_delta(array, meta, ...)`
restores the volume with a running sum modulo 2^16 (metadata without `encoding`
means `int16`).

All backends produce the same bytes:

| `backend` | runs | input / output |
|---|---|---|
| `numpy` | NumPy reference (single thread) | host |
| `cpu` | C++, OpenMP over the elements of a slice | host |
| `dcu` | HIP kernel, each DCU thread walks all slices for up to 8 consecutive elements | host or DCU; host input is copied in and the result back unless `output_on_dcu=True` |
| `auto` (default) | `dcu` for arrays in DCU memory, `cpu` otherwise | |

The DCU kernel parallelises over the elements of one slice, so it suits stacks of
2-D frames; a 1-D array (one element per slice) runs on a single DCU thread and
belongs on the CPU.

| `encoding` | stored value | use with |
|---|---|---|
| `int16` (default) | two's complement bit pattern (-1 -> 65535) | LC |
| `offset` | difference + 32768 (-1 -> 32767) | MANS |

MANS needs `offset`: its ADM only maps blocks whose `max - min` is below 3500 and
codes `|value - center|`, so with `int16` any block holding both positive and
negative differences spans ~65535 and is stored raw. On a 100 x 2048 x 2048
tomography stack, `int16` left 2% of the blocks to ADM and MANS (ratio 1.254) did
worse than plain ANS on the same bytes (1.473); with `offset` 99.9% of the blocks
use ADM and MANS reaches 1.479. The CLI picks `offset` for MANS.

### MANS geometry

MANS indexes elements as `x + y*nx + z*nx*ny` (`nx` fastest). PhotonZip arrays and
HDF5 chunks are C-ordered, so the codec (on every backend), the HDF5 options from
`mans.to_hdf5_compression_opts` and both H5Z-MANS plugins pass the *last* dimension
as `nx` (`nx = shape[-1]`, `ny = shape[-2]`, `nz` = product of the remaining
dimensions). Passing `shape[0]` as `nx` (as upstream did) transposes the ADM tiles:
still lossless, but e.g. a 8x512x512 volume chunk drops from ratio 1.96 to 1.48 and
decompresses at half the speed. The geometry is stored in every MANS stream, so
streams written with either convention decode anywhere.

## HDF5 filters

Building PhotonZip also creates, under `build/bin/plugins/`:

- `libH5Z-MANS.so` (filter ID 32032): the vendored CPU plugin, or the DCU plugin in DCU builds
- `libH5Z-LC.so` (filter ID 32771)
- `libH5Z-PFPL.so` (filter ID 32772)

Set `HDF5_PLUGIN_PATH=$PWD/build/bin/plugins` before importing `h5py`.
Round-trip examples are under `examples/hdf5/`; C++ and h5py correctness tests
are under `filters/test/` (`build/bin/filters/h5z_*_roundtrip`) and
`tests/python/test_hdf5_filters.py`. The MPI benchmark and the 32-rank PFPL REL
sweep are under `benchmarks/hdf5/` (their default data paths point at the RTX 5090
host they were run on).

`H5Z-LC` takes no `compression_opts`; the decoded size travels in the payload
header. It accepts uint16 datasets. In DCU builds it runs on the DCU when one is
present; `PHOTONZIP_LC_BACKEND=cpu|dcu|auto` overrides that. The stored chunks
are the same either way.

`H5Z-PFPL` accepts float32 datasets; its `compression_opts` come from
`photonzip.codec.pfpl.to_hdf5_compression_opts(PfplOptions(...))` (mode, error
bound and threshold as float bits).

`H5Z-MANS` accepts uint16/uint32 datasets and derives the ADM geometry from each
chunk (last chunk dimension as `nx`, see "MANS geometry"); it stores the geometry in
both the filter parameters and every compressed chunk. Compression does not cross
chunk boundaries; use the intended 3D volume as the chunk shape when cross-frame
correlation is required. With the CPU plugin, build the parameters with
`mans.to_hdf5_compression_opts(...)` (mode and autotuned threads). The DCU plugin
takes `compression_opts=(0,)` (P-mode) or nothing, and stores the same parameters
as the CPU plugin, so either plugin reads P-mode data written by the other; do not
put both plugins on one `HDF5_PLUGIN_PATH`.

```python
import h5py
import photonzip.codec.lc as lc
import photonzip.codec.mans as mans

with h5py.File("scan.h5", "w") as f:
    f.create_dataset("lc", data=volume, chunks=(16, 127, 127),
                     compression=lc.H5Z_FILTER_LC_ID)
    f.create_dataset("mans", data=volume, chunks=(16, 127, 127),
                     compression=mans.H5Z_FILTER_MANS_ID,
                     compression_opts=mans.to_hdf5_compression_opts(volume, chunks=(16, 127, 127)))
    # DCU builds: compression_opts=(mans.MANS_MODE_P,)
```

## 📊 Performance

### CPU Compression Throughput

![CPU compression throughput](./images/cpuTHR-CMP.png)

### CPU Decompression Throughput

![CPU decompression throughput](./images/cpuTHR-DECMP.png)

### NV Compression Throughput

![NV compression throughput](./images/gpu_cmp.png)

### NV Decompression Throughput

![NV decompression throughput](./images/gpu_decmp.png)

### Compression Ratio

![Compression Ratio](./images/CR.png)

### DCU

Hygon BW (gfx936, 80 CUs), DTK 26.04. All results verified against the input.

**Device-resident** (`photonzip_dcu_bench`: data already in DCU memory, no PCIe
transfers, median of 5 runs; shapes in C order):

| Input | LC ratio | LC compress / decompress | MANS ratio | MANS compress / decompress |
|---|---|---|---|---|
| EXAFEL uint16, 16 MiB (8192x1024) | 1.831 | 64.7 / 58.5 GB/s | 1.706 | 13.8 / 26.8 GB/s |
| EXAFEL x64 uint16, 1 GiB (524288x1024) | 1.831 | 89.4 / 74.7 GB/s | 1.705 | 15.5 / 40.3 GB/s |
| smooth volume uint16, 256 MiB (512^3) | 2.682 | 89.0 / 75.2 GB/s | 1.928 | 12.8 / 21.5 GB/s |
| volume chunk uint16, 4 MiB (8x512x512) | 2.678 | 38.8 / 32.4 GB/s | 1.960 | 7.3 / 11.9 GB/s |
| uniform random uint16, 4 MB | 1.000 | 44.7 / 63.6 GB/s | 0.824 | 7.5 / 14.2 GB/s |
| uint32, 2 MB (512x1000) | 1.450 | 30.1 / 25.2 GB/s | 2.666 | 8.1 / 8.4 GB/s |

The official LC GPU compressor for the same pipeline, built for gfx936, reaches
56.9 GB/s (compress) on the 16 MiB EXAFEL input.

**Host memory** (the path used by the Python API, the CLI and the HDF5 filters:
copy in, compress, copy out). One call is bounded by PCIe and per-call setup
(~2-4 GB/s), so throughput scales with concurrent calls. 1 GiB EXAFEL in 4 MiB
chunks (`photonzip_dcu_bench --chunk-rows 4096 --parallel D T`):

| DCUs x host threads | LC compress / decompress | MANS compress / decompress |
|---|---|---|
| 1 x 1 | 4.6 / 4.4 GB/s | 2.2 / 3.8 GB/s |
| 1 x 4 | 12.7 / 12.8 GB/s | 6.3 / 12.3 GB/s |
| 1 x 8 | 14.7 / 16.6 GB/s | 10.0 / 15.4 GB/s |
| 4 x 4 | 16.9 / 16.3 GB/s | 14.2 / 15.7 GB/s |
| 8 x 2 | 19.3 / 17.8 GB/s | 12.2 / 16.7 GB/s |

Beyond ~16-19 GB/s the host side (pageable copies, memory bandwidth) is the
limit. The Python bindings release the GIL during (de)compression, so a thread
pool works too: 8 threads on one DCU reach 10.0 / 10.1 GB/s for MANS and
7.9 / 10.7 GB/s for LC on the same data (vs. 2.6 / 2.5 and 2.8 / 2.8 GB/s with
one thread). The first DCU call in a process also pays ~50 ms of runtime
start-up; the CLI excludes it from its timings (see `--no-warmup`).

**Delta on the DCU** (100 x 2048 x 2048 uint16 tomography stack, 800 MiB, offset
encoding for MANS; MANS geometry (2048, R, 2048), see "MANS geometry"):

- The delta kernel takes 1.57 ms (inverse 1.56 ms), ~1.07 TB/s of DCU memory
  traffic. Device-resident MANS with the delta: 12.5 / 13.5 GB/s, against
  12.8 / 13.9 GB/s for MANS alone on precomputed differences (ratio 1.479).
- Host memory in and out, the delta alone costs 0.20 s on the DCU (two PCIe
  copies), 0.16 s with the OpenMP CPU backend (16 cores) and 2.0 s with NumPy.
  Keeping its output on the DCU for the codec avoids one of the copies: one Python
  call (`output_on_dcu=True`, see "Python API") compresses the stack at 3.7 GB/s
  and decompresses it at 3.6 GB/s, delta included.
- Host path on 8 DCUs x 2 threads in chunks of 10 frames, delta fused into each
  chunk (`photonzip_dcu_bench --delta offset --delta-scope ...`), against
  15.0 / 16.2 GB/s for MANS on precomputed differences with the delta not timed:
  - `global` (the bytes of the whole-stack delta, ratio 1.509): 14.9 / 11.3 GB/s.
    Decompression needs two passes, since every chunk starts from the restored
    last frame of the one before;
  - `chunk` (the delta restarts in every chunk, chunks decode independently,
    ratio 1.458): 14.5 / 15.5 GB/s.

### Known limitations (DCU)

- MANS runs in P-mode only; R-mode streams cannot be decoded on the DCU (use the CPU).
- MANS does not fall back to raw storage, so incompressible data expands.
- One MANS stream from the DCU holds at most 4 GiB of raw data.
- DTK's compiler reports spurious `-Wreturn-type` warnings for `void` functions.
- The MANS DCU sources carry local performance patches (stream format unchanged);
  see `3rdparty/lossless/MANS/VENDORED.md`.

## LC payload format

```
int64          decoded size in bytes
uint16[n]      compressed size of each 16 KiB chunk (n = ceil(size / 16384))
bytes          concatenated chunk payloads
```

## 🧾 Citation

- [MANS](https://doi.org/10.1145/3712285.3759825)

## 🕰️ History

- `2026-10-24`: PFPL REL codec and H5Z-PFPL, HDF5 filter tests, examples and MPI benchmarks.
- `2026-10-24`: Merged upstream PhotonZip (MANS CPU/CUDA, autotune) with the DCU backend; MANS vendored at hpdps-group/MANS 32793bc.
- `2026-10-08`: DCU backend (LC and MANS), delta preprocessing, HDF5 filters with DCU support.
- `2026-04-12`: Add HDF5 Python support for MANS.
- `2026-04-03`: Integrated MANS CPU and GPU paths into the tensor-first Python framework.

## License

BSD 3-Clause (see `LICENSE`). The vendored components under `3rdparty/` carry
their own notices (`3rdparty/lossless/lc` and `3rdparty/lossy/pfpl`: BSD 3-Clause;
`3rdparty/lossless/MANS`: see its `LICENSE`).
