"""Writes and reads a uint16 dataset through the H5Z-LC filter.

The LC pipeline is lossless and takes no parameters, so the dataset only has to name the
filter. Run ./build the project first so the plugin exists.
"""

from pathlib import Path
import os
import sys
from time import perf_counter

root = Path(__file__).resolve().parents[2]
os.environ["HDF5_PLUGIN_PATH"] = str(root / "build" / "bin" / "plugins")
sys.path.insert(0, str(root / "build" / "bindings" / "python"))
sys.path.insert(0, str(root / "python"))

import h5py
import numpy as np
import photonzip.codec.lc as lc

src = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
dst = root / "build" / "lc_compressed.h5"
x = np.fromfile(src, dtype=np.uint16).reshape(127, 127, 127)
chunks = (16, 127, 127)

t0 = perf_counter()
with h5py.File(dst, "w") as handle:
    handle.create_dataset("data", data=x, chunks=chunks, compression=lc.H5Z_FILTER_LC_ID)
t1 = perf_counter()

with h5py.File(dst, "r") as handle:
    restored = handle["data"][...]
t2 = perf_counter()

assert np.array_equal(restored, x), "H5Z-LC round trip is not lossless"

print(dst)
print(f"ratio:      {x.nbytes / dst.stat().st_size:.2f}x")
print(f"compress:   {x.nbytes / (t1 - t0) / 1e6:.2f} MB/s")
print(f"decompress: {x.nbytes / (t2 - t1) / 1e6:.2f} MB/s")
