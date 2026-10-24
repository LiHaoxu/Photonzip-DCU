"""Writes and reads a float32 dataset through the H5Z-PFPL filter.

PFPL is lossy with a guaranteed relative error bound, so the interesting output is the
compression ratio together with the error the reconstruction actually incurred. Build the
project first so the plugin exists.
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
import photonzip.codec.pfpl as pfpl

src = root / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"
dst = root / "build" / "pfpl_compressed.h5"
error_bound = 1e-3

# PFPL quantises in the log domain, so the sample data is shifted off zero before the cast.
x = (np.fromfile(src, dtype=np.uint16).reshape(127, 127, 127).astype(np.float32) + 1.0)
chunks = (16, 127, 127)

t0 = perf_counter()
with h5py.File(dst, "w") as handle:
    handle.create_dataset(
        "data",
        data=x,
        chunks=chunks,
        compression=pfpl.H5Z_FILTER_PFPL_ID,
        compression_opts=pfpl.to_hdf5_compression_opts(
            pfpl.PfplOptions(error_bound=error_bound),
        ),
    )
t1 = perf_counter()

with h5py.File(dst, "r") as handle:
    restored = handle["data"][...]
t2 = perf_counter()

relative = np.abs(restored - x) / np.abs(x)
assert relative.max() <= error_bound, "H5Z-PFPL exceeded the requested error bound"

print(dst)
print(f"ratio:      {x.nbytes / dst.stat().st_size:.2f}x")
print(f"compress:   {x.nbytes / (t1 - t0) / 1e6:.2f} MB/s")
print(f"decompress: {x.nbytes / (t2 - t1) / 1e6:.2f} MB/s")
print(f"worst relative error: {relative.max():.3g} (bound {error_bound:g})")
