"""DCU backend checks for PhotonZip (run on a machine with a DCU and a DCU build).

    python3 tests/test_dcu.py [raw_uint16_file] [--plugins build/bin/plugins]

Checks:
  1. LC: DCU payload is byte-identical to the CPU payload; CPU<->DCU cross decode.
  2. LC/MANS: host-memory and (if PyTorch sees the DCU) device-memory round trips.
  3. HDF5: H5Z-LC and H5Z-MANS round trips (if h5py is installed and --plugins is given).
"""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

import numpy as np


def cases(raw: np.ndarray | None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    out = {
        "tiny": np.arange(7, dtype=np.uint16),
        "one_chunk_minus": rng.integers(0, 50, 8191, dtype=np.uint16),
        "zeros_3d": np.zeros((16, 64, 64), np.uint16),
        "random": rng.integers(0, 65535, 1_000_003, dtype=np.uint16),
        "smooth_3d": (np.cumsum(rng.integers(-3, 4, (32, 128, 128)), axis=2) % 4096).astype(np.uint16),
        "u32_2d": rng.integers(0, 1000, (256, 300), dtype=np.uint32),
    }
    if raw is not None:
        out["file"] = raw
    return out


def timed(fn):
    t0 = perf_counter()
    value = fn()
    return value, perf_counter() - t0


def check_lc(photonzip, name: str, a: np.ndarray) -> None:
    cpu, _ = timed(lambda: photonzip.compress(a, codec="lc", backend="cpu"))
    dcu, t_c = timed(lambda: photonzip.compress(a, codec="lc", backend="dcu"))
    assert cpu.to_bytes() == dcu.to_bytes(), f"{name}: DCU and CPU LC payloads differ"
    r_dd, t_d = timed(lambda: np.from_dlpack(photonzip.decompress(dcu, backend="dcu")))
    r_cd = np.from_dlpack(photonzip.decompress(cpu, backend="dcu"))
    r_dc = np.from_dlpack(photonzip.decompress(dcu, backend="cpu"))
    for label, r in (("dcu->dcu", r_dd), ("cpu->dcu", r_cd), ("dcu->cpu", r_dc)):
        assert np.array_equal(r.reshape(a.shape), a), f"{name}: LC {label} mismatch"
    print(f"  lc   {name:16s} ratio {a.nbytes / dcu.nbytes:7.3f}  "
          f"c {a.nbytes / t_c / 1e9:6.2f} GB/s  d {a.nbytes / t_d / 1e9:6.2f} GB/s  OK")


def check_mans(photonzip, name: str, a: np.ndarray) -> None:
    packed, t_c = timed(lambda: photonzip.compress(a, codec="mans", backend="dcu"))
    restored, t_d = timed(lambda: np.from_dlpack(photonzip.decompress(packed, backend="dcu")))
    assert np.array_equal(restored.reshape(a.shape), a), f"{name}: MANS mismatch"
    print(f"  mans {name:16s} ratio {a.nbytes / packed.nbytes:7.3f}  "
          f"c {a.nbytes / t_c / 1e9:6.2f} GB/s  d {a.nbytes / t_d / 1e9:6.2f} GB/s  OK")


def check_device_tensors(photonzip, a: np.ndarray) -> None:
    try:
        import torch
    except ImportError:
        print("  (PyTorch not installed; skipping device-memory round trips)")
        return
    if not torch.cuda.is_available():
        print("  (PyTorch sees no DCU; skipping device-memory round trips)")
        return
    # torch has no uint16 arithmetic on older versions, but DLPack export works on int16 views.
    t = torch.from_numpy(a.view(np.int16)).to("cuda")
    dev = t.view(torch.uint16) if hasattr(torch, "uint16") else t
    if dev.dtype != getattr(torch, "uint16", None):
        print("  (this PyTorch has no uint16 dtype; skipping device-memory round trips)")
        return
    for codec in ("lc", "mans"):
        packed = photonzip.compress(dev, codec=codec, backend="dcu")
        restored = torch.from_dlpack(photonzip.decompress(packed, backend="dcu"))
        assert restored.device.type == "cuda", f"{codec}: output left the device"
        assert np.array_equal(restored.cpu().numpy().reshape(a.shape), a), f"{codec}: device round trip mismatch"
        print(f"  {codec:4s} device-memory round trip OK")


def check_hdf5(plugins: str, a: np.ndarray) -> None:
    os.environ["HDF5_PLUGIN_PATH"] = plugins
    try:
        import h5py
    except ImportError:
        print("  (h5py not installed; skipping HDF5 checks)")
        return
    import photonzip.codec.lc as lc
    import photonzip.codec.mans as mans

    vol = a.reshape(-1, 128, 128) if a.size % (128 * 128) == 0 else a
    chunks = (min(8, vol.shape[0]),) + vol.shape[1:]
    path = "photonzip_dcu_test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("lc", data=vol, chunks=chunks, compression=lc.H5Z_FILTER_LC_ID)
        f.create_dataset("mans", data=vol, chunks=chunks,
                         compression=mans.H5Z_FILTER_MANS_ID, compression_opts=(mans.MANS_MODE_P,))
    with h5py.File(path) as f:
        for key in ("lc", "mans"):
            assert np.array_equal(f[key][:], vol), f"HDF5 {key} mismatch"
            print(f"  h5z  {key:4s} ratio {vol.nbytes / f[key].id.get_storage_size():.3f} OK")
    os.remove(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", nargs="?", help="optional raw uint16 file")
    parser.add_argument("--plugins", help="directory containing libH5Z-LC.so and libH5Z-MANS.so")
    args = parser.parse_args()

    import photonzip

    codecs = photonzip.list_codecs()
    print("codecs:", codecs)
    if "mans" not in codecs:
        print("This build has no DCU support (configure with -DPHOTONZIP_ENABLE_DCU=ON).")
        return 1

    raw = np.fromfile(args.raw, dtype=np.uint16) if args.raw else None
    print("LC / MANS round trips (host memory):")
    for name, a in cases(raw).items():
        if a.dtype == np.uint16:
            check_lc(photonzip, name, a)
        if a.ndim <= 3:
            check_mans(photonzip, name, a)

    print("Device memory:")
    check_device_tensors(photonzip, cases(None)["smooth_3d"])

    if args.plugins:
        print("HDF5 filters:")
        check_hdf5(args.plugins, cases(None)["smooth_3d"])

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
