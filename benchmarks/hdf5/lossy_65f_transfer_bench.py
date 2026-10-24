#!/usr/bin/env python3
"""Same 65-frame volume, same HDF5 chunk, PFPL + downsample timings for the transfer table."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "HDF5_PLUGIN_PATH",
    str(Path(__file__).resolve().parents[2] / "build" / "bin" / "plugins"),
)

import h5py  # noqa: E402
import photonzip.codec.pfpl as pfpl  # noqa: E402

SHAPE = (65, 2048, 2048)
SOURCE = Path("/workspace/lhx/test_data/part/tomo_1_stack_65.bin")
WORK = Path("/workspace/lhx/PhotonZip/build/lossy_65f_scratch")
REPEATS = 3


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def evict(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def write_raw(path: Path, data: np.ndarray, chunks: tuple[int, ...], **filter_kwargs) -> None:
    with h5py.File(path, "w", libver="latest", rdcc_nbytes=1024 * 1024) as handle:
        dataset = handle.create_dataset(
            "data",
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
            **filter_kwargs,
        )
        dataset[...] = data
    fsync_path(path)


def read_h5(path: Path, cache_bytes: int) -> np.ndarray:
    with h5py.File(path, "r", rdcc_nbytes=cache_bytes, rdcc_nslots=1000003) as handle:
        return handle["data"][...]


def downsample(source: np.ndarray, factor: int) -> np.ndarray:
    return np.ascontiguousarray(source[:, ::factor, ::factor])


def restore(reduced: np.ndarray, shape: tuple[int, int, int], factor: int) -> np.ndarray:
    restored = np.repeat(np.repeat(reduced, factor, axis=1), factor, axis=2)
    return np.ascontiguousarray(restored[:, : shape[1], : shape[2]])


def quality(source: np.ndarray, restored: np.ndarray) -> tuple[float, float, float]:
    max_abs = 0.0
    sum_sq = 0.0
    for z in range(source.shape[0]):
        diff = source[z].astype(np.float64) - restored[z].astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        sum_sq += float(np.dot(diff.ravel(), diff.ravel()))
    rmse = (sum_sq / source.size) ** 0.5
    data_range = float(source.max() - source.min())
    psnr = 20.0 * np.log10(data_range / rmse) if rmse else float("inf")
    return max_abs, rmse, psnr


def best_of(fn) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        elapsed = fn()
        best = min(best, elapsed)
    return best


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    source = np.fromfile(SOURCE, dtype=np.uint16).reshape(SHAPE)
    source_mib = source.nbytes / (1024 * 1024)
    print(f"source={source_mib:.1f} MiB shape={SHAPE} repeats={REPEATS} cold=True", flush=True)

    # PFPL REL 3e-2: uint16 -> float32 is part of the sender, round-trip back to uint16
    # is part of the receiver so the pipeline returns the original dtype.
    pfpl_path = WORK / "pfpl_3e-2.h5"
    chunks = SHAPE
    opts = pfpl.to_hdf5_compression_opts(pfpl.PfplOptions(error_bound=3e-2))

    def pfpl_sender() -> float:
        start = time.perf_counter()
        payload = np.ascontiguousarray(source, dtype=np.float32)
        write_raw(
            pfpl_path,
            payload,
            chunks,
            compression=pfpl.H5Z_FILTER_PFPL_ID,
            compression_opts=opts,
        )
        del payload
        return time.perf_counter() - start

    sender = best_of(pfpl_sender)
    file_mib = pfpl_path.stat().st_size / (1024 * 1024)
    cache = max(8 * 1024 * 1024, source.nbytes * 2)

    restored_pfpl = None

    def pfpl_receiver() -> float:
        evict(pfpl_path)
        start = time.perf_counter()
        decoded = read_h5(pfpl_path, cache)
        restored = np.ascontiguousarray(np.clip(np.rint(decoded), 0, 65535).astype(np.uint16))
        elapsed = time.perf_counter() - start
        nonlocal restored_pfpl
        if restored_pfpl is None:
            restored_pfpl = restored
        del decoded
        del restored
        return elapsed

    receiver = best_of(pfpl_receiver)
    max_abs, rmse, psnr = quality(source, restored_pfpl)
    print(
        f"  pfpl_3e-2  sender {sender:.4f}s  receiver {receiver:.4f}s  "
        f"file {file_mib:.1f} MiB  ratio {source_mib / file_mib:.2f}x  "
        f"RMSE {rmse:.3f} PSNR {psnr:.2f} dB maxerr {max_abs:.1f}",
        flush=True,
    )
    del restored_pfpl

    # 4x XY downsample, nearest-neighbour restore, same HDF5 raw write path.
    ds_path = WORK / "downsample_4x.h5"
    factor = 4

    def ds_sender() -> float:
        start = time.perf_counter()
        reduced = downsample(source, factor)
        write_raw(ds_path, reduced, reduced.shape)
        del reduced
        return time.perf_counter() - start

    ds_sender_s = best_of(ds_sender)
    ds_file_mib = ds_path.stat().st_size / (1024 * 1024)
    ds_cache = max(8 * 1024 * 1024, int(ds_file_mib * 1024 * 1024) * 4)
    restored_ds = None

    def ds_receiver() -> float:
        evict(ds_path)
        start = time.perf_counter()
        reduced = read_h5(ds_path, ds_cache)
        restored = restore(reduced, SHAPE, factor)
        elapsed = time.perf_counter() - start
        nonlocal restored_ds
        if restored_ds is None:
            restored_ds = restored
        del reduced
        del restored
        return elapsed

    ds_receiver_s = best_of(ds_receiver)
    ds_abs, ds_rmse, ds_psnr = quality(source, restored_ds)
    print(
        f"  downsample_4x  sender {ds_sender_s:.4f}s  receiver {ds_receiver_s:.4f}s  "
        f"file {ds_file_mib:.1f} MiB  ratio {source_mib / ds_file_mib:.2f}x  "
        f"RMSE {ds_rmse:.3f} PSNR {ds_psnr:.2f} dB maxerr {ds_abs:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
