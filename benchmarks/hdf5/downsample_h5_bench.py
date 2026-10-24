#!/usr/bin/env python3
"""MPI HDF5 benchmark for direct resolution reduction.

The default "compression" is deliberately simple:

    reduced = source[::factor_z, ::factor_xy, ::factor_xy]

No entropy codec or HDF5 filter is used.  Each MPI rank writes its own HDF5
file, exactly like mpi_h5_bench.py.  Primary throughput is normalised to the
original uint16 source bytes; payload throughput is also reported so the
physical HDF5 transfer rate remains visible.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiff-dir",
        type=Path,
        default=Path("/data/lhx/test_data/part"),
        help="directory containing the uint16 TIFF stack",
    )
    parser.add_argument("--tiff-glob", default="*.tif")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "build" / "downsample_scratch",
    )
    parser.add_argument(
        "--mib-per-rank",
        type=int,
        default=256,
        help="original uint16 bytes assigned to each rank",
    )
    parser.add_argument("--factor-xy", type=int, default=2)
    parser.add_argument("--factor-z", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--keep-h5", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "build" / "downsample_results.csv",
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def probe_tiff_shape(tiff_dir: Path, pattern: str) -> tuple[int, int, int]:
    from PIL import Image

    paths = sorted(tiff_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matching {pattern!r} under {tiff_dir}")
    with Image.open(paths[0]) as image:
        width, height = image.size
    return len(paths), int(height), int(width)


def build_stack_cache(
    tiff_dir: Path,
    pattern: str,
    cache_path: Path,
) -> tuple[int, int, int]:
    from PIL import Image

    paths = sorted(tiff_dir.glob(pattern))
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frame = np.asarray(image)
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16)
        frames.append(frame)

    stack = np.stack(frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    stack.tofile(temporary)
    temporary.rename(cache_path)
    return tuple(int(dim) for dim in stack.shape)


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def evict_page_cache(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def write_h5(path: Path, data: np.ndarray, chunk_frames: int) -> None:
    chunks = (
        min(chunk_frames, data.shape[0]),
        data.shape[1],
        data.shape[2],
    )
    with h5py.File(path, "w", libver="latest", rdcc_nbytes=1024 * 1024) as handle:
        dataset = handle.create_dataset(
            "data",
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
        )
        dataset[...] = data
    fsync_path(path)


def read_h5(path: Path, chunk_frames: int, frame_bytes: int) -> np.ndarray:
    cache_bytes = max(8 * 1024 * 1024, chunk_frames * frame_bytes * 4)
    with h5py.File(
        path,
        "r",
        rdcc_nbytes=cache_bytes,
        rdcc_nslots=1000003,
    ) as handle:
        return handle["data"][...]


def reduce_resolution(
    source: np.ndarray,
    factor_z: int,
    factor_xy: int,
) -> np.ndarray:
    return np.ascontiguousarray(source[::factor_z, ::factor_xy, ::factor_xy])


def restore_resolution(
    reduced: np.ndarray,
    source_shape: tuple[int, int, int],
    factor_z: int,
    factor_xy: int,
) -> np.ndarray:
    """Nearest-neighbour expansion back to the original shape."""
    restored = reduced
    if factor_z != 1:
        restored = np.repeat(restored, factor_z, axis=0)
    restored = np.repeat(restored, factor_xy, axis=1)
    restored = np.repeat(restored, factor_xy, axis=2)
    return np.ascontiguousarray(
        restored[: source_shape[0], : source_shape[1], : source_shape[2]]
    )


def local_quality(
    source: np.ndarray,
    restored: np.ndarray,
) -> tuple[float, float, int]:
    """Compare the restored volume against its original."""
    max_abs = 0.0
    sum_squared = 0.0
    changed = 0

    for z_index in range(source.shape[0]):
        difference = source[z_index].astype(np.int32) - restored[z_index].astype(np.int32)
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        difference_f64 = difference.astype(np.float64)
        sum_squared += float(np.dot(difference_f64.ravel(), difference_f64.ravel()))
        changed += int(np.count_nonzero(difference))

    return max_abs, sum_squared, changed


def main() -> int:
    args = parse_args()
    if args.factor_xy < 1 or args.factor_z < 1:
        raise SystemExit("downsample factors must be positive integers")
    if args.chunk_frames < 1 or args.repeats < 1:
        raise SystemExit("chunk-frames and repeats must be positive")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()

    source_frames, height, width = probe_tiff_shape(args.tiff_dir, args.tiff_glob)
    cache_path = (
        args.work_dir
        / "cache"
        / f"stack_{source_frames}x{height}x{width}_u16.bin"
    )
    expected_cache_bytes = source_frames * height * width * np.dtype(np.uint16).itemsize

    if rank == 0:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists() or cache_path.stat().st_size != expected_cache_bytes:
            print(f"[prep] decoding TIFFs into {cache_path}", flush=True)
            build_stack_cache(args.tiff_dir, args.tiff_glob, cache_path)
    comm.Barrier()

    frame_bytes = height * width * np.dtype(np.uint16).itemsize
    bytes_per_rank = args.mib_per_rank * 1024 * 1024
    if bytes_per_rank % frame_bytes:
        raise SystemExit(
            f"--mib-per-rank must be a multiple of the "
            f"{frame_bytes / 1024 / 1024:g} MiB source frame"
        )
    frames_per_rank = bytes_per_rank // frame_bytes

    stack = np.memmap(
        cache_path,
        dtype=np.uint16,
        mode="r",
        shape=(source_frames, height, width),
    )
    offset = (rank * frames_per_rank) % source_frames
    indices = [(offset + index) % source_frames for index in range(frames_per_rank)]
    source = np.ascontiguousarray(stack[indices])
    del stack

    comm.Barrier()
    transform_start = time.perf_counter()
    reduced = reduce_resolution(source, args.factor_z, args.factor_xy)
    local_transform_sec = time.perf_counter() - transform_start
    transform_sec = comm.reduce(local_transform_sec, op=MPI.MAX, root=0)

    if rank == 0:
        print(
            f"ranks={ranks}, source/rank={source.nbytes / 2**20:.1f} MiB, "
            f"source_shape={source.shape}, reduced_shape={reduced.shape}, "
            f"factor=({args.factor_z},{args.factor_xy},{args.factor_xy}), "
            f"cold={args.cold}",
            flush=True,
        )

    rows = []
    raw_write_e2e_sec = None
    raw_read_e2e_sec = None
    for case, payload in (("raw", source), ("downsample", reduced)):
        case_dir = args.work_dir / case
        if rank == 0:
            case_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        h5_path = case_dir / f"rank{rank}.h5"

        best_write = float("inf")
        for _ in range(args.repeats):
            comm.Barrier()
            start = time.perf_counter()
            write_h5(h5_path, payload, args.chunk_frames)
            best_write = min(best_write, time.perf_counter() - start)

        best_write_e2e = best_write
        if case == "downsample":
            best_write_e2e = float("inf")
            for _ in range(args.repeats):
                comm.Barrier()
                start = time.perf_counter()
                generated = reduce_resolution(source, args.factor_z, args.factor_xy)
                write_h5(h5_path, generated, args.chunk_frames)
                best_write_e2e = min(
                    best_write_e2e,
                    time.perf_counter() - start,
                )
                del generated

        file_bytes = h5_path.stat().st_size
        payload_frame_bytes = payload.shape[1] * payload.shape[2] * payload.dtype.itemsize
        best_read = float("inf")
        read_back = None
        for _ in range(args.repeats):
            if args.cold:
                evict_page_cache(h5_path)
            comm.Barrier()
            start = time.perf_counter()
            current = read_h5(h5_path, args.chunk_frames, payload_frame_bytes)
            best_read = min(best_read, time.perf_counter() - start)
            if read_back is None:
                read_back = current
            else:
                del current

        mismatch = int(not np.array_equal(read_back, payload))

        max_abs = sum_squared = changed = 0
        local_restore_sec = 0.0
        best_read_e2e = best_read
        if case == "downsample":
            comm.Barrier()
            restore_start = time.perf_counter()
            restored = restore_resolution(
                read_back,
                source.shape,
                args.factor_z,
                args.factor_xy,
            )
            local_restore_sec = time.perf_counter() - restore_start
            max_abs, sum_squared, changed = local_quality(source, restored)
            del restored

            best_read_e2e = float("inf")
            for _ in range(args.repeats):
                if args.cold:
                    evict_page_cache(h5_path)
                comm.Barrier()
                start = time.perf_counter()
                encoded = read_h5(h5_path, args.chunk_frames, payload_frame_bytes)
                restored_e2e = restore_resolution(
                    encoded,
                    source.shape,
                    args.factor_z,
                    args.factor_xy,
                )
                best_read_e2e = min(
                    best_read_e2e,
                    time.perf_counter() - start,
                )
                del encoded
                del restored_e2e

        del read_back

        if not args.keep_h5:
            h5_path.unlink(missing_ok=True)

        write_sec = comm.reduce(best_write, op=MPI.MAX, root=0)
        read_sec = comm.reduce(best_read, op=MPI.MAX, root=0)
        write_e2e_sec = comm.reduce(best_write_e2e, op=MPI.MAX, root=0)
        read_e2e_sec = comm.reduce(best_read_e2e, op=MPI.MAX, root=0)
        restore_sec = comm.reduce(local_restore_sec, op=MPI.MAX, root=0)
        total_source = comm.reduce(source.nbytes, op=MPI.SUM, root=0)
        total_payload = comm.reduce(payload.nbytes, op=MPI.SUM, root=0)
        total_file = comm.reduce(file_bytes, op=MPI.SUM, root=0)
        total_mismatch = comm.reduce(mismatch, op=MPI.SUM, root=0)
        worst_abs = comm.reduce(max_abs, op=MPI.MAX, root=0)
        total_squared = comm.reduce(sum_squared, op=MPI.SUM, root=0)
        total_changed = comm.reduce(changed, op=MPI.SUM, root=0)
        source_min = comm.reduce(int(source.min()), op=MPI.MIN, root=0)
        source_max = comm.reduce(int(source.max()), op=MPI.MAX, root=0)

        if rank == 0:
            source_mib = total_source / 2**20
            payload_mib = total_payload / 2**20
            file_mib = total_file / 2**20
            sample_count = total_source // np.dtype(np.uint16).itemsize
            rmse = math.sqrt(total_squared / sample_count) if total_squared else 0.0
            data_range = source_max - source_min
            psnr = 20.0 * math.log10(data_range / rmse) if rmse and data_range else 0.0
            if case == "raw":
                raw_write_e2e_sec = write_e2e_sec
                raw_read_e2e_sec = read_e2e_sec
            write_speedup = raw_write_e2e_sec / write_e2e_sec
            read_speedup = raw_read_e2e_sec / read_e2e_sec
            row = {
                "tag": args.tag,
                "case": case,
                "ranks": ranks,
                "mib_per_rank": args.mib_per_rank,
                "factor_z": args.factor_z if case == "downsample" else 1,
                "factor_xy": args.factor_xy if case == "downsample" else 1,
                "repeats": args.repeats,
                "cold": int(args.cold),
                "total_source_mib": round(source_mib, 2),
                "total_payload_mib": round(payload_mib, 2),
                "total_file_mib": round(file_mib, 2),
                "size_ratio": round(total_source / total_file, 4),
                "transform_sec": round(transform_sec, 6) if case == "downsample" else "",
                "transform_source_mibs": (
                    round(source_mib / transform_sec, 2) if case == "downsample" else ""
                ),
                "write_sec": round(write_sec, 6),
                "read_sec": round(read_sec, 6),
                "write_source_mibs": round(source_mib / write_sec, 2),
                "read_source_mibs": round(source_mib / read_sec, 2),
                "write_payload_mibs": round(payload_mib / write_sec, 2),
                "read_payload_mibs": round(payload_mib / read_sec, 2),
                "restore_sec": round(restore_sec, 6) if case == "downsample" else "",
                "write_e2e_sec": round(write_e2e_sec, 6),
                "read_e2e_sec": round(read_e2e_sec, 6),
                "write_e2e_source_mibs": round(source_mib / write_e2e_sec, 2),
                "read_e2e_source_mibs": round(source_mib / read_e2e_sec, 2),
                "write_speedup_vs_raw": round(write_speedup, 4),
                "read_speedup_vs_raw": round(read_speedup, 4),
                "max_abs_error": worst_abs if case == "downsample" else "",
                "rmse": round(rmse, 6) if case == "downsample" else "",
                "psnr_db": round(psnr, 3) if case == "downsample" else "",
                "changed_percent": (
                    round(100.0 * total_changed / sample_count, 4)
                    if case == "downsample"
                    else ""
                ),
                "verify_mismatches": total_mismatch,
            }
            rows.append(row)
            quality = ""
            if case == "downsample":
                quality = (
                    f", preprocess={source_mib / transform_sec:.1f} source MiB/s, "
                    f"restore={restore_sec:.4f} s, "
                    f"RMSE={rmse:.3f}, PSNR={psnr:.2f} dB, maxerr={worst_abs:.0f}"
                )
            print(
                f"  {case:10s} ratio={total_source / total_file:.2f}x, "
                f"write={source_mib / write_sec:.1f}, "
                f"read={source_mib / read_sec:.1f} source MiB/s "
                f"(physical write/read={payload_mib / write_sec:.1f}/"
                f"{payload_mib / read_sec:.1f} MiB/s), "
                f"end-to-end write/read={source_mib / write_e2e_sec:.1f}/"
                f"{source_mib / read_e2e_sec:.1f} source MiB/s "
                f"({write_speedup:.2f}x/{read_speedup:.2f}x vs raw)"
                f"{quality}"
                + (" MISMATCH" if total_mismatch else ""),
                flush=True,
            )

    if rank == 0:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"results appended to {args.csv}", flush=True)

    comm.Barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
