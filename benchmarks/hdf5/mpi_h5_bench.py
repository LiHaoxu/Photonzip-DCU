#!/usr/bin/env python3
"""MPI HDF5 read/write benchmark for the PhotonZip filters.

Compares plain HDF5 against H5Z-LC, H5Z-MANS and H5Z-PFPL on the same slab of data.

Design notes that matter for the numbers this produces:

* File per rank. Every rank owns ``rank<N>.h5`` and drives serial HDF5 on it, so
  no MPI-enabled HDF5/h5py build is needed. MPI is used only to start the ranks
  together, hold barriers around the timed regions, and reduce the results.
* One ``H5Dwrite``/``H5Dread`` per rank per measurement. Looping over frames in
  Python costs one HDF5 call plus one filter invocation per frame and is the
  single biggest reason a naive version of this benchmark looks slow.
* Explicit MANS thread counts. The filter's ``set_local`` only consults
  ``best_threads.csv`` when it receives the short one-word ``cd_values``; its
  fallback is 32 threads, which means 64 ranks would ask for 2048 threads on a
  64-core node. Passing a full 9-word ``MansParams`` bypasses that entirely.
* PFPL is float-only and lossy. Its ``can_apply`` rejects integer types, so the
  uint16 frames are promoted to float32 before writing. Compression ratios are
  reported against both the float32 payload and the uint16 original, and
  throughput is always quoted per uint16 source byte so every case is comparable.

The cd_values come from the codec modules in ``python/photonzip/codec`` rather than
being rebuilt here, so the benchmark and the library cannot drift apart.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build" / "bindings" / "python"))
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ.setdefault("HDF5_PLUGIN_PATH", str(REPO_ROOT / "build" / "bin" / "plugins"))

import h5py  # noqa: E402  (must follow HDF5_PLUGIN_PATH)

import photonzip.codec.lc as lc  # noqa: E402
import photonzip.codec.mans as mans  # noqa: E402
import photonzip.codec.pfpl as pfpl  # noqa: E402

DEFAULT_STACK = REPO_ROOT / "testdata" / "u2" / "sfc-gi" / "sfc-gi_127x127x127_4096kB.u2"

# `dtype` is what actually lands in the HDF5 file. PFPL rejects anything that is not
# H5T_FLOAT, so the uint16 frames have to be promoted to float32 for that case.
CASES = {
    "raw": {"codec": None, "dtype": "uint16"},
    "lc": {"codec": "lc", "dtype": "uint16"},
    "mans_r": {"codec": "mans", "mode": "r", "dtype": "uint16"},
    "mans_p": {"codec": "mans", "mode": "p", "dtype": "uint16"},
    "raw_f32": {"codec": None, "dtype": "float32"},
    "pfpl_rel": {"codec": "pfpl", "quality": 1e-3, "dtype": "float32"},
}

LOSSY_CODECS = {"pfpl"}


def case_filter_kwargs(case: dict, payload: np.ndarray, chunks: tuple[int, ...], threads: int) -> dict:
    codec = case["codec"]
    if codec is None:
        return {}
    if codec == "lc":
        # The LC pipeline is parameterless: the decoded size lives in the payload header.
        return {"compression": lc.H5Z_FILTER_LC_ID}
    if codec == "mans":
        return {
            "compression": mans.H5Z_FILTER_MANS_ID,
            "compression_opts": mans.to_hdf5_compression_opts(
                payload,
                chunks=chunks,
                options=mans.MansOptions(
                    mode=case["mode"],
                    adm_compress_thread=threads,
                    adm_decompress_thread=threads,
                    warn_on_default_threads=False,
                ),
            ),
        }
    if codec == "pfpl":
        return {
            "compression": pfpl.H5Z_FILTER_PFPL_ID,
            "compression_opts": pfpl.to_hdf5_compression_opts(
                pfpl.PfplOptions(error_bound=case["quality"]),
            ),
        }
    raise ValueError(f"unknown codec: {codec}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK,
                        help="flat uint16 volume; the shape is read from a NxNxN in the file name")
    parser.add_argument("--stack-shape", default="",
                        help="explicit 'd,h,w' for --stack when the name does not carry one")
    parser.add_argument("--tiff-dir", type=Path, default=None,
                        help="decode a TIFF stack instead of --stack (needs Pillow)")
    parser.add_argument("--tiff-glob", default="*.tif")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "build" / "bench_scratch")
    parser.add_argument("--cases", default="raw,lc,mans_r,pfpl_rel",
                        help="comma-separated subset of " + ",".join(CASES))
    parser.add_argument("--mib-per-rank", type=int, default=256,
                        help="slab size each rank writes; the source stack is tiled to reach it")
    parser.add_argument("--chunk-frames", type=int, default=2, help="chunk shape is (chunk-frames, H, W)")
    parser.add_argument("--chunk-shape", default="",
                        help="explicit 'd,h,w' chunk overriding --chunk-frames")
    parser.add_argument("--threads", type=int, default=0,
                        help="MANS OpenMP threads per rank; 0 auto-splits the available cores")
    parser.add_argument("--repeats", type=int, default=3, help="timed iterations; the best one is reported")
    parser.add_argument("--cold", action="store_true",
                        help="drop each file from the page cache before every read")
    parser.add_argument("--verify", action="store_true", help="check the round-trip on the first repeat")
    parser.add_argument("--codec-only", action="store_true",
                        help="also time the filter alone through the in-memory core VFD")
    parser.add_argument("--pfpl-eb", type=float, default=None,
                        help="override the PFPL relative error bound")
    parser.add_argument("--keep-h5", action="store_true", help="do not delete the generated HDF5 files")
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "build" / "bench_results.csv")
    parser.add_argument("--tag", default="", help="free-form label copied into the CSV")
    return parser.parse_args(argv)


def stack_shape_from_name(path: Path) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)x(\d+)x(\d+)", path.name)
    return tuple(int(group) for group in match.groups()) if match else None


def build_stack_cache(tiff_dir: Path, glob: str, cache_path: Path) -> tuple[int, int, int]:
    """Decode the TIFF stack once into a flat uint16 file all ranks can memmap."""
    from PIL import Image

    paths = sorted(tiff_dir.glob(glob))
    if not paths:
        raise SystemExit(f"no files matching {glob!r} under {tiff_dir}")

    frames = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.dtype != np.uint16:
            array = array.astype(np.uint16)
        frames.append(array)

    stack = np.stack(frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    stack.tofile(tmp_path)
    tmp_path.rename(cache_path)
    return tuple(int(dim) for dim in stack.shape)


def probe_tiff_shape(tiff_dir: Path, glob: str) -> tuple[int, int, int]:
    from PIL import Image

    paths = sorted(tiff_dir.glob(glob))
    if not paths:
        raise SystemExit(f"no files matching {glob!r} under {tiff_dir}")
    with Image.open(paths[0]) as image:
        width, height = image.size
    return len(paths), int(height), int(width)


def evict_page_cache(path: Path) -> None:
    """Unprivileged page-cache eviction, unlike /proc/sys/vm/drop_caches."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path: Path, slab: np.ndarray, chunks: tuple[int, ...], filter_kwargs: dict) -> None:
    # The chunk cache is bypassed for whole-chunk writes, so keep it small and
    # let HDF5 hand each filtered chunk straight to the file.
    with h5py.File(path, "w", libver="latest", rdcc_nbytes=1024 * 1024) as handle:
        dset = handle.create_dataset(
            "data", shape=slab.shape, dtype=slab.dtype, chunks=chunks, **filter_kwargs
        )
        dset[...] = slab
    fsync_path(path)


def read_once(path: Path, rdcc_bytes: int) -> np.ndarray:
    with h5py.File(path, "r", rdcc_nbytes=rdcc_bytes, rdcc_nslots=1000003) as handle:
        return handle["data"][...]


def compress_only(slab: np.ndarray, chunks: tuple[int, ...], filter_kwargs: dict) -> tuple[float, int]:
    """Time the filter pipeline with the core VFD so no disk I/O is included."""
    name = f"mem-{os.getpid()}.h5"
    start = time.perf_counter()
    with h5py.File(name, "w", driver="core", backing_store=False, rdcc_nbytes=1024 * 1024) as handle:
        dset = handle.create_dataset(
            "data", shape=slab.shape, dtype=slab.dtype, chunks=chunks, **filter_kwargs
        )
        dset[...] = slab
        handle.flush()
        elapsed = time.perf_counter() - start
        nbytes = dset.id.get_storage_size()
    return elapsed, int(nbytes)


def decompress_only(path: Path, rdcc_bytes: int) -> float:
    """Same idea for the read side: the core VFD pulls the file into RAM at open."""
    with h5py.File(path, "r", driver="core", rdcc_nbytes=rdcc_bytes, rdcc_nslots=1000003) as handle:
        dset = handle["data"]
        start = time.perf_counter()
        data = dset[...]
        elapsed = time.perf_counter() - start
        del data
    return elapsed


def resolve_source(args: argparse.Namespace, comm, rank: int) -> tuple[Path, tuple[int, int, int]]:
    """Returns the flat uint16 file every rank memmaps, plus its shape."""
    if args.tiff_dir is not None:
        n_src, height, width = probe_tiff_shape(args.tiff_dir, args.tiff_glob)
        cache_path = args.work_dir / "cache" / f"stack_{n_src}x{height}x{width}_u16.bin"
        expected_bytes = n_src * height * width * 2
        if rank == 0:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists() or cache_path.stat().st_size != expected_bytes:
                print(f"[prep] decoding {n_src} TIFFs into {cache_path}", flush=True)
                build_stack_cache(args.tiff_dir, args.tiff_glob, cache_path)
        comm.Barrier()
        return cache_path, (n_src, height, width)

    if not args.stack.is_file():
        raise SystemExit(f"source volume not found: {args.stack}")
    if args.stack_shape:
        shape = tuple(int(part) for part in args.stack_shape.split(","))
        if len(shape) != 3:
            raise SystemExit(f"--stack-shape needs three comma-separated values, got {args.stack_shape!r}")
    else:
        shape = stack_shape_from_name(args.stack)
        if shape is None:
            raise SystemExit(f"cannot infer the shape of {args.stack.name}; pass --stack-shape")
    expected_bytes = shape[0] * shape[1] * shape[2] * 2
    if args.stack.stat().st_size != expected_bytes:
        raise SystemExit(
            f"{args.stack.name} is {args.stack.stat().st_size} bytes but shape {shape} needs {expected_bytes}"
        )
    return args.stack, shape


def main() -> int:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    cases = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = [name for name in cases if name not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; valid names are {sorted(CASES)}")

    threads = args.threads if args.threads > 0 else max(1, len(os.sched_getaffinity(0)) // nprocs)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    work_dir = args.work_dir
    if rank == 0:
        work_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    cache_path, (n_src, height, width) = resolve_source(args, comm, rank)

    frame_bytes = height * width * 2
    target_bytes = args.mib_per_rank * 1024 * 1024
    # Round up to whole frames. This keeps the benchmark usable with the bundled
    # 127x127x127 sample, whose frame size cannot divide an integral MiB exactly.
    frames_per_rank = max(1, math.ceil(target_bytes / frame_bytes))

    # Each rank starts at a different frame so no two ranks compress identical bytes.
    stack = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(n_src, height, width))
    offset = (rank * frames_per_rank) % n_src
    indices = [(offset + i) % n_src for i in range(frames_per_rank)]
    slab = np.ascontiguousarray(stack[indices])
    del stack

    if args.chunk_shape:
        chunks = tuple(int(part) for part in args.chunk_shape.split(","))
        if len(chunks) != 3:
            raise SystemExit(f"--chunk-shape needs three comma-separated values, got {args.chunk_shape!r}")
        chunks = (min(chunks[0], frames_per_rank), min(chunks[1], height), min(chunks[2], width))
    else:
        chunks = (min(args.chunk_frames, frames_per_rank), height, width)
    chunk_frames = chunks[0]
    chunk_elems = chunks[0] * chunks[1] * chunks[2]
    logical_bytes = slab.nbytes
    logical_mib_per_rank = logical_bytes / (1024 * 1024)

    if rank == 0:
        print(
            f"\nranks={nprocs} threads/rank={threads} slab={logical_mib_per_rank:.2f} MiB "
            f"({frames_per_rank}x{height}x{width}) "
            f"total={nprocs * logical_mib_per_rank / 1024:.2f} GiB "
            f"chunk={chunks} ({chunk_elems * 2 / 1024 / 1024:g} MiB as uint16) "
            f"repeats={args.repeats} cold={args.cold}",
            flush=True,
        )

    slab_f32 = None
    rows = []
    for case in cases:
        spec = CASES[case]
        if args.pfpl_eb is not None and spec["codec"] == "pfpl":
            spec = {**spec, "quality": args.pfpl_eb}
        if spec["dtype"] == "float32":
            if slab_f32 is None:
                # Match the validated benchmark: promote the uint16 samples without
                # changing their values. PFPL REL has an explicit representation for zero.
                slab_f32 = slab.astype(np.float32)
            payload = slab_f32
        else:
            payload = slab
        filter_kwargs = case_filter_kwargs(spec, payload, chunks, threads)
        payload_bytes = payload.nbytes
        chunk_mib = chunk_elems * payload.dtype.itemsize / (1024 * 1024)

        case_dir = work_dir / case
        if rank == 0:
            case_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        h5_path = case_dir / f"rank{rank}.h5"

        best_write = float("inf")
        for _ in range(args.repeats):
            comm.Barrier()
            start = time.perf_counter()
            write_once(h5_path, payload, chunks, filter_kwargs)
            best_write = min(best_write, time.perf_counter() - start)

        file_bytes = h5_path.stat().st_size

        best_read = float("inf")
        rdcc_bytes = max(8 * 1024 * 1024, int(chunk_mib * 1024 * 1024) * 4)
        mismatch = 0
        max_rel_err = 0.0
        sq_err = 0.0
        for iteration in range(args.repeats):
            if args.cold:
                evict_page_cache(h5_path)
            comm.Barrier()
            start = time.perf_counter()
            data = read_once(h5_path, rdcc_bytes)
            best_read = min(best_read, time.perf_counter() - start)
            if args.verify and iteration == 0:
                if spec["codec"] in LOSSY_CODECS:
                    # Both sides are float32, so this subtraction is exact and avoids
                    # materialising a float64 copy of the whole slab.
                    diff = np.subtract(data, payload)
                    sq_err = float(np.dot(diff.ravel(), diff.ravel()))
                    # PFPL bounds the error relative to each sample, so the relative
                    # figure is the one that can be checked against the request.
                    max_rel_err = float(np.max(np.abs(diff) / np.abs(payload)))
                    del diff
                elif not np.array_equal(data, payload):
                    mismatch = 1
            del data

        codec_write = codec_read = float("nan")
        if args.codec_only:
            codec_write = min(compress_only(payload, chunks, filter_kwargs)[0] for _ in range(args.repeats))
            codec_read = min(decompress_only(h5_path, rdcc_bytes) for _ in range(args.repeats))

        if not args.keep_h5:
            h5_path.unlink(missing_ok=True)

        write_sec = comm.reduce(best_write, op=MPI.MAX, root=0)
        read_sec = comm.reduce(best_read, op=MPI.MAX, root=0)
        codec_write_sec = comm.reduce(codec_write, op=MPI.MAX, root=0)
        codec_read_sec = comm.reduce(codec_read, op=MPI.MAX, root=0)
        total_payload = comm.reduce(payload_bytes, op=MPI.SUM, root=0)
        total_source = comm.reduce(logical_bytes, op=MPI.SUM, root=0)
        total_file = comm.reduce(file_bytes, op=MPI.SUM, root=0)
        total_mismatch = comm.reduce(mismatch, op=MPI.SUM, root=0)
        worst_rel = comm.reduce(max_rel_err, op=MPI.MAX, root=0)
        total_sq_err = comm.reduce(sq_err, op=MPI.SUM, root=0)
        data_hi = comm.reduce(float(slab.max()), op=MPI.MAX, root=0)
        data_lo = comm.reduce(float(slab.min()), op=MPI.MIN, root=0)

        if rank == 0:
            n_samples = total_payload / 4
            rmse = math.sqrt(total_sq_err / n_samples) if total_sq_err else 0.0
            psnr = 20.0 * math.log10((data_hi - data_lo) / rmse) if rmse else 0.0
            payload_mib = total_payload / (1024 * 1024)
            source_mib = total_source / (1024 * 1024)
            file_mib = total_file / (1024 * 1024)
            # Throughput is quoted per uint16 source byte throughout. The float32
            # promotion the lossy codecs need is an artefact of those filters, not
            # data the user asked to store, so charging them for it would make the
            # numbers incomparable with the uint16 cases.
            write_mibs = source_mib / write_sec
            read_mibs = source_mib / read_sec
            ratio = total_payload / total_file if total_file else 0.0
            ratio_u16 = total_source / total_file if total_file else 0.0
            row = {
                "tag": args.tag,
                "case": case,
                "dtype": spec["dtype"],
                "ranks": nprocs,
                "threads_per_rank": threads,
                "mib_per_rank": round(logical_mib_per_rank, 3),
                "chunk_frames": chunk_frames,
                "chunk_mib": round(chunk_mib, 3),
                "repeats": args.repeats,
                "cold": int(args.cold),
                "total_source_mib": round(source_mib, 1),
                "total_payload_mib": round(payload_mib, 1),
                "total_file_mib": round(file_mib, 1),
                "comp_ratio_vs_payload": round(ratio, 4),
                "comp_ratio": round(ratio_u16, 4),
                "write_sec": round(write_sec, 4),
                "read_sec": round(read_sec, 4),
                "write_mibs": round(write_mibs, 2),
                "read_mibs": round(read_mibs, 2),
                "write_mibs_per_rank": round(write_mibs / nprocs, 2),
                "read_mibs_per_rank": round(read_mibs / nprocs, 2),
                "write_mibs_payload": round(payload_mib / write_sec, 2),
                "read_mibs_payload": round(payload_mib / read_sec, 2),
                "codec_write_mibs": round(source_mib / codec_write_sec, 2) if args.codec_only else "",
                "codec_read_mibs": round(source_mib / codec_read_sec, 2) if args.codec_only else "",
                "pfpl_eb": spec.get("quality", "") if spec["codec"] == "pfpl" else "",
                "max_rel_err": worst_rel if args.verify else "",
                "rmse": round(rmse, 6) if rmse else "",
                "psnr_db": round(psnr, 2) if psnr else "",
                "verify_mismatches": total_mismatch,
            }
            rows.append(row)
            extra = ""
            if args.codec_only:
                extra = (f"   codec {source_mib / codec_write_sec:8.1f}/"
                         f"{source_mib / codec_read_sec:8.1f} MiB/s")
            if args.verify and spec["codec"] in LOSSY_CODECS:
                extra += f"   maxrel {worst_rel:<9.3g} rmse {rmse:<8.4g} psnr {psnr:5.1f} dB"
            print(
                f"  {case:11s} write {write_mibs:9.1f} MiB/s ({write_sec:6.3f} s)   "
                f"read {read_mibs:9.1f} MiB/s ({read_sec:6.3f} s)   "
                f"ratio {ratio_u16:5.2f}x" + extra
                + ("   MISMATCH" if total_mismatch else ""),
                flush=True,
            )

    if rank == 0 and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv.exists()
        with args.csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    comm.Barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
