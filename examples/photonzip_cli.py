from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import photonzip
import photonzip.codec.lc as lc
import photonzip.codec.mans as mans
from photonzip.preprocess import apply_delta, invert_delta

try:
    from photonzip import _native
except ImportError:
    import _native


MAGIC = b"PZC1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREAD_CSV = REPO_ROOT / "build" / "best_threads.csv"
LOSSLESS_CODEC = "mans"
SUPPORTED_CODECS = ("mans", "lc")
# Backends each codec runs on (a build has "cpu", plus "cuda" when MANS has its CUDA backend).
CODEC_BACKENDS = {"mans": ("cpu", "cuda"), "lc": ("cpu",)}
QUALITY_LEVEL_ERROR_BOUNDS = {
    "lossless": 0.0,
    "high": 1.0,
    "low": 8.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhotonZip CLI (lossless; MANS on CPU/CUDA, LC on CPU)")
    parser.add_argument("--mode", required=True, choices=("compress", "decompress", "roundtrip"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dims", nargs="+", type=int, help="Input tensor shape, e.g. --dims 127 127 127")
    parser.add_argument("--dtype", choices=("uint16", "uint32"))
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
        help="Execution backend. Default for compress/roundtrip: cpu; for decompress: the backend "
        "recorded in the file if this build has it, cpu otherwise. MANS P-mode streams decode on "
        "either backend.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Do not initialise the GPU before timing. By default a tiny untimed call runs first, "
        "so the reported throughput excludes the one-time runtime start-up.",
    )
    level_group = parser.add_argument_group("compression levels")
    level_group.add_argument(
        "--quality-level",
        dest="quality_level",
        default="lossless",
        choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS),
        help="Quality level: lossless, high, or low.",
    )
    level_group.add_argument("--throughput-level", choices=("high", "low"),
                             help="Throughput level (high selects lc).")
    level_group.add_argument("--ratio-level", choices=("high", "low"),
                             help="Compression-ratio level (high selects MANS R-mode).")
    parser.add_argument("--level", dest="quality_level", choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS), help=argparse.SUPPRESS)
    parser.add_argument("--fidelity", dest="quality_level", choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS), help=argparse.SUPPRESS)
    parser.add_argument("--speed", dest="throughput_level", choices=("high", "low"), help=argparse.SUPPRESS)
    parser.add_argument("--ratio", dest="ratio_level", choices=("high", "low"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--codec",
        choices=SUPPORTED_CODECS,
        help="Force a codec instead of selecting one from the quality/throughput level.",
    )
    parser.add_argument(
        "--delta",
        dest="preprocess",
        action="store_const",
        const="delta",
        help="Apply global inter-slice delta preprocessing (axis 0) before compression "
        "(stored as int16 bit patterns for LC, as difference + 32768 for MANS).",
    )
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        choices=("none", "delta"),
        help="Global preprocessing applied before compression (default: none).",
    )
    parser.add_argument(
        "--delta-backend",
        choices=("cpu", "numpy"),
        help="Where the delta (and its inverse on decompression) runs: cpu (OpenMP, default) or "
        "numpy. Both give the same bytes.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def available_backends() -> list[str]:
    return photonzip.available_backends()


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    backends = available_backends()
    if args.backend is not None and args.backend not in backends:
        parser.error(f"--backend {args.backend} is not available in this build (build the vendored "
                     "MANS with its NVIDIA backend: -DPHOTONZIP_MANS_TARGET_PLATFORM=cpu_nv)")
    if args.mode not in ("compress", "roundtrip"):
        return
    if not args.dims:
        parser.error("--dims is required for compress and roundtrip modes")
    if args.dtype is None:
        parser.error("--dtype is required for compress and roundtrip modes")
    if not 1 <= len(args.dims) <= 3 or min(args.dims) <= 0:
        parser.error("--dims takes 1 to 3 positive sizes (C order, slowest dimension first)")
    if args.quality_level != "lossless":
        parser.error(
            f"lossy compression is not wired yet (--quality-level {args.quality_level}, "
            f"preset error bound {QUALITY_LEVEL_ERROR_BOUNDS[args.quality_level]})"
        )
    if args.preprocess == "delta" and args.dtype != "uint16":
        parser.error("--delta supports --dtype uint16 only")

    codec = select_lossless_codec(args)
    if args.backend is None:
        args.backend = "cpu"
    if args.backend not in CODEC_BACKENDS[codec]:
        parser.error(f"the {codec} codec runs on {', '.join(CODEC_BACKENDS[codec])}; not on --backend {args.backend}")
    if codec == "lc" and args.dtype != "uint16":
        parser.error("the lc codec supports --dtype uint16 only")
    if args.backend == "cuda":
        try:
            import torch
        except ImportError:
            parser.error("--backend cuda needs PyTorch")
        if not torch.cuda.is_available():
            parser.error("--backend cuda was requested, but torch.cuda.is_available() is False")


def read_raw_array(path: Path, dtype_name: str, dims: list[int]) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    shape = tuple(int(dim) for dim in dims)
    expected_items = int(np.prod(shape, dtype=np.int64))
    array = np.fromfile(path, dtype=dtype)
    if array.size != expected_items:
        raise ValueError(
            f"Input element count mismatch for {path}: expected {expected_items}, got {array.size}."
        )
    return array.reshape(shape)


def to_backend_tensor(array, backend: str):
    """Host input for cuda becomes a CUDA tensor (outside the timed region)."""
    if backend != "cuda" or not isinstance(array, np.ndarray):
        return array
    import torch

    return torch.from_numpy(np.ascontiguousarray(array)).cuda()


def to_host_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.from_dlpack(value)


def resolve_thread_csv() -> Path | None:
    if DEFAULT_THREAD_CSV.exists():
        return DEFAULT_THREAD_CSV
    return None


def ensure_autotuned_csv() -> Path:
    csv_path = resolve_thread_csv()
    if csv_path is not None:
        print(f"thread_csv: {csv_path}")
        return csv_path

    print(f"thread_csv_missing: {DEFAULT_THREAD_CSV}")
    print("autotune: start")
    result = mans.autotune(
        mans.MansAutotuneOptions(
            data_size_mb_list=(4.0 / 1024.0, 8.0 / 1024.0, 16.0 / 1024.0, 32.0 / 1024.0, 1.0, 4.0),
            threads_min=1,
            threads_max=64,
            iter=3,
            verbose=False,
        )
    )
    DEFAULT_THREAD_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.thread_table.to_csv(DEFAULT_THREAD_CSV)
    print(f"autotune_csv: {DEFAULT_THREAD_CSV}")
    print(f"thread_rows: {len(result.thread_table.rows)}")
    return DEFAULT_THREAD_CSV


def resolve_lossless_mans_mode(args: argparse.Namespace) -> str:
    if args.ratio_level == "high":
        return "r"
    return "p"


def write_container(path: Path, *, codec: str, backend: str, dtype: str, shape: tuple[int, ...], codec_params: list[int], payload: bytes, preprocess: dict | None = None) -> None:
    metadata = {
        "codec": codec,
        "backend": backend,
        "dtype": dtype,
        "shape": [int(dim) for dim in shape],
        "codec_params": [int(value) for value in codec_params],
        "preprocess": preprocess or {"type": "none"},
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", len(metadata_bytes)))
        handle.write(metadata_bytes)
        handle.write(payload)


def read_container(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a PhotonZip container.")
        metadata_size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(metadata_size).decode("utf-8"))
        payload = handle.read()
    if metadata.get("codec") not in SUPPORTED_CODECS:
        raise ValueError(f"{path} was written with codec {metadata.get('codec')!r}; supported codecs: {SUPPORTED_CODECS}.")
    return metadata, payload


# Throughputs are raw bytes / time. "compress" and "decompress" time the codec alone; the
# *_with_delta lines add the delta or its inverse.
def print_delta_stats(label: str, nbytes: int, codec_elapsed: float, delta_elapsed: float | None, backend: str | None) -> None:
    if delta_elapsed is not None:
        name = "delta" if label == "compress" else "inverse_delta"
        print(f"{name}: {delta_elapsed:.4f} s ({backend})")
        print(f"{label}_with_delta: {nbytes / (codec_elapsed + delta_elapsed) / 1e6:.2f} MB/s")


def print_compress_stats(nbytes: int, elapsed: float, packed_nbytes: int, delta_elapsed: float | None = None, delta_backend: str | None = None) -> None:
    print(f"compress: {nbytes / elapsed / 1e6:.2f} MB/s")
    print_delta_stats("compress", nbytes, elapsed, delta_elapsed, delta_backend)
    print(f"ratio: {nbytes / packed_nbytes:.3f}x")


def print_decompress_stats(nbytes: int, elapsed: float, delta_elapsed: float | None = None, delta_backend: str | None = None) -> None:
    print(f"decompress: {nbytes / elapsed / 1e6:.2f} MB/s")
    print_delta_stats("decompress", nbytes, elapsed, delta_elapsed, delta_backend)


def print_roundtrip_stats(nbytes: int, compress_elapsed: float, decompress_elapsed: float, packed_nbytes: int, is_equal: bool, delta_elapsed: tuple[float, float] | None = None, delta_backend: str | None = None) -> None:
    print_compress_stats(nbytes, compress_elapsed, packed_nbytes, delta_elapsed and delta_elapsed[0], delta_backend)
    print_decompress_stats(nbytes, decompress_elapsed, delta_elapsed and delta_elapsed[1], delta_backend)
    print(f"is_equal: {is_equal}")


def resolve_delta_backend(args: argparse.Namespace) -> str:
    return args.delta_backend or "cpu"


def prepare_compress_input(args: argparse.Namespace, array: np.ndarray, delta_backend: str | None = None):
    if args.preprocess == "delta":
        # MANS needs order-preserving differences (see photonzip.preprocess.DELTA_ENCODINGS).
        encoding = "offset" if select_lossless_codec(args) == "mans" else "int16"
        delta_array, preprocess = apply_delta(array, encoding=encoding, backend=delta_backend)
        if not preprocess["lossless"]:
            print(
                f"delta_overflow_pixels: {preprocess['overflow_pixels']} "
                "(clipped, result is NOT strictly lossless)"
            )
        return delta_array, preprocess
    return array, {"type": "none"}


def select_lossless_codec(args: argparse.Namespace) -> str:
    if args.codec is not None:
        return args.codec
    if args.throughput_level == "high":
        return "lc"
    if args.ratio_level == "high":
        return "mans"
    return LOSSLESS_CODEC


def make_codec_options(args: argparse.Namespace, tensor) -> tuple[str, object, list[int]]:
    if args.quality_level != "lossless":
        error_bound = QUALITY_LEVEL_ERROR_BOUNDS[args.quality_level]
        raise NotImplementedError(
            f"Lossy compression is not wired yet for quality_level={args.quality_level!r} "
            f"(preset error_bound={error_bound})."
        )

    codec = select_lossless_codec(args)
    if codec == "lc":
        options = lc.LcOptions()
        codec_params = options.to_codec_params(tensor=tensor, backend=args.backend)
        return codec, options, codec_params
    if codec == "mans":
        csv_path = ensure_autotuned_csv()
        options = mans.MansOptions(mode=resolve_lossless_mans_mode(args), thread_csv_path=csv_path)
        codec_params = options.to_codec_params(tensor=tensor, backend=args.backend)
        return codec, options, codec_params

    raise ValueError(f"Unsupported lossless codec: {codec!r}.")


def warm_up(args: argparse.Namespace, codec: str) -> None:
    """Initialise the GPU runtime with a tiny untimed call so timings show steady-state speed."""
    if not args.warmup or args.backend != "cuda":
        return
    sample = np.zeros(4096, dtype=np.uint16)
    options = mans.MansOptions(mode="p", adm_compress_thread=32, adm_decompress_thread=32) if codec == "mans" else None
    photonzip.decompress(photonzip.compress(sample, codec=codec, backend=args.backend, codec_options=options),
                         backend=args.backend)


def compress_array(args: argparse.Namespace, array: np.ndarray):
    """Delta (if requested) and compression, each timed; the compressed stream ends in host memory."""
    codec, codec_options, codec_params = make_codec_options(args, array)
    delta_backend = resolve_delta_backend(args) if args.preprocess == "delta" else None
    warm_up(args, codec)
    t0 = perf_counter()
    compress_input, preprocess = prepare_compress_input(args, array, delta_backend)
    delta_elapsed = perf_counter() - t0 if delta_backend is not None else None
    # The host->GPU copy for --backend cuda is not timed.
    compress_input = to_backend_tensor(compress_input, args.backend)
    t1 = perf_counter()
    packed = photonzip.compress(compress_input, codec=codec, backend=args.backend, codec_params=codec_params)
    if packed.device != "cpu":
        packed = packed.to_host()
    t2 = perf_counter()
    return codec, packed, codec_params, preprocess, t2 - t1, delta_elapsed, delta_backend


def decompress_array(args: argparse.Namespace, packed, preprocess: dict):
    """Decompression and inverse delta, each timed; returns the restored host array."""
    delta_backend = resolve_delta_backend(args) if preprocess.get("type") == "delta" else None
    t0 = perf_counter()
    restored = photonzip.decompress(packed, backend=args.backend)
    if restored.device != "cpu":
        restored = restored.to_host()
    t1 = perf_counter()
    if delta_backend is not None:
        array = invert_delta(restored, preprocess, backend=delta_backend)
    else:
        array = to_host_numpy(restored)
    t2 = perf_counter()
    return array, t1 - t0, (t2 - t1 if delta_backend is not None else None), delta_backend


def run_compress(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    codec, packed, codec_params, preprocess, elapsed, delta_elapsed, delta_backend = compress_array(args, array)
    write_container(
        args.output,
        codec=codec,
        backend=args.backend,
        dtype=args.dtype,
        shape=array.shape,
        codec_params=codec_params,
        payload=packed.to_bytes(),
        preprocess=preprocess,
    )
    print_compress_stats(array.nbytes, elapsed, packed.nbytes, delta_elapsed, delta_backend)


def decompress_backend(args: argparse.Namespace, metadata: dict) -> str:
    backends = available_backends()
    if args.backend is not None:
        backend = args.backend
    else:
        recorded = metadata.get("backend", "cpu")
        backend = recorded if recorded in backends else "cpu"
    codec = metadata["codec"]
    if backend not in CODEC_BACKENDS[codec]:
        raise RuntimeError(f"the {codec} codec cannot decompress on --backend {backend}.")
    return backend


def run_decompress(args: argparse.Namespace) -> None:
    metadata, payload = read_container(args.input)
    args.backend = decompress_backend(args, metadata)
    preprocess = metadata.get("preprocess") or {"type": "none"}
    warm_up(args, metadata["codec"])
    packed = _native.compressed_from_bytes(
        metadata["codec"],
        payload,
        metadata["dtype"],
        metadata["shape"],
        args.backend,
        metadata["codec_params"],
    )
    array, elapsed, delta_elapsed, delta_backend = decompress_array(args, packed, preprocess)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output)
    print_decompress_stats(array.nbytes, elapsed, delta_elapsed, delta_backend)


def run_roundtrip(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    codec, packed, codec_params, preprocess, compress_elapsed, delta_elapsed, delta_backend = compress_array(args, array)
    restored_array, decompress_elapsed, inverse_elapsed, _ = decompress_array(args, packed, preprocess)
    write_container(
        args.output,
        codec=codec,
        backend=args.backend,
        dtype=args.dtype,
        shape=array.shape,
        codec_params=codec_params,
        payload=packed.to_bytes(),
        preprocess=preprocess,
    )
    print_roundtrip_stats(
        array.nbytes, compress_elapsed, decompress_elapsed, packed.nbytes,
        np.array_equal(restored_array.reshape(array.shape), array),
        (delta_elapsed, inverse_elapsed) if delta_elapsed is not None else None, delta_backend,
    )


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "compress":
            run_compress(args)
        elif args.mode == "decompress":
            run_decompress(args)
        else:
            run_roundtrip(args)
    except (ValueError, RuntimeError, OSError, _native.PhotonZipError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
