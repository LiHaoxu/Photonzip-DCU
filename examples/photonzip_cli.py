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
LOSSLESS_CODEC = "lc"
SUPPORTED_CODECS = ("lc", "mans")
MANS_MAX_BYTES = (1 << 32) - 1  # one MANS stream holds at most 4 GiB of raw data
QUALITY_LEVEL_ERROR_BOUNDS = {
    "lossless": 0.0,
    "high": 1.0,
    "low": 8.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhotonZip CLI (lossless; LC on CPU/DCU, MANS on DCU)")
    parser.add_argument("--mode", required=True, choices=("compress", "decompress", "roundtrip"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dims", nargs="+", type=int, help="Input tensor shape, e.g. --dims 127 127 127")
    parser.add_argument("--dtype", choices=("uint16", "uint32"))
    parser.add_argument(
        "--backend",
        choices=("cpu", "dcu"),
        help="Execution backend. Default for compress/roundtrip: dcu for --codec mans, cpu otherwise; "
        "for decompress: the backend recorded in the file (dcu for MANS, cpu if this build has no DCU). "
        "LC payloads are identical on both backends.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Do not initialise the DCU before timing. By default a tiny untimed call runs first, "
        "so the reported throughput excludes the one-time DCU runtime start-up (~50 ms).",
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
                             help="Throughput level (currently both values select lc).")
    level_group.add_argument("--ratio-level", choices=("high", "low"),
                             help="Compression-ratio level (currently has no effect; use --codec).")
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
        choices=("cpu", "dcu", "numpy"),
        help="Where the delta (and its inverse on decompression) runs. Default: dcu when the codec "
        "runs on the DCU and the data has 2 or more dimensions (the delta result then stays in DCU "
        "memory for the codec), cpu otherwise. All choices give the same bytes.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def dcu_available() -> bool:
    # The MANS codec is only compiled into DCU builds.
    return "mans" in photonzip.list_codecs()


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.codec is not None and args.codec not in photonzip.list_codecs():
        parser.error(f"codec {args.codec!r} is not available in this build (available: {photonzip.list_codecs()})")
    if args.backend == "dcu" and not dcu_available():
        parser.error("--backend dcu needs a DCU build (configure with -DPHOTONZIP_ENABLE_DCU=ON)")
    if args.delta_backend == "dcu" and not dcu_available():
        parser.error("--delta-backend dcu needs a DCU build (configure with -DPHOTONZIP_ENABLE_DCU=ON)")
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
        args.backend = "dcu" if codec == "mans" else "cpu"
    if codec == "mans" and args.backend != "dcu":
        parser.error("the MANS codec runs on the DCU only; use --backend dcu")
    nbytes = int(np.prod(args.dims)) * np.dtype(args.dtype).itemsize
    if codec == "mans" and nbytes > MANS_MAX_BYTES:
        parser.error(f"one MANS stream holds at most 4 GiB of raw data ({nbytes} bytes requested)")
    if args.codec is None and args.ratio_level is not None:
        warn("--ratio-level has no effect yet (the default codec lc is used); choose a codec with --codec")


def read_raw_array(path: Path, dtype_name: str, dims: list[int]) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    shape = tuple(int(dim) for dim in dims)
    array = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"Input element count mismatch: file contains {array.size} elements, but dims imply {expected}."
        )
    return array.reshape(shape)


def to_host_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.from_dlpack(value)


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
            raise ValueError(f"Unsupported file format in {path}.")
        metadata_size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(metadata_size).decode("utf-8"))
        payload = handle.read()
    if metadata.get("codec") not in SUPPORTED_CODECS:
        raise ValueError(f"{path} was written with codec {metadata.get('codec')!r}; supported codecs: {SUPPORTED_CODECS}.")
    return metadata, payload


# Throughputs are raw bytes / time. "compress" and "decompress" time the codec alone (with the
# DCU delta, its input or output is already in DCU memory, so the host<->DCU copy of the raw
# data counts as delta time); the *_with_delta lines add the delta or its inverse.
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


def resolve_delta_backend(args: argparse.Namespace, ndim: int) -> str:
    if args.delta_backend is not None:
        return args.delta_backend
    # The DCU kernel parallelises over the elements of a slice, so 1-D data stays on the CPU.
    return "dcu" if args.backend == "dcu" and ndim >= 2 else "cpu"


def prepare_compress_input(args: argparse.Namespace, array: np.ndarray, delta_backend: str | None = None):
    if args.preprocess == "delta":
        # MANS needs order-preserving differences (see photonzip.preprocess.DELTA_ENCODINGS).
        encoding = "offset" if select_lossless_codec(args) == "mans" else "int16"
        # A DCU delta feeding a DCU codec leaves its result in DCU memory: one host->DCU copy.
        on_dcu = delta_backend == "dcu" and args.backend == "dcu"
        delta_array, preprocess = apply_delta(array, encoding=encoding, backend=delta_backend, output_on_dcu=on_dcu)
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
        options = mans.MansOptions()
        codec_params = options.to_codec_params(tensor=tensor, backend=args.backend)
        return codec, options, codec_params

    raise ValueError(f"Unsupported lossless codec: {codec!r}.")


def warm_up(args: argparse.Namespace, codec: str, delta_backend: str | None = None) -> None:
    """Initialise the DCU runtime with tiny untimed calls so timings show steady-state speed."""
    if not args.warmup:
        return
    if args.backend == "dcu":
        sample = np.zeros(4096, dtype=np.uint16)
        photonzip.decompress(photonzip.compress(sample, codec=codec, backend="dcu"), backend="dcu")
    if delta_backend == "dcu":
        sample, meta = apply_delta(np.zeros((2, 4096), dtype=np.uint16), backend="dcu")
        invert_delta(sample, meta, backend="dcu")


def compress_array(args: argparse.Namespace, array: np.ndarray):
    """Delta (if requested) and compression, each timed; the compressed stream ends in host memory."""
    codec, codec_options, codec_params = make_codec_options(args, array)
    delta_backend = resolve_delta_backend(args, array.ndim) if args.preprocess == "delta" else None
    warm_up(args, codec, delta_backend)
    t0 = perf_counter()
    compress_input, preprocess = prepare_compress_input(args, array, delta_backend)
    t1 = perf_counter()
    packed = photonzip.compress(compress_input, codec=codec, backend=args.backend, codec_options=codec_options)
    if packed.device != "cpu":
        packed = packed.to_host()
    t2 = perf_counter()
    delta_elapsed = t1 - t0 if delta_backend is not None else None
    return codec, packed, codec_params, preprocess, t2 - t1, delta_elapsed, delta_backend


def decompress_array(args: argparse.Namespace, packed, preprocess: dict, ndim: int):
    """Decompression and inverse delta, each timed; returns the restored host array."""
    delta_backend = resolve_delta_backend(args, ndim) if preprocess.get("type") == "delta" else None
    t0 = perf_counter()
    if delta_backend == "dcu" and args.backend == "dcu":
        # Decode on the DCU and run the inverse delta there before the single copy back.
        packed = packed.to_device()
    restored = photonzip.decompress(packed, backend=args.backend)
    if delta_backend != "dcu" and restored.device != "cpu":
        restored = restored.to_host()
    t1 = perf_counter()
    if delta_backend is not None:
        array = invert_delta(restored, preprocess, backend=delta_backend, output_on_dcu=False)
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
    if args.backend is not None:
        backend = args.backend
    elif metadata["codec"] == "mans":
        backend = "dcu"
    else:
        backend = metadata.get("backend", "cpu") if dcu_available() else "cpu"
    if metadata["codec"] == "mans" and (backend != "dcu" or not dcu_available()):
        raise RuntimeError(f"{args.input} holds MANS data, which can only be decompressed on a DCU build with --backend dcu.")
    return backend


def run_decompress(args: argparse.Namespace) -> None:
    metadata, payload = read_container(args.input)
    args.backend = decompress_backend(args, metadata)
    preprocess = metadata.get("preprocess") or {"type": "none"}
    if preprocess.get("type") == "delta":
        warm_up(args, metadata["codec"], resolve_delta_backend(args, len(metadata["shape"])))
    else:
        warm_up(args, metadata["codec"])
    packed = _native.compressed_from_bytes(
        metadata["codec"],
        payload,
        metadata["dtype"],
        metadata["shape"],
        args.backend,
        metadata["codec_params"],
    )
    array, elapsed, delta_elapsed, delta_backend = decompress_array(args, packed, preprocess, len(metadata["shape"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output)
    print_decompress_stats(array.nbytes, elapsed, delta_elapsed, delta_backend)


def run_roundtrip(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    codec, packed, codec_params, preprocess, compress_elapsed, delta_elapsed, delta_backend = compress_array(args, array)
    restored_array, decompress_elapsed, inverse_elapsed, _ = decompress_array(args, packed, preprocess, array.ndim)
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
