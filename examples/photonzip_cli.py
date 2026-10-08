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
        help="Apply global inter-slice delta preprocessing (axis 0) before compression.",
    )
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        choices=("none", "delta"),
        help="Global preprocessing applied before compression (default: none).",
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
    if codec == "mans" and args.preprocess == "delta":
        warn("--delta stores signed differences as uint16 bit patterns, which usually lowers the MANS ratio")


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


def print_compress_stats(nbytes: int, elapsed: float, packed_nbytes: int) -> None:
    print(f"compress: {nbytes / elapsed / 1e6:.2f} MB/s")
    print(f"ratio: {nbytes / packed_nbytes:.3f}x")


def print_decompress_stats(nbytes: int, elapsed: float) -> None:
    print(f"decompress: {nbytes / elapsed / 1e6:.2f} MB/s")


def print_roundtrip_stats(nbytes: int, compress_elapsed: float, decompress_elapsed: float, packed_nbytes: int, is_equal: bool) -> None:
    print(f"compress: {nbytes / compress_elapsed / 1e6:.2f} MB/s")
    print(f"decompress: {nbytes / decompress_elapsed / 1e6:.2f} MB/s")
    print(f"ratio: {nbytes / packed_nbytes:.3f}x")
    print(f"is_equal: {is_equal}")


def prepare_compress_input(args: argparse.Namespace, array: np.ndarray) -> tuple[np.ndarray, dict]:
    if args.preprocess == "delta":
        delta_array, preprocess = apply_delta(array)
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


def warm_up(args: argparse.Namespace, codec: str) -> None:
    """Initialise the DCU runtime with a tiny untimed call so timings show steady-state speed."""
    if args.backend != "dcu" or not args.warmup:
        return
    sample = np.zeros(4096, dtype=np.uint16)
    photonzip.decompress(photonzip.compress(sample, codec=codec, backend="dcu"), backend="dcu")


def compress_array(args: argparse.Namespace, array: np.ndarray):
    codec, codec_options, codec_params = make_codec_options(args, array)
    warm_up(args, codec)
    t0 = perf_counter()
    packed = photonzip.compress(array, codec=codec, backend=args.backend, codec_options=codec_options)
    t1 = perf_counter()
    return codec, packed, codec_params, t1 - t0


def run_compress(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    compress_input, preprocess = prepare_compress_input(args, array)
    codec, packed, codec_params, elapsed = compress_array(args, compress_input)
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
    print_compress_stats(array.nbytes, elapsed, packed.nbytes)


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
    warm_up(args, metadata["codec"])
    packed = _native.compressed_from_bytes(
        metadata["codec"],
        payload,
        metadata["dtype"],
        metadata["shape"],
        args.backend,
        metadata["codec_params"],
    )
    t0 = perf_counter()
    restored = photonzip.decompress(packed, backend=args.backend)
    t1 = perf_counter()
    array = to_host_numpy(restored)
    preprocess = metadata.get("preprocess") or {"type": "none"}
    if preprocess.get("type") == "delta":
        array = invert_delta(array, preprocess)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(args.output)
    print_decompress_stats(array.nbytes, t1 - t0)


def run_roundtrip(args: argparse.Namespace) -> None:
    array = read_raw_array(args.input, args.dtype, args.dims)
    compress_input, preprocess = prepare_compress_input(args, array)
    codec, packed, codec_params, compress_elapsed = compress_array(args, compress_input)
    t0 = perf_counter()
    restored = photonzip.decompress(packed, backend=args.backend)
    t1 = perf_counter()
    restored_array = to_host_numpy(restored)
    if preprocess.get("type") == "delta":
        restored_array = invert_delta(restored_array, preprocess)
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
    print_roundtrip_stats(array.nbytes, compress_elapsed, t1 - t0, packed.nbytes, np.array_equal(restored_array, array))


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
