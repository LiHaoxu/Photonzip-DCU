from __future__ import annotations

import argparse
import json
import struct
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
        default="cpu",
        choices=("cpu", "dcu"),
        help="Execution backend (default: cpu). LC payloads are identical on both backends.",
    )
    level_group = parser.add_argument_group("compression levels")
    level_group.add_argument(
        "--quality-level",
        dest="quality_level",
        default="lossless",
        choices=tuple(QUALITY_LEVEL_ERROR_BOUNDS),
        help="Quality level: lossless, high, or low.",
    )
    level_group.add_argument("--throughput-level", choices=("high", "low"), help="Throughput level.")
    level_group.add_argument("--ratio-level", choices=("high", "low"), help="Compression-ratio level.")
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode in ("compress", "roundtrip"):
        if not args.dims:
            raise ValueError("--dims is required for compress and roundtrip modes.")
        if args.dtype is None:
            raise ValueError("--dtype is required for compress and roundtrip modes.")
        if args.quality_level != "lossless":
            error_bound = QUALITY_LEVEL_ERROR_BOUNDS[args.quality_level]
            raise NotImplementedError(
                f"Lossy compression is not wired yet for quality_level={args.quality_level!r} "
                f"(preset error_bound={error_bound})."
            )
        if args.preprocess == "delta" and args.dtype != "uint16":
            raise ValueError("--delta currently supports --dtype uint16 only.")
    if args.codec == "mans" and args.backend != "dcu":
        raise ValueError("The MANS codec runs on the DCU only; pass --backend dcu.")
    if args.codec is not None and args.codec not in photonzip.list_codecs():
        raise ValueError(
            f"Codec {args.codec!r} is not available in this build (available: {photonzip.list_codecs()})."
        )


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


def compress_array(args: argparse.Namespace, array: np.ndarray):
    codec, codec_options, codec_params = make_codec_options(args, array)
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


def run_decompress(args: argparse.Namespace) -> None:
    metadata, payload = read_container(args.input)
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "compress":
        run_compress(args)
        return
    if args.mode == "decompress":
        run_decompress(args)
        return
    run_roundtrip(args)


if __name__ == "__main__":
    main()
