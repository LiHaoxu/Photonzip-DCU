#!/usr/bin/env python3
"""Benchmark five TIFFs and simulate 100-frame transfer-to-visualisation latency."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "build" / "bindings" / "python"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import photonzip
import photonzip.codec.pfpl as pfpl


METHOD_RAW = "raw"
METHOD_PFPL = "photonzip-pfpl"
METHOD_DOWNSAMPLE = "photonzip-downsample"
METHODS = (METHOD_RAW, METHOD_PFPL, METHOD_DOWNSAMPLE)
COMPONENTS = ("sender_prep_sec", "receiver_reconstruct_sec", "visualise_sec")


class Renderer:
    """Persistent 1024x1024 off-screen renderer approximating a frame update."""

    def __init__(self, shape: tuple[int, int]) -> None:
        figure = Figure(figsize=(8, 8), dpi=128, frameon=False)
        self.canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axis.set_axis_off()
        self.artist = axis.imshow(
            np.zeros(shape, dtype=np.float32),
            cmap="gray",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

    def draw(self, image: np.ndarray) -> None:
        display_min, display_max = np.percentile(image, (0.5, 99.5))
        self.artist.set_data(image)
        self.artist.set_clim(float(display_min), float(display_max))
        self.canvas.draw()
        self.canvas.buffer_rgba()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiff-dir",
        type=Path,
        default=REPO_ROOT.parent / "test_data" / "part",
    )
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument("--file-count", type=int, default=5)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--error-bound", type=float, default=3e-2)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        nargs="+",
        default=(1.0, 10.0, 25.0, 100.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "build" / "tomo_pipeline_simulation.json",
    )
    return parser.parse_args()


def load_images(paths: list[Path]) -> tuple[list[np.ndarray], float]:
    started = perf_counter()
    images = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(f"{path} must be a 2D uint16 TIFF, got {array.shape} {array.dtype}")
        images.append(np.ascontiguousarray(array))
    return images, perf_counter() - started


def timed(callable_):
    started = perf_counter()
    value = callable_()
    return value, perf_counter() - started


def raw_sender(source: np.ndarray) -> bytes:
    return source.tobytes(order="C")


def raw_receiver(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.uint16).reshape(shape).copy()


def pfpl_sender(
    source: np.ndarray,
    options: pfpl.PfplOptions,
) -> bytes:
    source_f32 = np.ascontiguousarray(source, dtype=np.float32)
    packed = photonzip.compress(
        source_f32,
        codec="pfpl",
        backend="cpu",
        codec_options=options,
    )
    return packed.to_bytes()


def pfpl_receiver(
    payload: bytes,
    shape: tuple[int, int],
    codec_params: list[int],
) -> np.ndarray:
    packed = pfpl._native.compressed_from_bytes(
        "pfpl",
        payload,
        "float32",
        list(shape),
        "cpu",
        codec_params,
    )
    return np.ascontiguousarray(
        np.from_dlpack(photonzip.decompress(packed, backend="cpu"))
    )


def downsample_sender(source: np.ndarray, factor: int) -> tuple[bytes, tuple[int, int]]:
    reduced = np.ascontiguousarray(source[::factor, ::factor])
    return reduced.tobytes(order="C"), reduced.shape


def downsample_receiver(
    payload: bytes,
    reduced_shape: tuple[int, int],
    source_shape: tuple[int, int],
    factor: int,
) -> np.ndarray:
    reduced = np.frombuffer(payload, dtype=np.uint16).reshape(reduced_shape)
    restored = np.repeat(np.repeat(reduced, factor, axis=0), factor, axis=1)
    return np.ascontiguousarray(restored[: source_shape[0], : source_shape[1]])


def empty_trial() -> dict[str, dict[str, float]]:
    return {
        method: {component: 0.0 for component in COMPONENTS}
        for method in METHODS
    }


def run_trial(
    images: list[np.ndarray],
    renderer: Renderer,
    error_bound: float,
    downsample_factor: int,
    verify: bool,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, int],
    list[dict[str, dict[str, float | int]]],
]:
    timings = empty_trial()
    payload_bytes = {method: 0 for method in METHODS}
    per_file = []
    options = pfpl.PfplOptions(error_bound=error_bound)
    codec_params = options.to_codec_params(backend="cpu")

    gc.collect()
    gc.disable()
    try:
        for source in images:
            file_timings = empty_trial()
            file_payload_bytes = {method: 0 for method in METHODS}

            raw_payload, elapsed = timed(lambda: raw_sender(source))
            timings[METHOD_RAW]["sender_prep_sec"] += elapsed
            file_timings[METHOD_RAW]["sender_prep_sec"] = elapsed
            payload_bytes[METHOD_RAW] += len(raw_payload)
            file_payload_bytes[METHOD_RAW] = len(raw_payload)
            raw_restored, elapsed = timed(lambda: raw_receiver(raw_payload, source.shape))
            timings[METHOD_RAW]["receiver_reconstruct_sec"] += elapsed
            file_timings[METHOD_RAW]["receiver_reconstruct_sec"] = elapsed
            _, elapsed = timed(lambda: renderer.draw(raw_restored))
            timings[METHOD_RAW]["visualise_sec"] += elapsed
            file_timings[METHOD_RAW]["visualise_sec"] = elapsed

            pfpl_payload, elapsed = timed(lambda: pfpl_sender(source, options))
            timings[METHOD_PFPL]["sender_prep_sec"] += elapsed
            file_timings[METHOD_PFPL]["sender_prep_sec"] = elapsed
            payload_bytes[METHOD_PFPL] += len(pfpl_payload)
            file_payload_bytes[METHOD_PFPL] = len(pfpl_payload)
            pfpl_restored, elapsed = timed(
                lambda: pfpl_receiver(pfpl_payload, source.shape, codec_params)
            )
            timings[METHOD_PFPL]["receiver_reconstruct_sec"] += elapsed
            file_timings[METHOD_PFPL]["receiver_reconstruct_sec"] = elapsed
            _, elapsed = timed(lambda: renderer.draw(pfpl_restored))
            timings[METHOD_PFPL]["visualise_sec"] += elapsed
            file_timings[METHOD_PFPL]["visualise_sec"] = elapsed

            downsample_result, elapsed = timed(
                lambda: downsample_sender(source, downsample_factor)
            )
            timings[METHOD_DOWNSAMPLE]["sender_prep_sec"] += elapsed
            file_timings[METHOD_DOWNSAMPLE]["sender_prep_sec"] = elapsed
            downsample_payload, reduced_shape = downsample_result
            payload_bytes[METHOD_DOWNSAMPLE] += len(downsample_payload)
            file_payload_bytes[METHOD_DOWNSAMPLE] = len(downsample_payload)
            downsample_restored, elapsed = timed(
                lambda: downsample_receiver(
                    downsample_payload,
                    reduced_shape,
                    source.shape,
                    downsample_factor,
                )
            )
            timings[METHOD_DOWNSAMPLE]["receiver_reconstruct_sec"] += elapsed
            file_timings[METHOD_DOWNSAMPLE]["receiver_reconstruct_sec"] = elapsed
            _, elapsed = timed(lambda: renderer.draw(downsample_restored))
            timings[METHOD_DOWNSAMPLE]["visualise_sec"] += elapsed
            file_timings[METHOD_DOWNSAMPLE]["visualise_sec"] = elapsed

            if verify:
                if not np.array_equal(raw_restored, source):
                    raise RuntimeError("raw round trip changed the source")
                realised_bound = error_bound + 2.0 * float(np.finfo(np.float32).eps)
                source_f32 = source.astype(np.float32)
                if not np.all(
                    np.abs(pfpl_restored - source_f32)
                    <= realised_bound * np.abs(source_f32)
                ):
                    raise RuntimeError("photonzip-pfpl exceeded its relative error bound")
                if downsample_restored.shape != source.shape:
                    raise RuntimeError("photonzip-downsample restored the wrong shape")

            per_file.append(
                {
                    method: {
                        **file_timings[method],
                        "payload_bytes": file_payload_bytes[method],
                    }
                    for method in METHODS
                }
            )
    finally:
        gc.enable()
    return timings, payload_bytes, per_file


def median_timings(
    trials: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    return {
        method: {
            component: statistics.median(
                trial[method][component] for trial in trials
            )
            for component in COMPONENTS
        }
        for method in METHODS
    }


def median_per_file(
    trials: list[list[dict[str, dict[str, float | int]]]],
    bandwidths: list[float],
) -> list[dict[str, dict[str, object]]]:
    file_count = len(trials[0])
    results = []
    for file_index in range(file_count):
        methods = {}
        for method in METHODS:
            values = {
                component: statistics.median(
                    float(trial[file_index][method][component])
                    for trial in trials
                )
                for component in COMPONENTS
            }
            payload_bytes = int(trials[0][file_index][method]["payload_bytes"])
            fixed_sec = sum(values.values())
            methods[method] = {
                **values,
                "payload_bytes": payload_bytes,
                "payload_mib": payload_bytes / 2**20,
                "fixed_cpu_vis_sec": fixed_sec,
                "total_sec_by_bandwidth": {
                    f"{bandwidth:g}": fixed_sec
                    + payload_bytes * 8.0 / (bandwidth * 1e9)
                    for bandwidth in bandwidths
                },
            }
        results.append(methods)
    return results


def simulate(
    measured: dict[str, dict[str, float]],
    payload_bytes: dict[str, int],
    scale: float,
    bandwidths: list[float],
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]], dict[str, float | None]]:
    scaled = {
        method: {
            component: measured[method][component] * scale
            for component in COMPONENTS
        }
        for method in METHODS
    }
    scenarios = []
    for bandwidth_gbps in bandwidths:
        rows = []
        for method in METHODS:
            transfer_sec = (
                payload_bytes[method] * scale * 8.0 / (bandwidth_gbps * 1e9)
            )
            fixed_sec = sum(scaled[method].values())
            rows.append(
                {
                    "method": method,
                    **scaled[method],
                    "transfer_sec": transfer_sec,
                    "total_sec": fixed_sec + transfer_sec,
                    "payload_mib": payload_bytes[method] * scale / 2**20,
                }
            )
        raw_total = rows[0]["total_sec"]
        for row in rows:
            row["speedup_vs_raw"] = raw_total / row["total_sec"]
        scenarios.append({"bandwidth_gbps": bandwidth_gbps, "methods": rows})

    raw_fixed = sum(scaled[METHOD_RAW].values())
    raw_bytes = payload_bytes[METHOD_RAW] * scale
    break_even = {}
    for method in (METHOD_PFPL, METHOD_DOWNSAMPLE):
        extra_fixed = sum(scaled[method].values()) - raw_fixed
        saved_bits = (raw_bytes - payload_bytes[method] * scale) * 8.0
        break_even[method] = (
            saved_bits / extra_fixed / 1e9
            if extra_fixed > 0.0 and saved_bits > 0.0
            else None
        )
    return scaled, scenarios, break_even


def main() -> int:
    args = parse_args()
    if args.file_count != 5 or not math.isclose(args.scale, 20.0):
        print(
            "note: the requested tomo estimate is five TIFFs multiplied by 20; "
            f"current values are {args.file_count} and {args.scale:g}",
            file=sys.stderr,
        )
    if args.trials < 1 or args.downsample_factor < 1:
        raise ValueError("trials and downsample factor must be positive")
    if not (args.error_bound > 0.0 and math.isfinite(args.error_bound)):
        raise ValueError("error bound must be positive and finite")
    if any(value <= 0.0 for value in args.bandwidth_gbps):
        raise ValueError("transfer bandwidths must be positive")

    paths = sorted(args.tiff_dir.glob(args.glob))[: args.file_count]
    if len(paths) != args.file_count:
        raise FileNotFoundError(
            f"needed {args.file_count} TIFFs, found {len(paths)} under {args.tiff_dir}"
        )
    images, load_sec = load_images(paths)
    renderer = Renderer(images[0].shape)

    # Warm up codec, allocator, font cache, and renderer outside measurements.
    run_trial(
        images[:1],
        renderer,
        args.error_bound,
        args.downsample_factor,
        verify=True,
    )

    trials = []
    per_file_trials = []
    payload_bytes = {}
    for trial_index in range(args.trials):
        timings, current_payload_bytes, current_per_file = run_trial(
            images,
            renderer,
            args.error_bound,
            args.downsample_factor,
            verify=trial_index == 0,
        )
        trials.append(timings)
        per_file_trials.append(current_per_file)
        if payload_bytes and payload_bytes != current_payload_bytes:
            raise RuntimeError("payload sizes changed between trials")
        payload_bytes = current_payload_bytes

    measured = median_timings(trials)
    per_file = median_per_file(per_file_trials, list(args.bandwidth_gbps))
    scaled, scenarios, break_even = simulate(
        measured,
        payload_bytes,
        args.scale,
        list(args.bandwidth_gbps),
    )
    result = {
        "config": {
            "files": [str(path) for path in paths],
            "measured_file_count": len(paths),
            "scale_to_tomo": args.scale,
            "simulated_tomo_frames": int(len(paths) * args.scale),
            "trials": args.trials,
            "trial_aggregation": "median of component totals",
            "pfpl_relative_error_bound": args.error_bound,
            "downsample_linear_factor": args.downsample_factor,
            "visualisation": "persistent 1024x1024 Matplotlib Agg frame render",
            "transfer_model": "ideal payload_bits / link_bits_per_second",
            "source_tiff_load_excluded_from_pipeline": True,
            "source_tiff_load_sec_for_five_files": load_sec,
        },
        "measured_five_tiffs": {
            method: {
                **measured[method],
                "payload_bytes": payload_bytes[method],
                "payload_mib": payload_bytes[method] / 2**20,
            }
            for method in METHODS
        },
        "measured_per_file": [
            {
                "file": str(path),
                "methods": per_file[file_index],
            }
            for file_index, path in enumerate(paths)
        ],
        "scaled_cpu_times_100_tiffs": scaled,
        "simulated_100_tiffs": scenarios,
        "break_even_bandwidth_gbps": break_even,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"output={args.output}")
    print(f"files={len(paths)}, scale={args.scale:g}, trials={args.trials}")
    for method in METHODS:
        values = measured[method]
        print(
            f"{method}: five_tiffs sender={values['sender_prep_sec']:.6f}s, "
            f"receiver={values['receiver_reconstruct_sec']:.6f}s, "
            f"vis={values['visualise_sec']:.6f}s, "
            f"payload={payload_bytes[method] / 2**20:.6f}MiB"
        )
    for file_index, path in enumerate(paths):
        print(f"file={path.name}")
        for method in METHODS:
            values = per_file[file_index][method]
            print(
                f"  {method}: fixed={values['fixed_cpu_vis_sec']:.6f}s, "
                f"total_1Gbps={values['total_sec_by_bandwidth']['1']:.6f}s, "
                f"total_10Gbps={values['total_sec_by_bandwidth']['10']:.6f}s, "
                f"payload={values['payload_mib']:.6f}MiB"
            )
    for scenario in scenarios:
        print(f"bandwidth={scenario['bandwidth_gbps']:g}Gbps")
        for row in scenario["methods"]:
            print(
                f"  {row['method']}: total={row['total_sec']:.6f}s, "
                f"transfer={row['transfer_sec']:.6f}s, "
                f"speedup={row['speedup_vs_raw']:.6f}x"
            )
    for method, bandwidth in break_even.items():
        label = "always" if bandwidth is None else f"{bandwidth:.6f}Gbps"
        print(f"break_even {method}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
