#!/usr/bin/env python3
"""Visualise photonzip-pfpl and photonzip-downsample reconstruction quality."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HDF5_PLUGIN_PATH", str(REPO_ROOT / "build" / "bin" / "plugins"))
sys.path.insert(0, str(REPO_ROOT / "build" / "bindings" / "python"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import h5py
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import photonzip.codec.pfpl as pfpl


@dataclass(frozen=True)
class Quality:
    rmse: float
    psnr_db: float
    max_abs_error: float
    max_relative_error: float


def parse_args() -> argparse.Namespace:
    data_root = REPO_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=data_root / "test_data" / "part" / "tomo_1_00001.tif",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "build" / "tomo_reconstruction_comparison.png",
    )
    parser.add_argument("--error-bound", type=float, default=3e-2)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crop-center-x", type=int, default=1024)
    parser.add_argument("--crop-center-y", type=int, default=1400)
    return parser.parse_args()


def load_tiff(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(f"expected one 2D TIFF image, got shape {source.shape}")
    if source.dtype != np.uint16:
        raise ValueError(f"expected uint16 TIFF input, got {source.dtype}")
    return np.ascontiguousarray(source)


def pfpl_roundtrip(source: np.ndarray, error_bound: float) -> tuple[np.ndarray, int]:
    source_f32 = np.ascontiguousarray(source, dtype=np.float32)[None, ...]
    with tempfile.TemporaryDirectory(prefix="photonzip-pfpl-") as temporary:
        h5_path = Path(temporary) / "reconstruction.h5"
        with h5py.File(h5_path, "w") as handle:
            dataset = handle.create_dataset(
                "data",
                data=source_f32,
                chunks=source_f32.shape,
                compression=pfpl.H5Z_FILTER_PFPL_ID,
                compression_opts=pfpl.to_hdf5_compression_opts(
                    pfpl.PfplOptions(error_bound=error_bound)
                ),
            )
            handle.flush()
            filter_ids = {
                dataset.id.get_create_plist().get_filter(index)[0]
                for index in range(dataset.id.get_create_plist().get_nfilters())
            }
            if pfpl.H5Z_FILTER_PFPL_ID not in filter_ids:
                raise RuntimeError("H5Z-PFPL was not applied to the HDF5 dataset")
            storage_bytes = int(dataset.id.get_storage_size())

        with h5py.File(h5_path, "r") as handle:
            restored = np.ascontiguousarray(handle["data"][0])
    return restored, storage_bytes


def downsample_roundtrip(source: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor < 1:
        raise ValueError("downsampling factor must be positive")
    reduced = np.ascontiguousarray(source[::factor, ::factor])
    restored = np.repeat(np.repeat(reduced, factor, axis=0), factor, axis=1)
    return reduced, np.ascontiguousarray(restored[: source.shape[0], : source.shape[1]])


def quality(source: np.ndarray, restored: np.ndarray) -> Quality:
    source_f64 = source.astype(np.float64)
    difference = restored.astype(np.float64) - source_f64
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    data_range = float(source.max()) - float(source.min())
    psnr_db = math.inf if rmse == 0.0 else 20.0 * math.log10(data_range / rmse)
    nonzero = np.abs(source_f64) > 0.0
    relative = absolute[nonzero] / np.abs(source_f64[nonzero])
    return Quality(
        rmse=rmse,
        psnr_db=psnr_db,
        max_abs_error=float(absolute.max()),
        max_relative_error=float(relative.max(initial=0.0)),
    )


def crop_bounds(
    shape: tuple[int, int],
    center_x: int,
    center_y: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    size = min(crop_size, height, width)
    x0 = min(max(0, center_x - size // 2), width - size)
    y0 = min(max(0, center_y - size // 2), height - size)
    return x0, y0, x0 + size, y0 + size


def format_error_bound(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_figure(
    source_path: Path,
    output_path: Path,
    source: np.ndarray,
    reconstructions: list[np.ndarray],
    labels: list[str],
    qualities: list[Quality],
    ratios: list[float],
    error_bound: float,
    downsample_factor: int,
    crop: tuple[int, int, int, int],
) -> None:
    display_min, display_max = np.percentile(source, [0.5, 99.5])
    x0, y0, x1, y1 = crop
    images = [source, *reconstructions]
    errors = [
        np.abs(restored.astype(np.float64) - source.astype(np.float64))
        for restored in reconstructions
    ]
    error_max = max(float(np.percentile(error, 99.9)) for error in errors)
    positive_minima = [float(error[error > 0].min()) for error in errors if np.any(error > 0)]
    error_min = max(min(positive_minima, default=1.0), 1e-3)
    error_norm = colors.LogNorm(vmin=error_min, vmax=max(error_max, error_min * 10.0))
    error_cmap = plt.get_cmap("inferno").copy()
    error_cmap.set_bad("black")

    figure = plt.figure(figsize=(15, 15), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.82))

    column_titles = [
        f"Original\n{source.shape[1]} x {source.shape[0]} uint16",
        f"photonzip-pfpl, eb={format_error_bound(error_bound)}\n"
        f"U16 payload ratio {ratios[0]:.2f}x",
        f"photonzip-downsample, {downsample_factor}x linear + NN restore\n"
        f"pixel payload ratio {ratios[1]:.0f}x",
    ]

    for column, (image, title) in enumerate(zip(images, column_titles)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(image, cmap="gray", vmin=display_min, vmax=display_max)
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="tab:red",
                linewidth=1.5,
            )
        )
        axis.set_title(title, fontsize=12)
        axis.set_axis_off()

        crop_axis = figure.add_subplot(grid[1, column])
        crop_axis.imshow(
            image[y0:y1, x0:x1],
            cmap="gray",
            vmin=display_min,
            vmax=display_max,
            interpolation="nearest",
        )
        if column == 0:
            crop_axis.set_title(f"Detail crop ({x1 - x0} x {y1 - y0})", fontsize=11)
        else:
            metric = qualities[column - 1]
            crop_axis.set_title(
                f"RMSE {metric.rmse:.1f} | PSNR {metric.psnr_db:.2f} dB\n"
                f"max abs error {metric.max_abs_error:.1f}",
                fontsize=11,
            )
        crop_axis.set_axis_off()

    histogram_axis = figure.add_subplot(grid[2, 0])
    histogram_axis.hist(source.ravel(), bins=180, color="0.35", log=True)
    histogram_axis.axvline(display_min, color="tab:red", linewidth=1.2)
    histogram_axis.axvline(display_max, color="tab:red", linewidth=1.2)
    histogram_axis.set_title("Original intensity distribution")
    histogram_axis.set_xlabel("uint16 intensity")
    histogram_axis.set_ylabel("Pixel count (log scale)")
    histogram_axis.grid(alpha=0.2)

    error_axes = []
    for column, (error, label) in enumerate(zip(errors, labels), start=1):
        axis = figure.add_subplot(grid[2, column])
        masked_error = np.ma.masked_less_equal(error, 0.0)
        image_artist = axis.imshow(
            masked_error,
            cmap=error_cmap,
            norm=error_norm,
            interpolation="nearest",
        )
        axis.set_title(f"{label}: absolute error (log scale)", fontsize=11)
        axis.set_axis_off()
        error_axes.append(axis)

    figure.colorbar(
        image_artist,
        ax=error_axes,
        location="bottom",
        shrink=0.82,
        pad=0.03,
        label="Absolute intensity error",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not (args.error_bound > 0.0 and math.isfinite(args.error_bound)):
        raise ValueError("error bound must be positive and finite")

    source = load_tiff(args.input)
    lossy, lossy_bytes = pfpl_roundtrip(source, args.error_bound)
    reduced, downsampled_restored = downsample_roundtrip(source, args.downsample_factor)

    reconstructions = [lossy, downsampled_restored]
    qualities = [quality(source, restored) for restored in reconstructions]
    ratios = [
        source.nbytes / lossy_bytes,
        source.nbytes / reduced.nbytes,
    ]
    crop = crop_bounds(
        source.shape,
        args.crop_center_x,
        args.crop_center_y,
        args.crop_size,
    )
    make_figure(
        args.input,
        args.output,
        source,
        reconstructions,
        [
            f"photonzip-pfpl {format_error_bound(args.error_bound)}",
            f"photonzip-downsample {args.downsample_factor}x",
        ],
        qualities,
        ratios,
        args.error_bound,
        args.downsample_factor,
        crop,
    )

    print(f"output={args.output}")
    for label, metric, ratio in zip(
        [
            f"photonzip_pfpl_{format_error_bound(args.error_bound)}",
            f"photonzip_downsample_{args.downsample_factor}x",
        ],
        qualities,
        ratios,
    ):
        print(
            f"{label}: ratio_u16={ratio:.6f}x, rmse={metric.rmse:.6f}, "
            f"psnr_db={metric.psnr_db:.6f}, max_abs={metric.max_abs_error:.6f}, "
            f"max_rel={metric.max_relative_error:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
