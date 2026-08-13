#!/usr/bin/env python3
"""Stable fixed-size crop for Q42 hess-matched stitched frames.

The standard flip/crop script removes black borders independently per frame.
That is safe, but for Q42 the crop size changes enough to create visible
breathing/shaking after every frame is resized to 1920x1080. This script keeps
the same 16:9 crop size for every frame and only lets the crop center move
smoothly when the valid region requires it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_frames(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def read_crop_report(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[row["frame"]] = {
                key: float(value)
                for key, value in row.items()
                if key != "frame"
            }
        return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(values, dtype=np.float64)
    for dim in range(values.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def clamp_centers(centers: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(centers, lo), hi)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def invalid_count(image: np.ndarray, threshold: int) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return int((gray <= threshold).sum())


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(r"D:\visualization2026\dowsample2_Q42_hess_matched")
    parser = argparse.ArgumentParser(description="Create stable fixed-size Q42 flipped/cropped frames.")
    parser.add_argument("--input-dir", type=Path, default=root / "stitched_frames")
    parser.add_argument("--valid-crop-report", type=Path, default=root / "flip_crop_report.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "stitched_frames_flipped_cropped_stable")
    parser.add_argument("--report-csv", type=Path, default=root / "stable_crop_report.csv")
    parser.add_argument("--metadata-json", type=Path, default=root / "stable_crop_metadata.json")
    parser.add_argument("--crop-width", type=int, default=1408)
    parser.add_argument("--crop-height", type=int, default=792)
    parser.add_argument("--smooth-window", type=int, default=61)
    parser.add_argument("--smooth-iterations", type=int, default=4)
    parser.add_argument("--black-threshold", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.crop_width <= 0 or args.crop_height <= 0:
        raise ValueError("Crop size must be positive.")
    if abs(args.crop_width / args.crop_height - 16 / 9) > 0.01:
        raise ValueError("Crop size should be close to 16:9.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")

    frames = list_frames(args.input_dir)
    valid_rows = read_crop_report(args.valid_crop_report)
    if not frames:
        raise RuntimeError(f"No frames found in {args.input_dir}")

    crop_w = float(args.crop_width)
    crop_h = float(args.crop_height)
    raw_centers = []
    lo = []
    hi = []
    for path in frames:
        if path.name not in valid_rows:
            raise KeyError(f"Missing crop report row for {path.name}")
        row = valid_rows[path.name]
        if row["crop_width"] < crop_w or row["crop_height"] < crop_h:
            raise ValueError(
                f"{path.name} valid crop {row['crop_width']}x{row['crop_height']} "
                f"is smaller than requested {crop_w}x{crop_h}"
            )
        raw_centers.append(
            [
                0.5 * (row["crop_left"] + row["crop_right"]),
                0.5 * (row["crop_top"] + row["crop_bottom"]),
            ]
        )
        lo.append([row["crop_left"] + 0.5 * crop_w, row["crop_top"] + 0.5 * crop_h])
        hi.append([row["crop_right"] - 0.5 * crop_w, row["crop_bottom"] - 0.5 * crop_h])

    raw_centers_arr = np.array(raw_centers, dtype=np.float64)
    lo_arr = np.array(lo, dtype=np.float64)
    hi_arr = np.array(hi, dtype=np.float64)
    centers = clamp_centers(raw_centers_arr, lo_arr, hi_arr)
    for _ in range(max(1, args.smooth_iterations)):
        centers = moving_average(centers, args.smooth_window)
        centers = clamp_centers(centers, lo_arr, hi_arr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(frames, start=1):
        image = read_image(path)
        flipped = cv2.flip(image, 0)
        cx, cy = centers[index - 1]
        left = int(round(cx - 0.5 * crop_w))
        top = int(round(cy - 0.5 * crop_h))
        right = left + args.crop_width
        bottom = top + args.crop_height
        # Final integer clamp protects against rounding at tight valid boundaries.
        valid = valid_rows[path.name]
        left = max(int(np.ceil(valid["crop_left"])), min(left, int(np.floor(valid["crop_right"] - crop_w))))
        top = max(int(np.ceil(valid["crop_top"])), min(top, int(np.floor(valid["crop_bottom"] - crop_h))))
        right = left + args.crop_width
        bottom = top + args.crop_height
        cropped = flipped[top:bottom, left:right]
        if cropped.shape[0] != args.crop_height or cropped.shape[1] != args.crop_width:
            raise RuntimeError(f"Bad crop shape for {path.name}: {cropped.shape}")
        out_path = args.output_dir / path.name
        write_image(out_path, cropped, args.jpeg_quality)

        row = {
            "frame": path.name,
            "source_width": image.shape[1],
            "source_height": image.shape[0],
            "valid_crop_left": valid["crop_left"],
            "valid_crop_top": valid["crop_top"],
            "valid_crop_right": valid["crop_right"],
            "valid_crop_bottom": valid["crop_bottom"],
            "raw_center_x": raw_centers_arr[index - 1, 0],
            "raw_center_y": raw_centers_arr[index - 1, 1],
            "stable_center_x": left + 0.5 * args.crop_width,
            "stable_center_y": top + 0.5 * args.crop_height,
            "crop_left": left,
            "crop_top": top,
            "crop_right": right,
            "crop_bottom": bottom,
            "crop_width": args.crop_width,
            "crop_height": args.crop_height,
            "invalid_dark_pixels_inside_crop": invalid_count(cropped, args.black_threshold),
        }
        rows.append(row)
        if index == 1 or index % args.progress_every == 0 or index == len(frames):
            print(
                f"[{index}/{len(frames)}] {path.name}: "
                f"stable_crop={args.crop_width}x{args.crop_height} "
                f"center=({row['stable_center_x']:.1f},{row['stable_center_y']:.1f}) "
                f"dark={row['invalid_dark_pixels_inside_crop']}"
            )

    write_csv(args.report_csv, rows)
    centers_written = np.array([[row["stable_center_x"], row["stable_center_y"]] for row in rows], dtype=np.float64)
    center_step = np.linalg.norm(np.diff(centers_written, axis=0), axis=1)
    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "valid_crop_report": str(args.valid_crop_report),
        "frame_count": len(rows),
        "crop_width": args.crop_width,
        "crop_height": args.crop_height,
        "smooth_window": args.smooth_window,
        "smooth_iterations": args.smooth_iterations,
        "center_step_median_px": float(np.median(center_step)) if len(center_step) else 0.0,
        "center_step_p90_px": float(np.percentile(center_step, 90)) if len(center_step) else 0.0,
        "center_step_max_px": float(np.max(center_step)) if len(center_step) else 0.0,
        "max_dark_pixels_inside_crop": int(max(row["invalid_dark_pixels_inside_crop"] for row in rows)),
        "all_output_frames_same_size": True,
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} frames to {args.output_dir}")
    print(f"Report: {args.report_csv}")
    print(f"Metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
