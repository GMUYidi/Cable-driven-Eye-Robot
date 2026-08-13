#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def list_frames(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def border_connected_invalid_mask(
    image: np.ndarray,
    black_threshold: int,
    min_component_area: int,
    erode_pixels: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray <= black_threshold).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )

    invalid = np.zeros(dark.shape, dtype=np.uint8)
    for label in border_labels:
        if label == 0 or label_count <= label:
            continue
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            invalid[labels == label] = 1

    valid = 1 - invalid
    if erode_pixels > 0:
        kernel_size = 2 * erode_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=1)
        invalid = 1 - valid
    return invalid.astype(bool)


def blurred_cover_background(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (int(math.ceil(source_w * scale)), int(math.ceil(source_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    cropped = resized[y0 : y0 + height, x0 : x0 + width]
    if cropped.shape[0] != height or cropped.shape[1] != width:
        cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    small = cv2.resize(cropped, (320, 180), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 12)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flip stitched frames and place them on a fixed canvas without dynamic crop."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scale-mode", choices=["fit-all"], default="fit-all")
    parser.add_argument("--align", choices=["center", "top-left"], default="center")
    parser.add_argument("--black-threshold", type=int, default=12)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--erode-pixels", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames = list_frames(args.input_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in args.output_dir.iterdir():
            if path.is_file():
                path.unlink()

    sizes: list[tuple[int, int]] = []
    for frame in frames:
        image = read_image(frame)
        h, w = image.shape[:2]
        sizes.append((w, h))

    max_width = max(width for width, _ in sizes)
    max_height = max(height for _, height in sizes)
    scale = min(args.width / max_width, args.height / max_height)

    rows: list[dict[str, int | float | str]] = []
    for index, frame in enumerate(frames, start=1):
        image = cv2.flip(read_image(frame), 0)
        invalid = border_connected_invalid_mask(
            image,
            black_threshold=args.black_threshold,
            min_component_area=args.min_component_area,
            erode_pixels=args.erode_pixels,
        )
        valid = (~invalid).astype(np.uint8)
        source_h, source_w = image.shape[:2]
        scaled_w = max(1, int(round(source_w * scale)))
        scaled_h = max(1, int(round(source_h * scale)))

        resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        valid_resized = cv2.resize(valid, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) > 0
        canvas = blurred_cover_background(image, args.width, args.height)

        if args.align == "center":
            x0 = (args.width - scaled_w) // 2
            y0 = (args.height - scaled_h) // 2
        else:
            x0 = 0
            y0 = 0

        x1 = x0 + scaled_w
        y1 = y0 + scaled_h
        dest_x0 = max(0, x0)
        dest_y0 = max(0, y0)
        dest_x1 = min(args.width, x1)
        dest_y1 = min(args.height, y1)
        src_x0 = dest_x0 - x0
        src_y0 = dest_y0 - y0
        src_x1 = src_x0 + (dest_x1 - dest_x0)
        src_y1 = src_y0 + (dest_y1 - dest_y0)

        roi = canvas[dest_y0:dest_y1, dest_x0:dest_x1]
        src = resized[src_y0:src_y1, src_x0:src_x1]
        mask = valid_resized[src_y0:src_y1, src_x0:src_x1]
        roi[mask] = src[mask]

        output_path = args.output_dir / frame.name
        write_image(output_path, canvas, args.jpeg_quality)
        rows.append(
            {
                "frame": frame.name,
                "source_width": source_w,
                "source_height": source_h,
                "scaled_width": scaled_w,
                "scaled_height": scaled_h,
                "canvas_width": args.width,
                "canvas_height": args.height,
                "scale": scale,
                "x0": x0,
                "y0": y0,
                "valid_pixels_scaled": int(valid_resized.sum()),
            }
        )

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(frames)
        ):
            print(f"[{index}/{len(frames)}] {frame.name}: scaled={scaled_w}x{scaled_h}")

    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "frame_count": len(rows),
        "canvas_width": args.width,
        "canvas_height": args.height,
        "scale_mode": args.scale_mode,
        "align": args.align,
        "constant_scale": scale,
        "source_width_max": max_width,
        "source_height_max": max_height,
        "source_width_median": float(np.median([width for width, _ in sizes])),
        "source_height_median": float(np.median([height for _, height in sizes])),
        "black_threshold": args.black_threshold,
        "min_component_area": args.min_component_area,
        "erode_pixels": args.erode_pixels,
        "invalid_region_handling": "border-connected invalid pixels are replaced by blurred full-canvas background",
        "no_dynamic_crop": True,
        "no_extra_zoom_in": True,
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
