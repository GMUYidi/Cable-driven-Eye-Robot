#!/usr/bin/env python3
"""Fill Q42 tail chessboard edge with checkerboard-aware local synthesis.

The previous Q42 tail fix removed the right-eye-only chessboard sliver to avoid
double images. This script makes a less "cut off" version by extending visible
dark checker squares by one same-row checkerboard period in the final tail
segment. It works on the already flipped/cropped frames, so the gaze overlay
and homography matrices are not changed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def list_frames(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse frame number from {path.name}")
    return int(match.group(1))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def detect_dark_checker_components(image: np.ndarray) -> tuple[list[dict[str, float | int]], np.ndarray, float]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[: int(height * 0.70), int(width * 0.35) :] = 255
    roi_values = blurred[roi_mask > 0]
    threshold = max(70.0, min(135.0, float(np.percentile(roi_values, 22))))

    dark = ((blurred < threshold) & (roi_mask > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    components: list[dict[str, float | int]] = []
    for label in range(1, label_count):
        x, y, comp_w, comp_h, area = stats[label]
        aspect = comp_w / max(comp_h, 1)
        if not (250 <= area <= 20000):
            continue
        if not (12 <= comp_w <= 220 and 12 <= comp_h <= 220):
            continue
        if not (0.35 <= aspect <= 2.8):
            continue
        components.append(
            {
                "label": int(label),
                "x": int(x),
                "y": int(y),
                "w": int(comp_w),
                "h": int(comp_h),
                "area": int(area),
                "cx": float(centroids[label][0]),
                "cy": float(centroids[label][1]),
            }
        )
    return components, labels, threshold


def estimate_same_row_period(components: list[dict[str, float | int]]) -> float | None:
    deltas: list[float] = []
    for a in components:
        for b in components:
            if float(b["cx"]) <= float(a["cx"]):
                continue
            y_tolerance = max(int(a["h"]), int(b["h"])) * 0.65
            if abs(float(b["cy"]) - float(a["cy"])) > y_tolerance:
                continue
            delta = float(b["cx"]) - float(a["cx"])
            if 30 < delta < 260:
                deltas.append(delta)
    if deltas:
        return float(np.median(deltas))
    widths = [int(c["w"]) for c in components]
    if widths:
        return float(np.median(widths) * 2.1)
    return None


def synthesize_one_checker_period(
    image: np.ndarray,
    tail_alpha: float,
    copy_strength: float,
) -> tuple[np.ndarray, int]:
    if tail_alpha <= 1e-6:
        return image.copy(), 0

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    components, labels, threshold = detect_dark_checker_components(image)
    period = estimate_same_row_period(components)
    if period is None:
        return image.copy(), 0

    result = image.copy()
    copied = 0
    for component in components:
        if float(component["cx"]) < width * 0.45:
            continue
        src_x = int(component["x"])
        src_y = int(component["y"])
        comp_w = int(component["w"])
        comp_h = int(component["h"])
        dst_x = int(round(src_x + period))
        dst_y = src_y

        if dst_x + comp_w >= width or dst_x < int(width * 0.62):
            continue
        target = gray[
            max(0, dst_y) : min(height, dst_y + comp_h),
            max(0, dst_x) : min(width, dst_x + comp_w),
        ]
        if target.size == 0:
            continue
        if np.percentile(target, 20) < threshold + 8:
            continue

        patch = image[src_y : src_y + comp_h, src_x : src_x + comp_w]
        mask = (labels[src_y : src_y + comp_h, src_x : src_x + comp_w] == int(component["label"])).astype(
            np.uint8
        ) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask_f = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0
        mask_f *= copy_strength * tail_alpha

        y1 = min(height, dst_y + comp_h)
        x1 = min(width, dst_x + comp_w)
        patch_h = y1 - dst_y
        patch_w = x1 - dst_x
        if patch_h <= 0 or patch_w <= 0:
            continue

        roi = result[dst_y:y1, dst_x:x1].astype(np.float32)
        donor = cv2.GaussianBlur(patch[:patch_h, :patch_w].astype(np.float32), (0, 0), 1.2)
        alpha = mask_f[:patch_h, :patch_w]
        roi = roi * (1.0 - alpha[:, :, None]) + donor * alpha[:, :, None]
        result[dst_y:y1, dst_x:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        copied += 1

    return result, copied


def encode_video(frames_dir: Path, output_video: Path, width: int, height: int, fps: float) -> None:
    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")
    for frame in frames:
        image = read_image(frame)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(image)
    writer.release()


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(r"D:\visualization2026\dowsample2_Q42_hess_matched")
    parser = argparse.ArgumentParser(description="Create a Q42 tail version with checkerboard-aware edge fill.")
    parser.add_argument("--input-dir", type=Path, default=root / "stitched_frames_flipped_cropped")
    parser.add_argument("--output-dir", type=Path, default=root / "stitched_frames_flipped_cropped_ai_chess_filled")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=root / "stitched_frames_flipped_cropped_60fps_ai_chess_filled.mp4",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=root / "ai_chess_fill_metadata.json",
    )
    parser.add_argument("--fade-start-video-frame", type=int, default=630)
    parser.add_argument("--full-video-frame", type=int, default=690)
    parser.add_argument("--copy-strength", type=float, default=0.80)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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

    rows: list[dict[str, int | float | str]] = []
    for index, frame_path in enumerate(frames, start=1):
        image = read_image(frame_path)
        video_index = frame_number(frame_path) / 2.0
        tail_alpha = smoothstep(args.fade_start_video_frame, args.full_video_frame, video_index)
        filled, copied = synthesize_one_checker_period(image, tail_alpha, args.copy_strength)
        output_path = args.output_dir / frame_path.name
        write_image(output_path, filled, args.jpeg_quality)
        rows.append(
            {
                "frame": frame_path.name,
                "video_frame_index": video_index,
                "tail_alpha": tail_alpha,
                "copied_components": copied,
                "output_path": str(output_path),
            }
        )
        if index == 1 or index % args.progress_every == 0 or index == len(frames):
            print(
                f"[{index}/{len(frames)}] {frame_path.name}: "
                f"tail_alpha={tail_alpha:.3f}, copied={copied}"
            )

    encode_video(args.output_dir, args.output_video, args.width, args.height, args.fps)
    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "output_video": str(args.output_video),
        "frame_count": len(rows),
        "fade_start_video_frame": args.fade_start_video_frame,
        "full_video_frame": args.full_video_frame,
        "copy_strength": args.copy_strength,
        "method": "checkerboard-aware one-period dark-square synthesis on the Q42 tail",
        "rows_copied_total": int(sum(int(row["copied_components"]) for row in rows)),
        "output_width": args.width,
        "output_height": args.height,
        "fps": args.fps,
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
