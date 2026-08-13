#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flag)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
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


def estimate_pair_transform(
    previous: np.ndarray,
    current: np.ndarray,
    analysis_scale: float,
) -> tuple[float, float, float, float, int, int]:
    prev_small = cv2.resize(
        previous,
        None,
        fx=analysis_scale,
        fy=analysis_scale,
        interpolation=cv2.INTER_AREA,
    )
    curr_small = cv2.resize(
        current,
        None,
        fx=analysis_scale,
        fy=analysis_scale,
        interpolation=cv2.INTER_AREA,
    )
    prev_points = cv2.goodFeaturesToTrack(
        prev_small,
        maxCorners=700,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )
    if prev_points is None or len(prev_points) < 8:
        return 0.0, 0.0, 0.0, 1.0, 0, 0

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_small,
        curr_small,
        prev_points,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if curr_points is None or status is None:
        return 0.0, 0.0, 0.0, 1.0, 0, int(len(prev_points))

    good_prev = prev_points[status.reshape(-1) == 1]
    good_curr = curr_points[status.reshape(-1) == 1]
    if len(good_prev) < 8:
        return 0.0, 0.0, 0.0, 1.0, 0, int(len(good_prev))

    transform, inliers = cv2.estimateAffinePartial2D(
        good_prev,
        good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if transform is None or inliers is None or int(inliers.sum()) < 8:
        return 0.0, 0.0, 0.0, 1.0, 0, int(len(good_prev))

    dx = float(transform[0, 2] / analysis_scale)
    dy = float(transform[1, 2] / analysis_scale)
    angle = float(math.atan2(transform[1, 0], transform[0, 0]))
    scale = float(math.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
    return dx, dy, angle, scale, int(inliers.sum()), int(len(good_prev))


def moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    window = radius * 2 + 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.empty_like(values, dtype=np.float64)
    for dim in range(values.shape[1]):
        padded = np.pad(values[:, dim], (radius, radius), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def correction_matrix(
    dx: float,
    dy: float,
    angle: float,
    width: int,
    height: int,
    mode: str,
) -> np.ndarray:
    if mode == "translation":
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle * 180.0 / math.pi, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix.astype(np.float32)


def encode_video(frame_dir: Path, output_video: Path, width: int, height: int, fps: float) -> None:
    frames = list_frames(frame_dir)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame_path in frames:
            frame = read_image(frame_path)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-frame temporal stabilization without extra crop/zoom."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--motion-report-csv", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--smoothing-radius", type=int, default=45)
    parser.add_argument("--analysis-scale", type=float, default=0.35)
    parser.add_argument("--mode", choices=["translation", "rigid"], default="translation")
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

    transforms: list[tuple[float, float, float, float, int, int]] = []
    previous = cv2.resize(
        read_image(frames[0], grayscale=True),
        (args.width, args.height),
        interpolation=cv2.INTER_AREA,
    )
    print("Estimating full-frame motion...")
    for index, frame_path in enumerate(frames[1:], start=1):
        current = cv2.resize(
            read_image(frame_path, grayscale=True),
            (args.width, args.height),
            interpolation=cv2.INTER_AREA,
        )
        transforms.append(
            estimate_pair_transform(previous, current, analysis_scale=args.analysis_scale)
        )
        previous = current
        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(frames) - 1
        ):
            print(f"[motion {index}/{len(frames) - 1}] {frame_path.name}")

    transform_array = np.array(transforms, dtype=np.float64)
    steps = np.column_stack(
        [
            transform_array[:, 0],
            transform_array[:, 1],
            transform_array[:, 2],
            np.log(np.clip(transform_array[:, 3], 0.8, 1.2)),
        ]
    )
    trajectory = np.vstack([np.zeros(4, dtype=np.float64), np.cumsum(steps, axis=0)])
    smoothed = moving_average(trajectory, args.smoothing_radius)
    corrections = smoothed - trajectory
    if args.mode == "translation":
        corrections[:, 2:] = 0.0

    print("Writing stabilized frames...")
    rows: list[dict[str, float | int | str]] = []
    for index, frame_path in enumerate(frames, start=1):
        image = cv2.resize(
            read_image(frame_path),
            (args.width, args.height),
            interpolation=cv2.INTER_AREA,
        )
        dx, dy, angle = corrections[index - 1, 0:3]
        matrix = correction_matrix(
            float(dx),
            float(dy),
            float(angle),
            args.width,
            args.height,
            args.mode,
        )
        stabilized = cv2.warpAffine(
            image,
            matrix,
            (args.width, args.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        output_path = args.output_dir / frame_path.name
        write_image(output_path, stabilized, quality=args.jpeg_quality)

        transform_index = min(index - 1, len(transform_array) - 1)
        rows.append(
            {
                "frame": frame_path.name,
                "correction_dx": float(dx),
                "correction_dy": float(dy),
                "correction_angle_deg": float(angle * 180.0 / math.pi),
                "raw_pair_dx": float(transform_array[transform_index, 0]),
                "raw_pair_dy": float(transform_array[transform_index, 1]),
                "raw_pair_angle_deg": float(transform_array[transform_index, 2] * 180.0 / math.pi),
                "raw_pair_scale": float(transform_array[transform_index, 3]),
                "raw_pair_inliers": int(transform_array[transform_index, 4]),
                "raw_pair_tracked_points": int(transform_array[transform_index, 5]),
            }
        )

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(frames)
        ):
            print(
                f"[write {index}/{len(frames)}] {frame_path.name}: "
                f"shift=({dx:.1f},{dy:.1f})"
            )

    args.motion_report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.motion_report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    raw_motion = np.linalg.norm(transform_array[:, :2], axis=1)
    residual_steps = np.diff(smoothed[:, :2], axis=0)
    residual_motion = np.linalg.norm(residual_steps, axis=1)
    correction_motion = np.linalg.norm(corrections[:, :2], axis=1)
    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "output_video": str(args.output_video),
        "frame_count": len(frames),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "mode": args.mode,
        "smoothing_radius": args.smoothing_radius,
        "smoothing_window": args.smoothing_radius * 2 + 1,
        "analysis_scale": args.analysis_scale,
        "border_mode": "BORDER_REFLECT101",
        "no_extra_crop_or_zoom": True,
        "raw_motion_p95_px": float(np.percentile(raw_motion, 95)),
        "raw_motion_max_px": float(np.max(raw_motion)),
        "residual_smoothed_motion_p95_px_per_frame": float(np.percentile(residual_motion, 95)),
        "residual_smoothed_motion_max_px_per_frame": float(np.max(residual_motion)),
        "correction_p95_px": float(np.percentile(correction_motion, 95)),
        "correction_max_px": float(np.max(correction_motion)),
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Encoding video...")
    encode_video(args.output_dir, args.output_video, args.width, args.height, args.fps)
    print(f"Done: {args.output_video}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
