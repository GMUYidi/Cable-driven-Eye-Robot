#!/usr/bin/env python3
"""
Repair the initial health_second frames whose homographies were reused before
the first reliable chessboard solution.

The first part of health_second has visible green markers, but the original
run reused frame_0000 homographies until frame_0212. This script tracks the
right marker backward from frame_0212, detects the left marker with edge
guards, then applies a symmetric canvas-space translation so the healthy
left/right green markers overlap in the repaired interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from stitch_q3_hess_consistent import (
    blend_warped_pair,
    detect_green_candidates,
    project_point,
    read_image,
    translation_matrix,
    write_image,
)


DEFAULT_LEFT_DIR = Path(r"D:\visualization2026\visualization\left_downsample2_health_second")
DEFAULT_RIGHT_DIR = Path(r"D:\visualization2026\visualization\right_downsample2_health_second")
DEFAULT_OUTPUT_ROOT = Path(r"D:\visualization2026\dowsample2_health_second")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def frame_number(stem: str) -> int:
    return int(stem.split("_")[-1])


def list_frame_stems(folder: Path) -> list[str]:
    return sorted(
        [p.stem for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=frame_number,
    )


def marker_feature(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image)
    green_excess = green.astype(np.float32) - 0.5 * (
        red.astype(np.float32) + blue.astype(np.float32)
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    green_norm = cv2.normalize(green_excess, None, 0, 255, cv2.NORM_MINMAX)
    return (0.65 * green_norm + 0.35 * gray).astype(np.float32)


def crop_feature(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    feature = marker_feature(image)
    height, width = feature.shape
    half = size // 2
    x0 = int(round(float(center[0]) - half))
    y0 = int(round(float(center[1]) - half))
    x1 = x0 + size
    y1 = y0 + size

    patch = np.zeros((size, size), dtype=np.float32)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(width, x1)
    sy1 = min(height, y1)
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = feature[sy0:sy1, sx0:sx1]
    return patch


def track_right_marker_backward(
    right_dir: Path,
    start_stem: str,
    stems: list[str],
    template_size: int,
    search_size: int,
    template_update: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    start_image = read_image(right_dir / f"{start_stem}.jpg")
    start_candidates = detect_green_candidates(start_image, 8)
    lower_candidates = [
        candidate
        for candidate in start_candidates
        if candidate.y > 0.55 * start_image.shape[0] and candidate.x > 80
    ]
    if not lower_candidates:
        lower_candidates = start_candidates
    if not lower_candidates:
        raise RuntimeError(f"Could not initialize right marker from {start_stem}")

    point = np.array([lower_candidates[0].x, lower_candidates[0].y], dtype=np.float64)
    tracks = {start_stem: point.copy()}
    scores = {start_stem: 1.0}
    template = crop_feature(start_image, point, template_size)

    ordered = sorted(stems, key=frame_number, reverse=True)
    ordered = [stem for stem in ordered if frame_number(stem) < frame_number(start_stem)]
    for stem in ordered:
        image = read_image(right_dir / f"{stem}.jpg")
        feature = marker_feature(image)
        height, width = feature.shape
        half_search = search_size // 2
        x0 = max(0, int(round(float(point[0]) - half_search)))
        y0 = max(0, int(round(float(point[1]) - half_search)))
        x1 = min(width, int(round(float(point[0]) + half_search)))
        y1 = min(height, int(round(float(point[1]) + half_search)))
        roi = feature[y0:y1, x0:x1]
        if roi.shape[0] < template_size or roi.shape[1] < template_size:
            tracks[stem] = point.copy()
            scores[stem] = float("nan")
            continue

        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        point = np.array(
            [
                x0 + max_location[0] + template_size / 2.0,
                y0 + max_location[1] + template_size / 2.0,
            ],
            dtype=np.float64,
        )
        tracks[stem] = point.copy()
        scores[stem] = float(max_value)
        new_patch = crop_feature(image, point, template_size)
        template = (1.0 - template_update) * template + template_update * new_patch

    return tracks, scores


def choose_left_marker(image: np.ndarray) -> tuple[np.ndarray, int]:
    candidates = detect_green_candidates(image, 12)
    guarded = [
        candidate
        for candidate in candidates
        if candidate.x > 100 and candidate.y > 0.55 * image.shape[0]
    ]
    if not guarded:
        guarded = candidates
    if not guarded:
        raise RuntimeError("Could not detect left green marker")
    candidate = guarded[0]
    return np.array([candidate.x, candidate.y], dtype=np.float64), candidate.rank


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def read_summary(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_summary(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_RIGHT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repair-start-frame", type=int, default=0)
    parser.add_argument("--repair-end-frame", type=int, default=210)
    parser.add_argument("--anchor-frame", type=int, default=212)
    parser.add_argument("--template-size", type=int, default=90)
    parser.add_argument("--search-size", type=int, default=220)
    parser.add_argument("--template-update", type=float, default=0.20)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stitched_dir = args.output_root / "stitched_frames"
    matrices_dir = args.output_root / "matrices"
    npz_path = matrices_dir / "homography_matrices_and_points.npz"
    summary_path = matrices_dir / "homography_summary.csv"
    report_path = matrices_dir / "initial_green_alignment_repair_report.csv"
    metadata_path = matrices_dir / "initial_green_alignment_repair_metadata.json"

    for path in (args.left_dir, args.right_dir, stitched_dir, matrices_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not npz_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Missing health_second matrix outputs")

    anchor_stem = f"frame_{args.anchor_frame:04d}"
    stems = [
        stem
        for stem in list_frame_stems(args.left_dir)
        if args.repair_start_frame <= frame_number(stem) <= args.anchor_frame
    ]
    repair_stems = [
        stem
        for stem in stems
        if args.repair_start_frame <= frame_number(stem) <= args.repair_end_frame
    ]

    matrix_npz = np.load(npz_path, allow_pickle=True)
    npz_data = {key: matrix_npz[key] for key in matrix_npz.files}
    rows, fieldnames = read_summary(summary_path)
    rows_by_stem = {Path(row["frame"]).stem: row for row in rows}

    right_tracks, right_scores = track_right_marker_backward(
        args.right_dir,
        anchor_stem,
        stems,
        template_size=args.template_size,
        search_size=args.search_size,
        template_update=args.template_update,
    )

    if not args.overwrite:
        raise RuntimeError("Use --overwrite to modify health_second outputs.")

    backup_file(npz_path, "_before_initial_green_fix")
    backup_file(summary_path, "_before_initial_green_fix")

    report_rows: list[dict[str, float | int | str]] = []
    for index, stem in enumerate(repair_stems, start=1):
        left_path = args.left_dir / f"{stem}.jpg"
        right_path = args.right_dir / f"{stem}.jpg"
        output_path = stitched_dir / f"{stem}.jpg"
        if stem not in right_tracks:
            raise RuntimeError(f"Missing right marker track for {stem}")

        left_image = read_image(left_path)
        right_image = read_image(right_path)
        previous_stitched = read_image(output_path)
        output_size = (previous_stitched.shape[1], previous_stitched.shape[0])

        left_h = npz_data[f"{stem}_canvas_left_h"].astype(np.float64)
        right_h = npz_data[f"{stem}_canvas_right_h"].astype(np.float64)
        left_marker, left_rank = choose_left_marker(left_image)
        right_marker = right_tracks[stem]

        left_canvas = project_point(left_h, float(left_marker[0]), float(left_marker[1]))
        right_canvas = project_point(right_h, float(right_marker[0]), float(right_marker[1]))
        measured_vec = left_canvas - right_canvas
        original_error = float(np.linalg.norm(measured_vec))

        left_shift = -0.5 * measured_vec
        right_shift = 0.5 * measured_vec
        corrected_left_h = translation_matrix(float(left_shift[0]), float(left_shift[1])) @ left_h
        corrected_right_h = translation_matrix(float(right_shift[0]), float(right_shift[1])) @ right_h

        stitched = blend_warped_pair(
            left_image,
            right_image,
            corrected_left_h,
            corrected_right_h,
            output_size,
        )
        write_image(output_path, stitched, args.jpeg_quality)

        corrected_left_canvas = project_point(
            corrected_left_h, float(left_marker[0]), float(left_marker[1])
        )
        corrected_right_canvas = project_point(
            corrected_right_h, float(right_marker[0]), float(right_marker[1])
        )
        corrected_error = float(np.linalg.norm(corrected_left_canvas - corrected_right_canvas))

        npz_data[f"{stem}_canvas_left_h"] = corrected_left_h
        npz_data[f"{stem}_canvas_right_h"] = corrected_right_h
        direct_h = np.linalg.inv(corrected_right_h) @ corrected_left_h
        if abs(direct_h[2, 2]) > 1e-12:
            direct_h = direct_h / direct_h[2, 2]
        npz_data[f"{stem}_direct_left_to_right_h"] = direct_h
        npz_data[f"{stem}_green_repair_left_shift"] = left_shift.astype(np.float64)
        npz_data[f"{stem}_green_repair_right_shift"] = right_shift.astype(np.float64)
        npz_data[f"{stem}_green_repair_left_marker_raw"] = left_marker.astype(np.float64)
        npz_data[f"{stem}_green_repair_right_marker_raw"] = right_marker.astype(np.float64)
        npz_data[f"{stem}_green_repair_original_error"] = np.array(original_error)
        npz_data[f"{stem}_green_repair_corrected_error"] = np.array(corrected_error)

        row = rows_by_stem.get(stem)
        if row is not None:
            row["status"] = "green_aligned_repair"
            row["source"] = "marker_track"
            row["green_error"] = f"{corrected_error:.6f}"
            row["notes"] = (
                (row.get("notes", "") + ";") if row.get("notes") else ""
            ) + f"initial-green-repair;pre_error={original_error:.3f}"

        report_rows.append(
            {
                "frame": f"{stem}.jpg",
                "original_green_error_px": original_error,
                "corrected_green_error_px": corrected_error,
                "left_shift_x": float(left_shift[0]),
                "left_shift_y": float(left_shift[1]),
                "right_shift_x": float(right_shift[0]),
                "right_shift_y": float(right_shift[1]),
                "left_marker_x": float(left_marker[0]),
                "left_marker_y": float(left_marker[1]),
                "right_marker_x": float(right_marker[0]),
                "right_marker_y": float(right_marker[1]),
                "right_track_score": float(right_scores.get(stem, float("nan"))),
                "left_marker_rank": int(left_rank),
                "output_width": int(output_size[0]),
                "output_height": int(output_size[1]),
            }
        )

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(repair_stems)
        ):
            print(
                f"[{index}/{len(repair_stems)}] {stem}: "
                f"green {original_error:.1f}px -> {corrected_error:.3f}px"
            )

    np.savez_compressed(npz_path, **npz_data)
    write_summary(summary_path, rows, fieldnames)

    if report_rows:
        with report_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    metadata = {
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "output_root": str(args.output_root),
        "anchor_frame": args.anchor_frame,
        "repair_start_frame": args.repair_start_frame,
        "repair_end_frame": args.repair_end_frame,
        "repaired_frame_count": len(report_rows),
        "mean_original_green_error_px": float(
            np.mean([row["original_green_error_px"] for row in report_rows])
        )
        if report_rows
        else None,
        "max_corrected_green_error_px": float(
            np.max([row["corrected_green_error_px"] for row in report_rows])
        )
        if report_rows
        else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Repaired {len(report_rows)} frames.")
    print(f"Report: {report_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
