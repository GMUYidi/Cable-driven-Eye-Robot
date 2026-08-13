#!/usr/bin/env python3
"""
Repair a selected health_second frame range by tracking the true green markers.

The normal chessboard solver can reuse older homographies when only a partial
board is visible. In healthy footage the left/right green markers should land
on the same canvas point, so this script tracks the real marker on both raw
streams and applies a symmetric canvas-space translation to the two homography
matrices for the requested range.
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


def find_frame_path(folder: Path, stem: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        path = folder / f"{stem}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(folder / f"{stem}.*")


def marker_feature(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image)
    green_excess = green.astype(np.float32) - 0.5 * (
        red.astype(np.float32) + blue.astype(np.float32)
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    green_norm = cv2.normalize(green_excess, None, 0, 255, cv2.NORM_MINMAX)
    return (0.70 * green_norm + 0.30 * gray).astype(np.float32)


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


def choose_anchor_marker(image: np.ndarray) -> np.ndarray:
    candidates = detect_green_candidates(image, 12)
    if not candidates:
        raise RuntimeError("Could not detect an anchor green marker")

    height, width = image.shape[:2]
    guarded = [
        candidate
        for candidate in candidates
        if 0.10 * width < candidate.x < 0.95 * width
        and 0.08 * height < candidate.y < 0.95 * height
        and candidate.area >= 25.0
    ]
    if not guarded:
        guarded = candidates
    marker = guarded[0]
    return np.array([marker.x, marker.y], dtype=np.float64)


def snap_to_nearby_candidate(
    image: np.ndarray,
    point: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, int]:
    candidates = detect_green_candidates(image, 12)
    if not candidates:
        return point, -1

    best = min(
        candidates,
        key=lambda candidate: math.hypot(float(candidate.x - point[0]), float(candidate.y - point[1]))
        + 4.0 * candidate.rank,
    )
    distance = math.hypot(float(best.x - point[0]), float(best.y - point[1]))
    if distance <= max_distance:
        return np.array([best.x, best.y], dtype=np.float64), int(best.rank)
    return point, -1


def track_marker_direction(
    folder: Path,
    ordered_stems: list[str],
    start_index: int,
    step: int,
    start_point: np.ndarray,
    template_size: int,
    search_size: int,
    template_update: float,
    snap_distance: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int]]:
    tracks: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, int] = {}

    point = start_point.astype(np.float64).copy()
    start_image = read_image(find_frame_path(folder, ordered_stems[start_index]))
    template = crop_feature(start_image, point, template_size)

    index = start_index
    while 0 <= index < len(ordered_stems):
        stem = ordered_stems[index]
        image = read_image(find_frame_path(folder, stem))
        feature = marker_feature(image)
        height, width = feature.shape
        half_search = search_size // 2
        x0 = max(0, int(round(float(point[0]) - half_search)))
        y0 = max(0, int(round(float(point[1]) - half_search)))
        x1 = min(width, int(round(float(point[0]) + half_search)))
        y1 = min(height, int(round(float(point[1]) + half_search)))
        roi = feature[y0:y1, x0:x1]

        score = float("nan")
        if roi.shape[0] >= template_size and roi.shape[1] >= template_size:
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            point = np.array(
                [
                    x0 + max_location[0] + template_size / 2.0,
                    y0 + max_location[1] + template_size / 2.0,
                ],
                dtype=np.float64,
            )
            score = float(max_value)

        point, rank = snap_to_nearby_candidate(image, point, snap_distance)
        tracks[stem] = point.copy()
        scores[stem] = score
        ranks[stem] = rank

        new_patch = crop_feature(image, point, template_size)
        template = (1.0 - template_update) * template + template_update * new_patch
        index += step

    return tracks, scores, ranks


def track_marker(
    folder: Path,
    stems: list[str],
    anchor_stem: str,
    template_size: int,
    search_size: int,
    template_update: float,
    snap_distance: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int]]:
    anchor_index = stems.index(anchor_stem)
    anchor_image = read_image(find_frame_path(folder, anchor_stem))
    anchor_point = choose_anchor_marker(anchor_image)

    forward_tracks, forward_scores, forward_ranks = track_marker_direction(
        folder,
        stems,
        anchor_index,
        1,
        anchor_point,
        template_size,
        search_size,
        template_update,
        snap_distance,
    )
    backward_tracks, backward_scores, backward_ranks = track_marker_direction(
        folder,
        stems,
        anchor_index - 1,
        -1,
        anchor_point,
        template_size,
        search_size,
        template_update,
        snap_distance,
    )

    tracks = {**forward_tracks, **backward_tracks}
    scores = {**forward_scores, **backward_scores}
    ranks = {**forward_ranks, **backward_ranks}
    return tracks, scores, ranks


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
    parser.add_argument("--repair-start-frame", type=int, default=480)
    parser.add_argument("--repair-end-frame", type=int, default=720)
    parser.add_argument("--anchor-frame", type=int, default=540)
    parser.add_argument("--template-size", type=int, default=90)
    parser.add_argument("--search-size", type=int, default=230)
    parser.add_argument("--template-update", type=float, default=0.18)
    parser.add_argument("--snap-distance", type=float, default=70.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--backup-suffix", default="_before_mid_green_fix")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stitched_dir = args.output_root / "stitched_frames"
    matrices_dir = args.output_root / "matrices"
    npz_path = matrices_dir / "homography_matrices_and_points.npz"
    summary_path = matrices_dir / "homography_summary.csv"
    report_path = matrices_dir / "mid_green_alignment_repair_report.csv"
    metadata_path = matrices_dir / "mid_green_alignment_repair_metadata.json"

    for path in (args.left_dir, args.right_dir, stitched_dir, matrices_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not npz_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Missing health_second matrix outputs")

    all_stems = [
        stem
        for stem in list_frame_stems(args.left_dir)
        if (args.right_dir / f"{stem}.jpg").exists()
    ]
    repair_stems = [
        stem
        for stem in all_stems
        if args.repair_start_frame <= frame_number(stem) <= args.repair_end_frame
    ]
    anchor_stem = f"frame_{args.anchor_frame:04d}"
    if anchor_stem not in repair_stems:
        raise ValueError(f"Anchor {anchor_stem} is outside the selected repair range")

    matrix_npz = np.load(npz_path, allow_pickle=True)
    npz_data = {key: matrix_npz[key] for key in matrix_npz.files}
    rows, fieldnames = read_summary(summary_path)
    rows_by_stem = {Path(row["frame"]).stem: row for row in rows}

    left_tracks, left_scores, left_ranks = track_marker(
        args.left_dir,
        repair_stems,
        anchor_stem,
        args.template_size,
        args.search_size,
        args.template_update,
        args.snap_distance,
    )
    right_tracks, right_scores, right_ranks = track_marker(
        args.right_dir,
        repair_stems,
        anchor_stem,
        args.template_size,
        args.search_size,
        args.template_update,
        args.snap_distance,
    )

    output_dir = stitched_dir if args.overwrite else args.preview_dir
    if output_dir is None:
        output_dir = args.output_root / "debug_mid_green_fix_preview_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        backup_file(npz_path, args.backup_suffix)
        backup_file(summary_path, args.backup_suffix)

    report_rows: list[dict[str, float | int | str]] = []
    for index, stem in enumerate(repair_stems, start=1):
        left_image = read_image(find_frame_path(args.left_dir, stem))
        right_image = read_image(find_frame_path(args.right_dir, stem))
        previous_stitched = read_image(find_frame_path(stitched_dir, stem))
        output_size = (previous_stitched.shape[1], previous_stitched.shape[0])

        left_h = npz_data[f"{stem}_canvas_left_h"].astype(np.float64)
        right_h = npz_data[f"{stem}_canvas_right_h"].astype(np.float64)
        left_marker = left_tracks[stem]
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
        write_image(output_dir / f"{stem}.jpg", stitched, args.jpeg_quality)

        corrected_left_canvas = project_point(
            corrected_left_h, float(left_marker[0]), float(left_marker[1])
        )
        corrected_right_canvas = project_point(
            corrected_right_h, float(right_marker[0]), float(right_marker[1])
        )
        corrected_error = float(np.linalg.norm(corrected_left_canvas - corrected_right_canvas))

        if args.overwrite:
            npz_data[f"{stem}_canvas_left_h"] = corrected_left_h
            npz_data[f"{stem}_canvas_right_h"] = corrected_right_h
            direct_h = np.linalg.inv(corrected_right_h) @ corrected_left_h
            if abs(direct_h[2, 2]) > 1e-12:
                direct_h = direct_h / direct_h[2, 2]
            npz_data[f"{stem}_direct_left_to_right_h"] = direct_h
            npz_data[f"{stem}_mid_green_repair_left_shift"] = left_shift.astype(np.float64)
            npz_data[f"{stem}_mid_green_repair_right_shift"] = right_shift.astype(np.float64)
            npz_data[f"{stem}_mid_green_repair_left_marker_raw"] = left_marker.astype(np.float64)
            npz_data[f"{stem}_mid_green_repair_right_marker_raw"] = right_marker.astype(np.float64)
            npz_data[f"{stem}_mid_green_repair_original_error"] = np.array(original_error)
            npz_data[f"{stem}_mid_green_repair_corrected_error"] = np.array(corrected_error)

            row = rows_by_stem.get(stem)
            if row is not None:
                row["status"] = "green_aligned_mid_repair"
                row["source"] = "tracked_marker"
                row["green_error"] = f"{corrected_error:.6f}"
                row["notes"] = (
                    (row.get("notes", "") + ";") if row.get("notes") else ""
                ) + f"mid-green-repair;pre_error={original_error:.3f}"

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
                "left_track_score": float(left_scores.get(stem, float("nan"))),
                "right_track_score": float(right_scores.get(stem, float("nan"))),
                "left_marker_rank": int(left_ranks.get(stem, -1)),
                "right_marker_rank": int(right_ranks.get(stem, -1)),
                "output_width": int(output_size[0]),
                "output_height": int(output_size[1]),
                "mode": "overwrite" if args.overwrite else "preview",
            }
        )

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(repair_stems)
        ):
            print(
                f"[{index}/{len(repair_stems)}] {stem}: "
                f"green {original_error:.1f}px -> {corrected_error:.3f}px"
            )

    if args.overwrite:
        np.savez_compressed(npz_path, **npz_data)
        write_summary(summary_path, rows, fieldnames)

    if report_rows:
        target_report = report_path if args.overwrite else output_dir / "mid_green_preview_report.csv"
        with target_report.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    metadata = {
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "output_root": str(args.output_root),
        "repair_start_frame": args.repair_start_frame,
        "repair_end_frame": args.repair_end_frame,
        "anchor_frame": args.anchor_frame,
        "repaired_frame_count": len(report_rows),
        "mode": "overwrite" if args.overwrite else "preview",
        "output_dir": str(output_dir),
        "mean_original_green_error_px": float(
            np.mean([row["original_green_error_px"] for row in report_rows])
        )
        if report_rows
        else None,
        "max_original_green_error_px": float(
            np.max([row["original_green_error_px"] for row in report_rows])
        )
        if report_rows
        else None,
        "max_corrected_green_error_px": float(
            np.max([row["corrected_green_error_px"] for row in report_rows])
        )
        if report_rows
        else None,
    }
    target_metadata = metadata_path if args.overwrite else output_dir / "mid_green_preview_metadata.json"
    target_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done. Processed {len(report_rows)} frames in {metadata['mode']} mode.")
    print(f"Output frames: {output_dir}")
    print(f"Metadata: {target_metadata}")


if __name__ == "__main__":
    main()
