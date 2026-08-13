#!/usr/bin/env python3
"""
Refine Hess-matched stereo stitching from raw left/right green markers.

This script is meant for cases where a fused stitched frame or a chessboard-like
green cast can confuse per-frame marker selection. It detects the green marker
in each raw left/right frame, projects those detections through the base canvas
homographies, fits a robust pixel mapping from simulated left-right gaze angles,
and applies split canvas translations so the rendered L/R marker vector follows
the gaze-angle trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio

from stitch_q3_hess_consistent import (
    blend_warped_pair,
    flatten_matrix,
    matrix_frame_names,
    read_image,
    resize_to_width,
    translation_matrix,
    write_image,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class MarkerCandidate:
    x: float
    y: float
    score: float
    area: float
    circularity: float
    aspect: float
    green_excess: float
    saturation: float
    green_minus_red: float
    rank: int = -1
    canvas_x: float = float("nan")
    canvas_y: float = float("nan")

    @property
    def raw_point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def canvas_point(self) -> np.ndarray:
        return np.array([self.canvas_x, self.canvas_y], dtype=np.float64)


@dataclass
class PairSelection:
    left_rank: int
    right_rank: int
    measured_vec: np.ndarray
    error_to_target: float
    cost: float


def list_frames(input_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def frame_number(stem: str) -> int:
    return int(stem.split("_")[-1])


def project_point(h_matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    p = h_matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(p[2])) < 1e-12:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def detect_raw_green_marker_candidates(
    image: np.ndarray,
    max_candidates: int,
) -> list[MarkerCandidate]:
    blue, green, red = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _value = cv2.split(hsv)

    green_excess = green.astype(np.float32) - 0.5 * (
        red.astype(np.float32) + blue.astype(np.float32)
    )
    green_minus_red = green.astype(np.int16) - red.astype(np.int16)
    green_minus_blue = green.astype(np.int16) - blue.astype(np.int16)

    # The real marker can be pale in the right-eye image, so the mask is not
    # very saturated. Shape/area filtering below removes chessboard artifacts.
    mask = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 22)
        & (green >= 50)
        & (green_excess >= 5.5)
        & (green_minus_red >= -6)
        & (green_minus_blue >= -12)
    ).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    candidates: list[MarkerCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 70.0 or area > 5200.0 or perimeter <= 0.0:
            continue

        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, box_w, box_h = cv2.boundingRect(contour)
        aspect = max(box_w / box_h, box_h / box_w) if box_w and box_h else 999.0
        if circularity < 0.28 or aspect > 3.0:
            continue

        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-12:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if cx < 4 or cy < 4 or cx > width - 4 or cy > height - 4:
            continue

        radius = 24
        x0 = max(0, int(cx) - radius)
        x1 = min(width, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(height, int(cy) + radius + 1)
        patch = image[y0:y1, x0:x1]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        patch_b, patch_g, patch_r = cv2.split(patch)

        patch_excess = float(
            (
                patch_g.astype(np.float32)
                - 0.5 * (patch_r.astype(np.float32) + patch_b.astype(np.float32))
            ).mean()
        )
        patch_saturation = float(patch_hsv[:, :, 1].mean())
        patch_green_minus_red = float(
            (patch_g.astype(np.int16) - patch_r.astype(np.int16)).mean()
        )
        if patch_excess < 4.0 or patch_saturation < 14.0:
            continue

        # The high-score chessboard false positives tend to be huge and not
        # round; the real marker is compact, moderately round, and green.
        area_term = min(area, 2800.0) ** 0.72
        round_term = max(circularity, 0.05) ** 1.15
        color_term = max(patch_excess, 1.0) * max(patch_saturation, 1.0)
        aspect_penalty = 1.0 + 0.18 * max(0.0, aspect - 1.0)
        score = area_term * round_term * color_term / aspect_penalty
        candidates.append(
            MarkerCandidate(
                x=cx,
                y=cy,
                score=float(score),
                area=area,
                circularity=circularity,
                aspect=float(aspect),
                green_excess=patch_excess,
                saturation=patch_saturation,
                green_minus_red=patch_green_minus_red,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = candidates[:max_candidates]
    for rank, candidate in enumerate(candidates):
        candidate.rank = rank
    return candidates


def detect_fused_green_marker_candidates(
    image: np.ndarray,
    max_candidates: int,
) -> list[MarkerCandidate]:
    blue, green, red = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _value = cv2.split(hsv)
    green_excess = green.astype(np.float32) - 0.5 * (
        red.astype(np.float32) + blue.astype(np.float32)
    )
    green_minus_red = green.astype(np.int16) - red.astype(np.int16)
    green_minus_blue = green.astype(np.int16) - blue.astype(np.int16)

    mask = (
        (hue >= 35)
        & (hue <= 98)
        & (saturation >= 16)
        & (green >= 45)
        & (green_excess >= 3.2)
        & (green_minus_red >= -8)
        & (green_minus_blue >= -14)
    ).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    candidates: list[MarkerCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 35.0 or area > 7000.0 or perimeter <= 0.0:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, box_w, box_h = cv2.boundingRect(contour)
        aspect = max(box_w / box_h, box_h / box_w) if box_w and box_h else 999.0
        if circularity < 0.16 or aspect > 4.2:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-12:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if cx < 4 or cy < 4 or cx > width - 4 or cy > height - 4:
            continue

        radius = 26
        x0 = max(0, int(cx) - radius)
        x1 = min(width, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(height, int(cy) + radius + 1)
        patch = image[y0:y1, x0:x1]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        patch_b, patch_g, patch_r = cv2.split(patch)
        patch_excess = float(
            (
                patch_g.astype(np.float32)
                - 0.5 * (patch_r.astype(np.float32) + patch_b.astype(np.float32))
            ).mean()
        )
        patch_saturation = float(patch_hsv[:, :, 1].mean())
        patch_green_minus_red = float(
            (patch_g.astype(np.int16) - patch_r.astype(np.int16)).mean()
        )
        if patch_excess < 2.4 or patch_saturation < 10.0:
            continue

        area_term = min(area, 3600.0) ** 0.66
        round_term = max(circularity, 0.04)
        color_term = max(patch_excess, 1.0) * max(patch_saturation, 1.0)
        score = area_term * round_term * color_term / (1.0 + 0.12 * max(0.0, aspect - 1.0))
        candidates.append(
            MarkerCandidate(
                x=cx,
                y=cy,
                score=float(score),
                area=area,
                circularity=circularity,
                aspect=float(aspect),
                green_excess=patch_excess,
                saturation=patch_saturation,
                green_minus_red=patch_green_minus_red,
                canvas_x=cx,
                canvas_y=cy,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = candidates[:max_candidates]
    for rank, candidate in enumerate(candidates):
        candidate.rank = rank
    return candidates


def load_angle_splits(mat_path: Path, stems: list[str], mat_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    data = sio.loadmat(str(mat_path))
    if mat_key not in data:
        available = ", ".join(key for key in data if not key.startswith("__"))
        raise KeyError(f"Missing {mat_key} in {mat_path}. Available: {available}")
    angles = np.asarray(data[mat_key], dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] < 4:
        raise ValueError(f"{mat_key} must be N-by-4, got {angles.shape}")

    frame_indices = np.array([frame_number(stem) for stem in stems], dtype=np.int64)
    if int(frame_indices.max(initial=0)) >= len(angles):
        raise ValueError(f"Angle data has {len(angles)} rows, max frame index is {frame_indices.max()}")
    splits = angles[frame_indices, 0:2] - angles[frame_indices, 2:4]
    metadata = {
        "mat_key": mat_key,
        "angle_rows": int(len(angles)),
        "frame_index_min": int(frame_indices.min(initial=0)),
        "frame_index_max": int(frame_indices.max(initial=0)),
        "split_min_xy_deg": splits.min(axis=0).tolist(),
        "split_max_xy_deg": splits.max(axis=0).tolist(),
    }
    return splits, metadata


def fit_linear_map(
    source_vectors: np.ndarray,
    measured_vectors: np.ndarray,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = (
        np.isfinite(source_vectors).all(axis=1)
        & np.isfinite(measured_vectors).all(axis=1)
        & (np.linalg.norm(source_vectors, axis=1) > 1e-9)
    )
    active = finite.copy()
    mapping: np.ndarray | None = None
    for _ in range(12):
        if int(active.sum()) < min_inliers:
            break
        coeff, *_ = np.linalg.lstsq(source_vectors[active], measured_vectors[active], rcond=None)
        mapping = coeff.T
        predicted = source_vectors @ mapping.T
        errors = np.linalg.norm(measured_vectors - predicted, axis=1)
        active_errors = errors[active]
        median = float(np.median(active_errors))
        mad = float(np.median(np.abs(active_errors - median)) + 1e-6)
        threshold = max(45.0, median + 2.6 * 1.4826 * mad)
        new_active = finite & (errors < threshold)
        if np.array_equal(new_active, active):
            break
        active = new_active

    if mapping is None:
        if int(finite.sum()) < 2:
            raise RuntimeError("Not enough finite marker vectors to fit mapping.")
        coeff, *_ = np.linalg.lstsq(source_vectors[finite], measured_vectors[finite], rcond=None)
        mapping = coeff.T
        active = finite

    final_errors = np.linalg.norm(measured_vectors - (source_vectors @ mapping.T), axis=1)
    return mapping, active, final_errors


def choose_pairs(
    left_candidates: list[list[MarkerCandidate]],
    right_candidates: list[list[MarkerCandidate]],
    target_vectors: np.ndarray,
    rank_penalty: float,
    distance_penalty: float,
) -> list[PairSelection | None]:
    selections: list[PairSelection | None] = []
    for left_list, right_list, target in zip(left_candidates, right_candidates, target_vectors):
        best: PairSelection | None = None
        for left in left_list:
            for right in right_list:
                measured = left.canvas_point - right.canvas_point
                if not np.isfinite(measured).all():
                    continue
                distance = float(np.linalg.norm(measured))
                if distance > 650.0:
                    continue
                error = float(np.linalg.norm(measured - target))
                cost = (
                    error
                    + rank_penalty * float(left.rank + right.rank)
                    + distance_penalty * max(0.0, distance - 420.0)
                )
                selection = PairSelection(
                    left_rank=left.rank,
                    right_rank=right.rank,
                    measured_vec=measured,
                    error_to_target=error,
                    cost=float(cost),
                )
                if best is None or selection.cost < best.cost:
                    best = selection
        selections.append(best)
    return selections


def choose_fused_pairs(
    fused_candidates: list[list[MarkerCandidate]],
    target_vectors: np.ndarray,
    rank_penalty: float,
) -> list[PairSelection | None]:
    selections: list[PairSelection | None] = []
    for candidates, target in zip(fused_candidates, target_vectors):
        best: PairSelection | None = None
        for left_index, left in enumerate(candidates[:12]):
            for right_index, right in enumerate(candidates[:12]):
                if left_index == right_index:
                    continue
                measured = left.canvas_point - right.canvas_point
                if not np.isfinite(measured).all():
                    continue
                distance = float(np.linalg.norm(measured))
                if distance < 8.0 or distance > 700.0:
                    continue
                error = float(np.linalg.norm(measured - target))
                cost = error + rank_penalty * float(left.rank + right.rank)
                selection = PairSelection(
                    left_rank=left.rank,
                    right_rank=right.rank,
                    measured_vec=measured,
                    error_to_target=error,
                    cost=float(cost),
                )
                if best is None or selection.cost < best.cost:
                    best = selection
        selections.append(best)
    return selections


def interpolate_vectors(vectors: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    result = vectors.astype(np.float64).copy()
    indices = np.arange(len(vectors), dtype=np.float64)
    reliable_indices = indices[reliable]
    if reliable_indices.size == 0:
        return np.zeros_like(result)
    for dim in range(vectors.shape[1]):
        result[:, dim] = np.interp(indices, reliable_indices, vectors[reliable, dim])
    return result


def mark_temporal_outliers(
    vectors: np.ndarray,
    reliable: np.ndarray,
    radius: int,
    max_deviation: float,
    iterations: int,
) -> np.ndarray:
    mask = reliable.copy()
    for _ in range(iterations):
        bad: list[int] = []
        for index in range(len(vectors)):
            if not mask[index] or not np.isfinite(vectors[index]).all():
                continue
            start = max(0, index - radius)
            stop = min(len(vectors), index + radius + 1)
            neighbors = [i for i in range(start, stop) if i != index and mask[i]]
            if len(neighbors) < 3:
                continue
            local_median = np.median(vectors[neighbors], axis=0)
            if float(np.linalg.norm(vectors[index] - local_median)) > max_deviation:
                bad.append(index)
        if not bad:
            break
        mask[bad] = False
    return mask


def smooth_vectors(vectors: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return vectors
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    result = np.empty_like(vectors, dtype=np.float64)
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    for dim in range(vectors.shape[1]):
        padded = np.pad(vectors[:, dim], (pad, pad), mode="edge")
        result[:, dim] = np.convolve(padded, kernel, mode="valid")
    return result


def backup_file(path: Path, suffix: str) -> None:
    if path.exists():
        backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not backup.exists():
            shutil.copy2(path, backup)


def make_debug_sheet(
    output_path: Path,
    stems: list[str],
    output_dir: Path,
    rows_by_stem: dict[str, dict[str, Any]],
    jpeg_quality: int,
) -> None:
    tiles: list[np.ndarray] = []
    for stem in stems:
        row = rows_by_stem.get(stem)
        image_path = output_dir / f"{stem}.jpg"
        if row is None or not image_path.exists():
            continue
        image = read_image(image_path)
        target = np.array([row["target_dx"], row["target_dy"]], dtype=np.float64)
        left_point = np.array([row["selected_left_canvas_x"], row["selected_left_canvas_y"]], dtype=np.float64)
        right_point = np.array([row["selected_right_canvas_x"], row["selected_right_canvas_y"]], dtype=np.float64)
        final_left = left_point + np.array([row["left_translation_x"], row["left_translation_y"]])
        final_right = right_point + np.array([row["right_translation_x"], row["right_translation_y"]])

        annotated = image.copy()
        if np.isfinite(final_left).all() and np.isfinite(final_right).all():
            lp = tuple(np.round(final_left).astype(int))
            rp = tuple(np.round(final_right).astype(int))
            cv2.circle(annotated, lp, 18, (0, 0, 255), 4)
            cv2.circle(annotated, rp, 18, (255, 0, 0), 4)
            cv2.arrowedLine(annotated, rp, lp, (0, 255, 255), 3, tipLength=0.14)
        text = (
            f"{stem} target=({target[0]:.1f},{target[1]:.1f}) "
            f"err={row['final_error_to_target']:.1f}"
        )
        cv2.putText(annotated, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
        tiles.append(resize_to_width(annotated, 520))

    if not tiles:
        return
    rows: list[np.ndarray] = []
    for index in range(0, len(tiles), 2):
        pair = tiles[index : index + 2]
        height = max(tile.shape[0] for tile in pair)
        padded = [
            cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for tile in pair
        ]
        if len(padded) == 1:
            padded.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    write_image(output_path, sheet, jpeg_quality)


def build_parser() -> argparse.ArgumentParser:
    output_root = Path(r"D:\visualization2026\dowsample2_Q42_hess_matched")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="Q42")
    parser.add_argument("--left-dir", type=Path, default=Path(r"D:\visualization2026\visualization\left_downsample2_Q42"))
    parser.add_argument("--right-dir", type=Path, default=Path(r"D:\visualization2026\visualization\right_downsample2_Q42"))
    parser.add_argument(
        "--base-matrix-npz",
        type=Path,
        default=Path(r"D:\visualization2026\dowsample2_Q42\matrices\applied_health_second_homographies_Q42.npz"),
    )
    parser.add_argument("--base-stitched-dir", type=Path, default=Path(r"D:\visualization2026\dowsample2_Q42\stitched_frames"))
    parser.add_argument(
        "--strabismus-mat",
        type=Path,
        default=Path(r"D:\visualization2026\Hess chart\bilinear interpolation\strabismus data\simulated_strabismus_Q42.mat"),
    )
    parser.add_argument("--mat-key", default="SGazeAngleSmoothed_cell")
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--output-dir", type=Path, default=output_root / "stitched_frames")
    parser.add_argument("--matrix-output-dir", type=Path, default=output_root / "matrices")
    parser.add_argument("--report-csv", type=Path, default=output_root / "hess_match_report.csv")
    parser.add_argument("--metadata-json", type=Path, default=output_root / "hess_match_metadata.json")
    parser.add_argument("--debug-dir", type=Path, default=output_root / "debug")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--rank-penalty", type=float, default=12.0)
    parser.add_argument("--distance-penalty", type=float, default=0.25)
    parser.add_argument("--refit-iterations", type=int, default=4)
    parser.add_argument("--min-map-inliers", type=int, default=180)
    parser.add_argument("--max-raw-pair-error", type=float, default=95.0)
    parser.add_argument("--max-correction-norm", type=float, default=180.0)
    parser.add_argument("--max-temporal-deviation", type=float, default=65.0)
    parser.add_argument("--temporal-radius", type=int, default=5)
    parser.add_argument("--temporal-iterations", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for folder in (args.left_dir, args.right_dir, args.base_stitched_dir):
        if not folder.exists():
            raise FileNotFoundError(folder)
    for path in (args.base_matrix_npz, args.strabismus_mat):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matrix_output_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (
            args.matrix_output_dir / f"hess_matched_homographies_{args.case_id}.npz",
            args.report_csv,
            args.metadata_json,
        ):
            backup_file(path, "_before_raw_green_refine")

    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    matrix_npz = np.load(args.base_matrix_npz, allow_pickle=True)
    stems = matrix_frame_names(matrix_npz)
    angle_splits, angle_metadata = load_angle_splits(args.strabismus_mat, stems, args.mat_key)

    all_left: list[list[MarkerCandidate]] = []
    all_right: list[list[MarkerCandidate]] = []
    all_fused: list[list[MarkerCandidate]] = []
    top_vectors: list[np.ndarray] = []

    print("Detecting raw left/right green markers and projecting to canvas...")
    for index, stem in enumerate(stems, start=1):
        left_path = left_frames.get(stem)
        right_path = right_frames.get(stem)
        if left_path is None or right_path is None:
            raise FileNotFoundError(f"Missing raw frame pair for {stem}")
        base_path = args.base_stitched_dir / f"{stem}.jpg"
        if not base_path.exists():
            raise FileNotFoundError(base_path)

        left_h = matrix_npz[f"{stem}_canvas_left_h"].astype(np.float64)
        right_h = matrix_npz[f"{stem}_canvas_right_h"].astype(np.float64)
        left_candidates = detect_raw_green_marker_candidates(read_image(left_path), args.max_candidates)
        right_candidates = detect_raw_green_marker_candidates(read_image(right_path), args.max_candidates)
        fused_candidates = detect_fused_green_marker_candidates(
            read_image(base_path),
            args.max_candidates,
        )

        for candidate in left_candidates:
            candidate.canvas_x, candidate.canvas_y = project_point(left_h, candidate.raw_point)
        for candidate in right_candidates:
            candidate.canvas_x, candidate.canvas_y = project_point(right_h, candidate.raw_point)

        all_left.append(left_candidates)
        all_right.append(right_candidates)
        all_fused.append(fused_candidates)
        if left_candidates and right_candidates:
            top_vectors.append(left_candidates[0].canvas_point - right_candidates[0].canvas_point)
        else:
            top_vectors.append(np.array([np.nan, np.nan], dtype=np.float64))

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(stems)
        ):
            print(
                f"[detect {index}/{len(stems)}] {stem}: "
                f"L={len(left_candidates)} R={len(right_candidates)} F={len(fused_candidates)}"
            )

    measured_initial = np.vstack(top_vectors)
    mapping, mapping_inliers, mapping_errors = fit_linear_map(
        angle_splits,
        measured_initial,
        min_inliers=args.min_map_inliers,
    )
    target_vectors = angle_splits @ mapping.T

    selections: list[PairSelection | None] = []
    for iteration in range(args.refit_iterations):
        selections = choose_pairs(
            all_left,
            all_right,
            target_vectors,
            rank_penalty=args.rank_penalty,
            distance_penalty=args.distance_penalty,
        )
        selected_vectors = np.vstack(
            [
                selection.measured_vec if selection is not None else np.array([np.nan, np.nan])
                for selection in selections
            ]
        )
        mapping, mapping_inliers, mapping_errors = fit_linear_map(
            angle_splits,
            selected_vectors,
            min_inliers=args.min_map_inliers,
        )
        target_vectors = angle_splits @ mapping.T
        print(
            f"Refit {iteration + 1}/{args.refit_iterations}: "
            f"inliers={int(mapping_inliers.sum())}, "
            f"median_error={float(np.nanmedian(mapping_errors[mapping_inliers])):.2f}px"
        )

    selections = choose_pairs(
        all_left,
        all_right,
        target_vectors,
        rank_penalty=args.rank_penalty,
        distance_penalty=args.distance_penalty,
    )
    fused_selections = choose_fused_pairs(
        all_fused,
        target_vectors,
        rank_penalty=max(0.0, args.rank_penalty * 0.35),
    )

    raw_measured = np.zeros((len(stems), 2), dtype=np.float64)
    raw_corrections = np.zeros((len(stems), 2), dtype=np.float64)
    raw_errors = np.full(len(stems), np.nan, dtype=np.float64)
    reliable = np.ones(len(stems), dtype=bool)
    selected_left_points = np.full((len(stems), 2), np.nan, dtype=np.float64)
    selected_right_points = np.full((len(stems), 2), np.nan, dtype=np.float64)
    measurement_sources: list[str] = []

    final_selections: list[PairSelection | None] = []
    for index, raw_selection in enumerate(selections):
        fused_selection = fused_selections[index]
        selection = raw_selection
        source = "raw"
        if fused_selection is not None and (
            raw_selection is None
            or fused_selection.error_to_target + 12.0 < raw_selection.error_to_target
            or raw_selection.error_to_target > args.max_raw_pair_error
        ):
            selection = fused_selection
            source = "fused"
        final_selections.append(selection)
        measurement_sources.append(source if selection is not None else "none")

        if selection is None:
            raw_measured[index] = np.array([np.nan, np.nan], dtype=np.float64)
            raw_corrections[index] = np.array([0.0, 0.0], dtype=np.float64)
            reliable[index] = False
            continue
        raw_measured[index] = selection.measured_vec
        raw_corrections[index] = target_vectors[index] - selection.measured_vec
        raw_errors[index] = selection.error_to_target
        if source == "fused":
            selected_left_points[index] = all_fused[index][selection.left_rank].canvas_point
            selected_right_points[index] = all_fused[index][selection.right_rank].canvas_point
        else:
            selected_left_points[index] = all_left[index][selection.left_rank].canvas_point
            selected_right_points[index] = all_right[index][selection.right_rank].canvas_point
        if (
            selection.error_to_target > args.max_raw_pair_error
            or float(np.linalg.norm(raw_corrections[index])) > args.max_correction_norm
        ):
            reliable[index] = False

    norm_reliable = reliable.copy()
    reliable = mark_temporal_outliers(
        raw_corrections,
        reliable,
        radius=args.temporal_radius,
        max_deviation=args.max_temporal_deviation,
        iterations=args.temporal_iterations,
    )
    interpolated_count = int((~reliable).sum())
    correction_vectors = interpolate_vectors(raw_corrections, reliable)
    if args.smooth_window > 1:
        correction_vectors = smooth_vectors(correction_vectors, args.smooth_window)

    output_npz: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    rows_by_stem: dict[str, dict[str, Any]] = {}

    print("Writing refined Hess-matched frames...")
    for index, stem in enumerate(stems, start=1):
        left_path = left_frames[stem]
        right_path = right_frames[stem]
        base_path = args.base_stitched_dir / f"{stem}.jpg"
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        base_image = read_image(base_path)
        out_h, out_w = base_image.shape[:2]
        base_left_h = matrix_npz[f"{stem}_canvas_left_h"].astype(np.float64)
        base_right_h = matrix_npz[f"{stem}_canvas_right_h"].astype(np.float64)
        correction = correction_vectors[index - 1]
        left_shift = 0.5 * correction
        right_shift = -0.5 * correction
        corrected_left_h = translation_matrix(float(left_shift[0]), float(left_shift[1])) @ base_left_h
        corrected_right_h = translation_matrix(float(right_shift[0]), float(right_shift[1])) @ base_right_h

        stitched = blend_warped_pair(
            read_image(left_path),
            read_image(right_path),
            corrected_left_h,
            corrected_right_h,
            output_size=(out_w, out_h),
        )
        output_path = args.output_dir / left_path.name
        write_image(output_path, stitched, args.jpeg_quality)

        final_vec = raw_measured[index - 1] + correction
        final_error = float(np.linalg.norm(final_vec - target_vectors[index - 1]))
        selection = final_selections[index - 1]
        left_rank = -1 if selection is None else selection.left_rank
        right_rank = -1 if selection is None else selection.right_rank

        output_npz[f"{stem}_canvas_left_h"] = corrected_left_h
        output_npz[f"{stem}_canvas_right_h"] = corrected_right_h
        output_npz[f"{stem}_base_canvas_left_h"] = base_left_h
        output_npz[f"{stem}_base_canvas_right_h"] = base_right_h
        output_npz[f"{stem}_left_translation"] = left_shift.astype(np.float64)
        output_npz[f"{stem}_right_translation"] = right_shift.astype(np.float64)
        output_npz[f"{stem}_angle_split_deg"] = angle_splits[index - 1].astype(np.float64)
        output_npz[f"{stem}_target_left_minus_right_vec"] = target_vectors[index - 1].astype(np.float64)
        output_npz[f"{stem}_measured_left_minus_right_vec"] = raw_measured[index - 1].astype(np.float64)
        output_npz[f"{stem}_raw_correction"] = raw_corrections[index - 1].astype(np.float64)
        output_npz[f"{stem}_final_correction"] = correction.astype(np.float64)

        row: dict[str, Any] = {
            "frame": f"{stem}.jpg",
            "stem": stem,
            "frame_index": frame_number(stem),
            "left_path": str(left_path),
            "right_path": str(right_path),
            "base_stitched_path": str(base_path),
            "output_path": str(output_path),
            "output_width": out_w,
            "output_height": out_h,
            "angle_split_dx_deg": float(angle_splits[index - 1, 0]),
            "angle_split_dy_deg": float(angle_splits[index - 1, 1]),
            "target_dx": float(target_vectors[index - 1, 0]),
            "target_dy": float(target_vectors[index - 1, 1]),
            "measured_dx": float(raw_measured[index - 1, 0]),
            "measured_dy": float(raw_measured[index - 1, 1]),
            "raw_correction_dx": float(raw_corrections[index - 1, 0]),
            "raw_correction_dy": float(raw_corrections[index - 1, 1]),
            "correction_dx": float(correction[0]),
            "correction_dy": float(correction[1]),
            "correction_norm": float(np.linalg.norm(correction)),
            "correction_interpolated": int(not reliable[index - 1]),
            "correction_norm_reliable": int(norm_reliable[index - 1]),
            "selected_pair_error_to_target": float(raw_errors[index - 1]),
            "final_error_to_target": final_error,
            "measurement_source": measurement_sources[index - 1],
            "left_translation_x": float(left_shift[0]),
            "left_translation_y": float(left_shift[1]),
            "right_translation_x": float(right_shift[0]),
            "right_translation_y": float(right_shift[1]),
            "selected_left_rank": int(left_rank),
            "selected_right_rank": int(right_rank),
            "selected_left_canvas_x": float(selected_left_points[index - 1, 0]),
            "selected_left_canvas_y": float(selected_left_points[index - 1, 1]),
            "selected_right_canvas_x": float(selected_right_points[index - 1, 0]),
            "selected_right_canvas_y": float(selected_right_points[index - 1, 1]),
            "left_candidate_count": len(all_left[index - 1]),
            "right_candidate_count": len(all_right[index - 1]),
            "fused_candidate_count": len(all_fused[index - 1]),
        }
        row.update(flatten_matrix("canvas_left_h", corrected_left_h))
        row.update(flatten_matrix("canvas_right_h", corrected_right_h))
        rows.append(row)
        rows_by_stem[stem] = row

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(stems)
        ):
            print(
                f"[write {index}/{len(stems)}] {stem}: "
                f"target=({target_vectors[index - 1, 0]:.1f},{target_vectors[index - 1, 1]:.1f}) "
                f"final_err={final_error:.1f}"
            )

    with args.report_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matrix_out = args.matrix_output_dir / f"hess_matched_homographies_{args.case_id}.npz"
    output_npz["angle_splits_deg"] = angle_splits.astype(np.float64)
    output_npz["angle_to_pixel_mapping"] = mapping.astype(np.float64)
    output_npz["mapping_inlier_mask"] = mapping_inliers.astype(np.uint8)
    output_npz["correction_norm_reliable_mask"] = norm_reliable.astype(np.uint8)
    output_npz["correction_reliable_mask"] = reliable.astype(np.uint8)
    np.savez_compressed(matrix_out, **output_npz)

    sample_stems = [
        "frame_0000",
        "frame_0030",
        "frame_0060",
        "frame_0090",
        "frame_0300",
        "frame_0600",
        "frame_0900",
        "frame_1200",
        "frame_1500",
        "frame_1798",
    ]
    debug_sheet = args.debug_dir / f"{args.case_id.lower()}_raw_green_angle_match_samples.jpg"
    make_debug_sheet(debug_sheet, sample_stems, args.output_dir, rows_by_stem, args.jpeg_quality)

    selected_errors = np.array([row["selected_pair_error_to_target"] for row in rows], dtype=float)
    final_errors = np.array([row["final_error_to_target"] for row in rows], dtype=float)
    correction_norms = np.array([row["correction_norm"] for row in rows], dtype=float)
    metadata = {
        "case_id": args.case_id,
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "base_matrix_npz": str(args.base_matrix_npz),
        "base_stitched_dir": str(args.base_stitched_dir),
        "strabismus_mat": str(args.strabismus_mat),
        "mat_key": args.mat_key,
        "output_dir": str(args.output_dir),
        "matrix_output_npz": str(matrix_out),
        "report_csv": str(args.report_csv),
        "debug_sheet": str(debug_sheet),
        "frame_count": len(rows),
        "angle_metadata": angle_metadata,
        "angle_to_pixel_mapping": mapping.tolist(),
        "mapping_inlier_count": int(mapping_inliers.sum()),
        "selected_pair_error_median_px": float(np.nanmedian(selected_errors)),
        "selected_pair_error_p90_px": float(np.nanpercentile(selected_errors, 90)),
        "selected_pair_error_max_px": float(np.nanmax(selected_errors)),
        "final_error_median_px": float(np.nanmedian(final_errors)),
        "final_error_p90_px": float(np.nanpercentile(final_errors, 90)),
        "final_error_max_px": float(np.nanmax(final_errors)),
        "correction_median_px": float(np.nanmedian(correction_norms)),
        "correction_p90_px": float(np.nanpercentile(correction_norms, 90)),
        "correction_max_px": float(np.nanmax(correction_norms)),
        "interpolated_correction_count": interpolated_count,
        "method": (
            "Raw left/right green-marker detections projected through repaired "
            "health_second canvas homographies; split translations force the "
            "left-minus-right image-space marker vector to follow the fitted "
            "SGazeAngleSmoothed_cell trajectory."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote refined frames to {args.output_dir}")
    print(f"Matrix output: {matrix_out}")
    print(f"Debug sheet: {debug_sheet}")


if __name__ == "__main__":
    main()
