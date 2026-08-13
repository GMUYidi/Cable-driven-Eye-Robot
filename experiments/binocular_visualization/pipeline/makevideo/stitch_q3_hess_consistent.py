#!/usr/bin/env python3
"""
Restitch Q3 frames with a Hess-consistent left/right green-dot offset.

The previous Q3 stitching reused the health_second homographies directly. This
script keeps those matrices as the base scene alignment, but estimates the
left-minus-right green-dot vector expected from the Q3 Hess-derived simulated
trajectory and applies a small split translation to the two canvas homographies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Candidate:
    x: float
    y: float
    score: float
    area: float
    gray_mean: float
    green_excess: float
    circularity: float
    aspect: float
    rank: int
    canvas_x: float = float("nan")
    canvas_y: float = float("nan")

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


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def list_frames(input_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def matrix_frame_names(npz: np.lib.npyio.NpzFile) -> list[str]:
    names = [key[: -len("_canvas_left_h")] for key in npz.files if key.endswith("_canvas_left_h")]
    return sorted(names)


def frame_number(stem: str) -> int:
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric frame index from {stem}") from exc


def project_point(h: np.ndarray, x: float, y: float) -> np.ndarray:
    p = h @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-12:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def detect_green_candidates(image: np.ndarray, max_candidates: int) -> list[Candidate]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 30, 35], dtype=np.uint8),
        np.array([95, 240, 255], dtype=np.uint8),
    )

    blue, green, red = cv2.split(image)
    green_excess = green.astype(np.int16) - (
        (red.astype(np.int16) + blue.astype(np.int16)) // 2
    )
    excess_mask = ((green_excess > 6) & (green > 45)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, excess_mask)

    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    candidates: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 15.0 or area > 5000.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.12:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-12:
            continue
        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        if x < 5 or y < 5 or x > width - 5 or y > height - 5:
            continue

        radius = 16
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        patch = image[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]
        patch_b, patch_g, patch_r = cv2.split(patch)
        patch_excess = float(
            (
                patch_g.astype(np.float32)
                - 0.5 * (patch_r.astype(np.float32) + patch_b.astype(np.float32))
            ).mean()
        )
        gray_mean = float(patch_gray.mean())

        # Keep bright low-contrast dots and saturated green dots, while dropping
        # dark edge artifacts from the chessboard/frame border.
        if not (gray_mean >= 70.0 or patch_excess >= 14.0):
            continue
        if patch_excess < 4.0:
            continue

        _, _, box_w, box_h = cv2.boundingRect(contour)
        aspect = max(box_w / box_h, box_h / box_w) if box_w and box_h else 999.0
        if aspect > 4.5:
            continue

        score = (
            area
            * (0.6 + min(circularity, 1.0))
            * max(patch_excess, 1.0)
            * (0.5 + min(gray_mean / 150.0, 1.0))
            / (1.0 + 0.08 * max(0.0, aspect - 1.0))
        )
        candidates.append(
            Candidate(
                x=x,
                y=y,
                score=float(score),
                area=area,
                gray_mean=gray_mean,
                green_excess=patch_excess,
                circularity=float(circularity),
                aspect=float(aspect),
                rank=-1,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    candidates = candidates[:max_candidates]
    for rank, candidate in enumerate(candidates):
        candidate.rank = rank
    return candidates


def load_hess_residuals(mat_path: Path, stems: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    mat = sio.loadmat(mat_path)
    required = ("SGazePointSmoothed_cell", "GazePointSmoothed_cell")
    for key in required:
        if key not in mat:
            raise KeyError(f"Missing {key} in {mat_path}")

    strabismus_points = np.asarray(mat["SGazePointSmoothed_cell"], dtype=np.float64)
    healthy_points = np.asarray(mat["GazePointSmoothed_cell"], dtype=np.float64)
    frame_indices = np.array([frame_number(stem) for stem in stems], dtype=np.int64)
    if frame_indices.max(initial=0) >= len(strabismus_points):
        raise ValueError(
            f"Hess data has {len(strabismus_points)} rows, but max frame index is "
            f"{int(frame_indices.max())}."
        )

    s_lr = strabismus_points[frame_indices, 0:2] - strabismus_points[frame_indices, 2:4]
    h_lr = healthy_points[frame_indices, 0:2] - healthy_points[frame_indices, 2:4]
    residual = s_lr - h_lr
    metadata = {
        "hess_data_rows": int(len(strabismus_points)),
        "frame_index_min": int(frame_indices.min(initial=0)),
        "frame_index_max": int(frame_indices.max(initial=0)),
        "residual_min_xy": residual.min(axis=0).tolist(),
        "residual_max_xy": residual.max(axis=0).tolist(),
    }
    return residual, metadata


def fit_linear_map(
    hess_residuals: np.ndarray,
    measured_vectors: np.ndarray,
    min_inliers: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = (
        np.isfinite(hess_residuals).all(axis=1)
        & np.isfinite(measured_vectors).all(axis=1)
        & (np.linalg.norm(hess_residuals, axis=1) > 1e-9)
    )
    active = finite.copy()
    mapping: np.ndarray | None = None

    for _ in range(10):
        if int(active.sum()) < min_inliers:
            break
        coeff, *_ = np.linalg.lstsq(hess_residuals[active], measured_vectors[active], rcond=None)
        predicted = hess_residuals @ coeff
        errors = np.linalg.norm(measured_vectors - predicted, axis=1)
        active_errors = errors[active]
        median = float(np.median(active_errors))
        mad = float(np.median(np.abs(active_errors - median)) + 1e-6)
        threshold = max(50.0, median + 2.8 * 1.4826 * mad)
        new_active = finite & (errors < threshold)
        mapping = coeff.T
        if np.array_equal(new_active, active):
            break
        active = new_active

    if mapping is None:
        coeff, *_ = np.linalg.lstsq(hess_residuals[finite], measured_vectors[finite], rcond=None)
        mapping = coeff.T

    final_errors = np.linalg.norm(measured_vectors - (hess_residuals @ mapping.T), axis=1)
    return mapping, active, final_errors


def choose_pairs(
    all_left: list[list[Candidate]],
    all_right: list[list[Candidate]],
    target_vectors: np.ndarray,
    rank_penalty: float,
) -> list[PairSelection | None]:
    selections: list[PairSelection | None] = []
    for left_candidates, right_candidates, target in zip(all_left, all_right, target_vectors):
        best: PairSelection | None = None
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            selections.append(None)
            continue

        for left in left_candidates:
            for right in right_candidates:
                measured = left.canvas_point - right.canvas_point
                if not np.isfinite(measured).all():
                    continue
                error = float(np.linalg.norm(measured - target))
                cost = error + rank_penalty * float(left.rank + right.rank)
                selection = PairSelection(
                    left_rank=left.rank,
                    right_rank=right.rank,
                    measured_vec=measured,
                    error_to_target=error,
                    cost=cost,
                )
                if best is None or selection.cost < best.cost:
                    best = selection
        selections.append(best)
    return selections


def interpolate_unreliable_corrections(
    corrections: np.ndarray,
    reliable_mask: np.ndarray,
) -> np.ndarray:
    corrected = corrections.astype(np.float64).copy()
    indices = np.arange(len(corrections), dtype=np.float64)
    reliable_indices = indices[reliable_mask]
    if reliable_indices.size == 0:
        return np.zeros_like(corrected)

    for dim in range(2):
        corrected[:, dim] = np.interp(
            indices,
            reliable_indices,
            corrections[reliable_mask, dim],
        )
    return corrected


def mark_temporal_correction_outliers(
    corrections: np.ndarray,
    reliable_mask: np.ndarray,
    window_radius: int,
    max_deviation: float,
    iterations: int,
) -> np.ndarray:
    if max_deviation <= 0 or window_radius <= 0 or iterations <= 0:
        return reliable_mask.copy()

    corrected_mask = reliable_mask.copy()
    n_frames = len(corrections)
    for _ in range(iterations):
        new_bad: list[int] = []
        for index in range(n_frames):
            if not corrected_mask[index] or not np.isfinite(corrections[index]).all():
                continue

            start = max(0, index - window_radius)
            stop = min(n_frames, index + window_radius + 1)
            neighbor_indices = [
                neighbor
                for neighbor in range(start, stop)
                if neighbor != index
                and corrected_mask[neighbor]
                and np.isfinite(corrections[neighbor]).all()
            ]
            if len(neighbor_indices) < 3:
                continue

            local_median = np.median(corrections[neighbor_indices], axis=0)
            if float(np.linalg.norm(corrections[index] - local_median)) > max_deviation:
                new_bad.append(index)

        if not new_bad:
            break
        corrected_mask[new_bad] = False
    return corrected_mask


def smooth_correction_vectors(corrections: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return corrections
    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    weights = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.empty_like(corrections, dtype=np.float64)
    for dim in range(corrections.shape[1]):
        padded = np.pad(corrections[:, dim], (pad, pad), mode="edge")
        smoothed[:, dim] = np.convolve(padded, weights, mode="valid")
    return smoothed


def blend_warped_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    canvas_left_h: np.ndarray,
    canvas_right_h: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    out_w, out_h = output_size
    warped_l = cv2.warpPerspective(left_image, canvas_left_h, (out_w, out_h))
    warped_r = cv2.warpPerspective(right_image, canvas_right_h, (out_w, out_h))

    mask_src_l = np.ones(left_image.shape[:2], dtype=np.uint8) * 255
    mask_src_r = np.ones(right_image.shape[:2], dtype=np.uint8) * 255
    mask_l = cv2.warpPerspective(
        mask_src_l, canvas_left_h, (out_w, out_h), flags=cv2.INTER_NEAREST
    )
    mask_r = cv2.warpPerspective(
        mask_src_r, canvas_right_h, (out_w, out_h), flags=cv2.INTER_NEAREST
    )

    dist_l = cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(
        np.float32
    )
    dist_r = cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(
        np.float32
    )
    denom = dist_l + dist_r + 1e-6
    weight_l = dist_l / denom
    weight_r = dist_r / denom

    only_l = (mask_l > 0) & (mask_r == 0)
    only_r = (mask_r > 0) & (mask_l == 0)
    weight_l[only_l] = 1.0
    weight_r[only_l] = 0.0
    weight_l[only_r] = 0.0
    weight_r[only_r] = 1.0

    blended = (
        warped_l.astype(np.float32) * weight_l[..., None]
        + warped_r.astype(np.float32) * weight_r[..., None]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def flatten_matrix(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_{row + 1}{col + 1}": float(matrix[row, col])
        for row in range(3)
        for col in range(3)
    }


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def pad_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas


def annotate_pair(
    image: np.ndarray,
    title: str,
    left_point: np.ndarray | None,
    right_point: np.ndarray | None,
    vector_text: str,
) -> np.ndarray:
    annotated = image.copy()
    if (
        left_point is not None
        and right_point is not None
        and np.all(np.isfinite(left_point))
        and np.all(np.isfinite(right_point))
    ):
        lp = tuple(np.round(left_point).astype(int))
        rp = tuple(np.round(right_point).astype(int))
        cv2.circle(annotated, lp, 18, (0, 0, 255), 4)
        cv2.circle(annotated, rp, 18, (255, 0, 0), 4)
        cv2.line(annotated, rp, lp, (0, 255, 255), 3)
    cv2.putText(annotated, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 255), 3)
    cv2.putText(
        annotated,
        vector_text,
        (24, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def make_debug_sheet(
    output_path: Path,
    sample_stems: list[str],
    base_stitched_dir: Path,
    output_stitched_dir: Path,
    rows_by_stem: dict[str, dict[str, Any]],
    tile_width: int,
    jpeg_quality: int,
) -> None:
    row_images: list[np.ndarray] = []
    for stem in sample_stems:
        if stem not in rows_by_stem:
            continue
        base_path = base_stitched_dir / f"{stem}.jpg"
        new_path = output_stitched_dir / f"{stem}.jpg"
        if not base_path.exists() or not new_path.exists():
            continue

        row = rows_by_stem[stem]
        old_left = np.array([row["selected_left_x"], row["selected_left_y"]], dtype=np.float64)
        old_right = np.array([row["selected_right_x"], row["selected_right_y"]], dtype=np.float64)
        left_shift = np.array([row["left_translation_x"], row["left_translation_y"]], dtype=np.float64)
        right_shift = np.array([row["right_translation_x"], row["right_translation_y"]], dtype=np.float64)
        new_left = old_left + left_shift
        new_right = old_right + right_shift

        base_img = annotate_pair(
            read_image(base_path),
            f"{stem} old",
            old_left,
            old_right,
            f"old=({row['measured_dx']:.1f},{row['measured_dy']:.1f})",
        )
        new_img = annotate_pair(
            read_image(new_path),
            f"{stem} Hess matched",
            new_left,
            new_right,
            f"target=({row['target_dx']:.1f},{row['target_dy']:.1f})",
        )
        base_small = resize_to_width(base_img, tile_width)
        new_small = resize_to_width(new_img, tile_width)
        height = max(base_small.shape[0], new_small.shape[0])
        row_images.append(
            np.hstack(
                [
                    pad_to(base_small, tile_width, height),
                    pad_to(new_small, tile_width, height),
                ]
            )
        )

    if not row_images:
        return

    width = max(row.shape[1] for row in row_images)
    padded_rows = [pad_to(row, width, row.shape[0]) for row in row_images]
    sheet = np.vstack(padded_rows)
    write_image(output_path, sheet, jpeg_quality=jpeg_quality)


def build_arg_parser() -> argparse.ArgumentParser:
    output_root = Path(r"D:\visualization2026\dowsample2_Q3_hess_matched")
    parser = argparse.ArgumentParser(
        description="Restitch Q3 frames so left/right green-dot offsets follow Q3 Hess data."
    )
    parser.add_argument(
        "--left-dir",
        type=Path,
        default=Path(r"D:\visualization2026\visualization\left_downsample2_Q3"),
    )
    parser.add_argument(
        "--right-dir",
        type=Path,
        default=Path(r"D:\visualization2026\visualization\right_downsample2_Q3"),
    )
    parser.add_argument(
        "--base-matrix-npz",
        type=Path,
        default=Path(
            r"D:\visualization2026\dowsample2_Q3\matrices"
            r"\applied_health_second_homographies_Q3.npz"
        ),
    )
    parser.add_argument(
        "--base-stitched-dir",
        type=Path,
        default=Path(r"D:\visualization2026\dowsample2_Q3\stitched_frames"),
    )
    parser.add_argument(
        "--strabismus-mat",
        type=Path,
        default=Path(
            r"D:\visualization2026\Hess chart\bilinear interpolation\strabismus data"
            r"\simulated_strabismus_Q3.mat"
        ),
    )
    parser.add_argument(
        "--hess-chart-image",
        type=Path,
        default=Path(r"D:\visualization2026\Hess chart\small bias data\Q3 Pre-OpHess test 12-1-15.jpg"),
    )
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--output-dir", type=Path, default=output_root / "stitched_frames")
    parser.add_argument("--matrix-output-dir", type=Path, default=output_root / "matrices")
    parser.add_argument("--report-csv", type=Path, default=output_root / "hess_match_report.csv")
    parser.add_argument("--metadata-json", type=Path, default=output_root / "hess_match_metadata.json")
    parser.add_argument("--debug-dir", type=Path, default=output_root / "debug")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--case-id", default="Q3")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--rank-penalty", type=float, default=8.0)
    parser.add_argument("--refit-iterations", type=int, default=2)
    parser.add_argument(
        "--max-correction-norm",
        type=float,
        default=0.0,
        help=(
            "If positive, corrections larger than this many pixels are treated "
            "as unreliable candidate matches and replaced by temporal interpolation."
        ),
    )
    parser.add_argument(
        "--max-correction-temporal-deviation",
        type=float,
        default=0.0,
        help=(
            "If positive, a correction that differs from the local temporal median "
            "by more than this many pixels is treated as an unreliable candidate match."
        ),
    )
    parser.add_argument(
        "--temporal-outlier-radius",
        type=int,
        default=5,
        help="Neighbor radius used by --max-correction-temporal-deviation.",
    )
    parser.add_argument(
        "--temporal-outlier-iterations",
        type=int,
        default=4,
        help="Number of iterative passes for temporal correction outlier rejection.",
    )
    parser.add_argument(
        "--smooth-correction-window",
        type=int,
        default=1,
        help="Odd moving-average window applied to correction vectors after interpolation.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    for folder in (args.left_dir, args.right_dir, args.base_stitched_dir):
        if not folder.exists():
            raise FileNotFoundError(f"Required directory does not exist: {folder}")
    for file_path in (args.base_matrix_npz, args.strabismus_mat):
        if not file_path.exists():
            raise FileNotFoundError(f"Required file does not exist: {file_path}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {args.output_dir}\n"
            "Use --overwrite or choose a different --output-dir."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matrix_output_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    matrix_npz = np.load(args.base_matrix_npz, allow_pickle=True)
    stems = matrix_frame_names(matrix_npz)
    hess_residuals, hess_metadata = load_hess_residuals(args.strabismus_mat, stems)

    all_left_candidates: list[list[Candidate]] = []
    all_right_candidates: list[list[Candidate]] = []
    top_vectors: list[np.ndarray] = []
    skipped: list[str] = []

    print("Detecting green-dot candidates and projecting them to canvas coordinates...")
    for index, stem in enumerate(stems, start=1):
        left_path = left_frames.get(stem)
        right_path = right_frames.get(stem)
        if left_path is None or right_path is None:
            skipped.append(stem)
            all_left_candidates.append([])
            all_right_candidates.append([])
            top_vectors.append(np.array([np.nan, np.nan], dtype=np.float64))
            continue

        left_image = read_image(left_path)
        right_image = read_image(right_path)
        left_candidates = detect_green_candidates(left_image, args.max_candidates)
        right_candidates = detect_green_candidates(right_image, args.max_candidates)
        left_h = matrix_npz[f"{stem}_canvas_left_h"].astype(np.float64)
        right_h = matrix_npz[f"{stem}_canvas_right_h"].astype(np.float64)

        for candidate in left_candidates:
            candidate.canvas_x, candidate.canvas_y = project_point(left_h, candidate.x, candidate.y)
        for candidate in right_candidates:
            candidate.canvas_x, candidate.canvas_y = project_point(right_h, candidate.x, candidate.y)

        all_left_candidates.append(left_candidates)
        all_right_candidates.append(right_candidates)
        if left_candidates and right_candidates:
            top_vectors.append(left_candidates[0].canvas_point - right_candidates[0].canvas_point)
        else:
            top_vectors.append(np.array([np.nan, np.nan], dtype=np.float64))

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(stems)
        ):
            print(
                f"[detect {index}/{len(stems)}] {stem}: "
                f"L={len(left_candidates)} R={len(right_candidates)}"
            )

    if skipped:
        skipped_preview = ", ".join(skipped[:10])
        raise RuntimeError(f"Missing raw inputs for {len(skipped)} frames: {skipped_preview}")

    measured_top = np.vstack(top_vectors)
    mapping, inlier_mask, mapping_errors = fit_linear_map(hess_residuals, measured_top)
    target_vectors = hess_residuals @ mapping.T

    selections: list[PairSelection | None] = []
    for refit_index in range(args.refit_iterations):
        selections = choose_pairs(
            all_left_candidates,
            all_right_candidates,
            target_vectors,
            rank_penalty=args.rank_penalty,
        )
        selected_vectors = np.vstack(
            [
                selection.measured_vec if selection is not None else np.array([np.nan, np.nan])
                for selection in selections
            ]
        )
        mapping, inlier_mask, mapping_errors = fit_linear_map(hess_residuals, selected_vectors)
        target_vectors = hess_residuals @ mapping.T
        print(
            f"Refit {refit_index + 1}/{args.refit_iterations}: "
            f"inliers={int(inlier_mask.sum())}, "
            f"median_error={float(np.nanmedian(mapping_errors[inlier_mask])):.2f}px"
        )

    selections = choose_pairs(
        all_left_candidates,
        all_right_candidates,
        target_vectors,
        rank_penalty=args.rank_penalty,
    )

    raw_corrections: list[np.ndarray] = []
    raw_measured_vectors: list[np.ndarray] = []
    raw_pair_errors: list[float] = []
    raw_reliable = np.ones(len(stems), dtype=bool)
    for selection_index, selection in enumerate(selections):
        if selection is None:
            raw_measured_vectors.append(np.array([np.nan, np.nan], dtype=np.float64))
            raw_corrections.append(np.array([0.0, 0.0], dtype=np.float64))
            raw_pair_errors.append(float("nan"))
            raw_reliable[selection_index] = False
            continue

        raw_measured_vectors.append(selection.measured_vec)
        raw_correction = target_vectors[selection_index] - selection.measured_vec
        raw_corrections.append(raw_correction)
        raw_pair_errors.append(selection.error_to_target)
        if not np.isfinite(raw_correction).all():
            raw_reliable[selection_index] = False
        elif (
            args.max_correction_norm > 0
            and np.linalg.norm(raw_correction) > args.max_correction_norm
        ):
            raw_reliable[selection_index] = False

    correction_vectors = np.vstack(raw_corrections)
    norm_reliable = raw_reliable.copy()
    temporal_reliable = mark_temporal_correction_outliers(
        correction_vectors,
        raw_reliable,
        window_radius=args.temporal_outlier_radius,
        max_deviation=args.max_correction_temporal_deviation,
        iterations=args.temporal_outlier_iterations,
    )
    temporal_outlier_count = int(raw_reliable.sum() - temporal_reliable.sum())
    raw_reliable = temporal_reliable
    if temporal_outlier_count > 0:
        print(
            f"Marked {temporal_outlier_count} temporal correction outliers "
            f"(max_temporal_deviation={args.max_correction_temporal_deviation:.1f})."
        )

    if not raw_reliable.all():
        correction_vectors = interpolate_unreliable_corrections(
            correction_vectors,
            raw_reliable,
        )
        print(
            f"Interpolated {int((~raw_reliable).sum())} unreliable correction vectors "
            f"(max_correction_norm={args.max_correction_norm:.1f})."
        )
    if args.smooth_correction_window > 1:
        correction_vectors = smooth_correction_vectors(
            correction_vectors,
            args.smooth_correction_window,
        )
        print(f"Smoothed correction vectors with window={args.smooth_correction_window}.")

    output_npz: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    rows_by_stem: dict[str, dict[str, Any]] = {}

    print("Writing Hess-matched stitched frames...")
    for index, stem in enumerate(stems, start=1):
        left_path = left_frames[stem]
        right_path = right_frames[stem]
        base_stitched_path = args.base_stitched_dir / f"{stem}.jpg"
        if not base_stitched_path.exists():
            raise FileNotFoundError(f"Missing base stitched frame for canvas size: {base_stitched_path}")

        left_image = read_image(left_path)
        right_image = read_image(right_path)
        base_stitched = read_image(base_stitched_path)
        out_h, out_w = base_stitched.shape[:2]

        base_left_h = matrix_npz[f"{stem}_canvas_left_h"].astype(np.float64)
        base_right_h = matrix_npz[f"{stem}_canvas_right_h"].astype(np.float64)
        target_vec = target_vectors[index - 1]
        selection = selections[index - 1]
        if selection is None:
            measured_vec = raw_measured_vectors[index - 1]
            correction = correction_vectors[index - 1]
            raw_correction = raw_corrections[index - 1]
            left_rank = -1
            right_rank = -1
            error_to_target = raw_pair_errors[index - 1]
            selected_left_point = np.array([np.nan, np.nan], dtype=np.float64)
            selected_right_point = np.array([np.nan, np.nan], dtype=np.float64)
        else:
            measured_vec = raw_measured_vectors[index - 1]
            correction = correction_vectors[index - 1]
            raw_correction = raw_corrections[index - 1]
            left_rank = selection.left_rank
            right_rank = selection.right_rank
            error_to_target = raw_pair_errors[index - 1]
            selected_left_point = all_left_candidates[index - 1][left_rank].canvas_point
            selected_right_point = all_right_candidates[index - 1][right_rank].canvas_point

        left_shift = 0.5 * correction
        right_shift = -0.5 * correction
        corrected_left_h = translation_matrix(float(left_shift[0]), float(left_shift[1])) @ base_left_h
        corrected_right_h = translation_matrix(float(right_shift[0]), float(right_shift[1])) @ base_right_h

        stitched = blend_warped_pair(
            left_image,
            right_image,
            corrected_left_h,
            corrected_right_h,
            output_size=(out_w, out_h),
        )
        output_path = args.output_dir / left_path.name
        write_image(output_path, stitched, jpeg_quality=args.jpeg_quality)

        output_npz[f"{stem}_canvas_left_h"] = corrected_left_h
        output_npz[f"{stem}_canvas_right_h"] = corrected_right_h
        output_npz[f"{stem}_base_canvas_left_h"] = base_left_h
        output_npz[f"{stem}_base_canvas_right_h"] = base_right_h
        output_npz[f"{stem}_left_translation"] = left_shift.astype(np.float64)
        output_npz[f"{stem}_right_translation"] = right_shift.astype(np.float64)
        output_npz[f"{stem}_target_left_minus_right_vec"] = target_vec.astype(np.float64)
        output_npz[f"{stem}_measured_left_minus_right_vec"] = measured_vec.astype(np.float64)
        output_npz[f"{stem}_raw_correction"] = raw_correction.astype(np.float64)
        output_npz[f"{stem}_final_correction"] = correction.astype(np.float64)

        row: dict[str, Any] = {
            "frame": f"{stem}.jpg",
            "stem": stem,
            "frame_index": frame_number(stem),
            "left_path": str(left_path),
            "right_path": str(right_path),
            "base_stitched_path": str(base_stitched_path),
            "output_path": str(output_path),
            "output_width": out_w,
            "output_height": out_h,
            "hess_residual_x": float(hess_residuals[index - 1, 0]),
            "hess_residual_y": float(hess_residuals[index - 1, 1]),
            "target_dx": float(target_vec[0]),
            "target_dy": float(target_vec[1]),
            "measured_dx": float(measured_vec[0]),
            "measured_dy": float(measured_vec[1]),
            "raw_correction_dx": float(raw_correction[0]),
            "raw_correction_dy": float(raw_correction[1]),
            "raw_correction_norm": float(np.linalg.norm(raw_correction)),
            "correction_dx": float(correction[0]),
            "correction_dy": float(correction[1]),
            "correction_norm": float(np.linalg.norm(correction)),
            "correction_interpolated": int(not raw_reliable[index - 1]),
            "correction_norm_reliable": int(norm_reliable[index - 1]),
            "left_translation_x": float(left_shift[0]),
            "left_translation_y": float(left_shift[1]),
            "right_translation_x": float(right_shift[0]),
            "right_translation_y": float(right_shift[1]),
            "selected_left_rank": int(left_rank),
            "selected_right_rank": int(right_rank),
            "selected_pair_error_to_target": float(error_to_target),
            "selected_left_x": float(selected_left_point[0]),
            "selected_left_y": float(selected_left_point[1]),
            "selected_right_x": float(selected_right_point[0]),
            "selected_right_y": float(selected_right_point[1]),
            "left_candidate_count": len(all_left_candidates[index - 1]),
            "right_candidate_count": len(all_right_candidates[index - 1]),
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
                f"target=({target_vec[0]:.1f},{target_vec[1]:.1f}) "
                f"correction=({correction[0]:.1f},{correction[1]:.1f})"
            )

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matrix_out = args.matrix_output_dir / f"hess_matched_homographies_{args.case_id}.npz"
    output_npz["hess_residuals"] = hess_residuals.astype(np.float64)
    output_npz["hess_to_pixel_mapping"] = mapping.astype(np.float64)
    output_npz["mapping_inlier_mask"] = inlier_mask.astype(np.uint8)
    output_npz["correction_norm_reliable_mask"] = norm_reliable.astype(np.uint8)
    output_npz["correction_reliable_mask"] = raw_reliable.astype(np.uint8)
    np.savez_compressed(matrix_out, **output_npz)

    sample_stems = [
        "frame_0000",
        "frame_0300",
        "frame_0600",
        "frame_0900",
        "frame_1034",
        "frame_1200",
        "frame_1500",
        "frame_1798",
    ]
    debug_sheet = args.debug_dir / f"{args.case_id.lower()}_hess_match_samples.jpg"
    make_debug_sheet(
        debug_sheet,
        sample_stems,
        args.base_stitched_dir,
        args.output_dir,
        rows_by_stem,
        tile_width=520,
        jpeg_quality=args.jpeg_quality,
    )

    selected_errors = np.array([row["selected_pair_error_to_target"] for row in rows], dtype=float)
    correction_norms = np.array([row["correction_norm"] for row in rows], dtype=float)
    raw_correction_norms = np.array([row["raw_correction_norm"] for row in rows], dtype=float)
    metadata = {
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "base_matrix_npz": str(args.base_matrix_npz),
        "base_stitched_dir": str(args.base_stitched_dir),
        "strabismus_mat": str(args.strabismus_mat),
        "hess_chart_image": str(args.hess_chart_image),
        "output_dir": str(args.output_dir),
        "matrix_output_npz": str(matrix_out),
        "report_csv": str(args.report_csv),
        "debug_sheet": str(debug_sheet),
        "frame_count": len(rows),
        "jpeg_quality": args.jpeg_quality,
        "hess_metadata": hess_metadata,
        "hess_to_pixel_mapping": mapping.tolist(),
        "mapping_inlier_count": int(inlier_mask.sum()),
        "selected_pair_error_median_px": float(np.nanmedian(selected_errors)),
        "selected_pair_error_p90_px": float(np.nanpercentile(selected_errors, 90)),
        "selected_pair_error_max_px": float(np.nanmax(selected_errors)),
        "raw_correction_median_px": float(np.nanmedian(raw_correction_norms)),
        "raw_correction_p90_px": float(np.nanpercentile(raw_correction_norms, 90)),
        "raw_correction_max_px": float(np.nanmax(raw_correction_norms)),
        "correction_median_px": float(np.nanmedian(correction_norms)),
        "correction_p90_px": float(np.nanpercentile(correction_norms, 90)),
        "correction_max_px": float(np.nanmax(correction_norms)),
        "max_correction_norm": args.max_correction_norm,
        "max_correction_temporal_deviation": args.max_correction_temporal_deviation,
        "temporal_outlier_radius": args.temporal_outlier_radius,
        "temporal_outlier_iterations": args.temporal_outlier_iterations,
        "temporal_outlier_count": temporal_outlier_count,
        "smooth_correction_window": args.smooth_correction_window,
        "interpolated_correction_count": int((~raw_reliable).sum()),
        "method": (
            "Base health_second canvas homographies plus split per-frame translations. "
            f"Target vectors are fitted from {args.case_id} Hess-derived simulated gaze residuals "
            "using frame-number indexing."
        ),
    }
    with args.metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Wrote {len(rows)} frames to {args.output_dir}")
    print(f"Matrices: {matrix_out}")
    print(f"Report: {args.report_csv}")
    print(f"Debug sheet: {debug_sheet}")


if __name__ == "__main__":
    main()
