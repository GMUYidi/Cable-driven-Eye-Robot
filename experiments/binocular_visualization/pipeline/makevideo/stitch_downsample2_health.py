"""
Robust stereo frame stitching for downsample2_health.

The old notebooks use a full 6x8 inner-corner chessboard and estimate a
homography directly from left corners to right corners. This script keeps that
idea, but it also searches for visible sub-boards when the full board is partly
outside the image. Candidate matches are checked against the previous accepted
homography and by a light edge-overlap score so one-row/one-column chessboard
offsets are less likely to slip through.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_LEFT_DIR = Path(r"D:\visualization2026\visualization\left_downsample2_health")
DEFAULT_RIGHT_DIR = Path(r"D:\visualization2026\visualization\right_downsample2_health")
DEFAULT_OUTPUT_DIR = Path(r"D:\visualization2026\dowsample2_health")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class BoardDetection:
    pattern: tuple[int, int]
    points: np.ndarray
    method: str
    quality: float


@dataclass
class MatchCandidate:
    left_points: np.ndarray
    right_points: np.ndarray
    common_pattern: tuple[int, int]
    left_window: tuple[int, int]
    right_window: tuple[int, int]
    right_orientation: str
    source: str
    homography: np.ndarray
    inliers: int
    reprojection_error: float
    prior_error: float
    edge_error: float
    green_error: float
    twist_angle_deg: float
    local_det: float
    score: float


@dataclass
class ProcessResult:
    status: str
    source: str
    left_h: np.ndarray
    right_h: np.ndarray
    direct_h: np.ndarray
    canvas_left_h: np.ndarray
    canvas_right_h: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    left_pattern: tuple[int, int] | None
    right_pattern: tuple[int, int] | None
    common_pattern: tuple[int, int] | None
    inliers: int
    reprojection_error: float
    prior_error: float
    edge_error: float
    green_error: float
    twist_angle_deg: float
    local_det: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch left/right downsample2_health frames and save homographies."
    )
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_RIGHT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern-cols", type=int, default=6, help="Inner chessboard columns.")
    parser.add_argument("--pattern-rows", type=int, default=8, help="Inner chessboard rows.")
    parser.add_argument("--min-pattern-cols", type=int, default=3)
    parser.add_argument("--min-pattern-rows", type=int, default=3)
    parser.add_argument("--detect-scale", type=float, default=0.5)
    parser.add_argument("--edge-scale", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all matched frames.")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stitch-plane", choices=("right", "middle"), default="right")
    parser.add_argument("--max-board-candidates", type=int, default=4000)
    parser.add_argument("--max-edge-candidates", type=int, default=36)
    parser.add_argument("--max-board-prior-error", type=float, default=85.0)
    parser.add_argument("--max-sift-prior-error", type=float, default=45.0)
    parser.add_argument("--max-reprojection-error", type=float, default=8.0)
    parser.add_argument("--max-edge-error", type=float, default=0.72)
    parser.add_argument(
        "--max-green-error",
        type=float,
        default=20.0,
        help="Reject homographies whose main green marker centers differ by more than this many pixels.",
    )
    parser.add_argument(
        "--green-score-weight",
        type=float,
        default=0.08,
        help="Candidate-score penalty per pixel of green marker misalignment.",
    )
    parser.add_argument("--max-twist-degrees", type=float, default=15.0)
    parser.add_argument("--min-local-det", type=float, default=0.05)
    parser.add_argument("--sift-features", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-debug-matches", action="store_true")
    parser.add_argument("--debug-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def frame_number(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=numeric_key,
    )


def paired_frames(left_dir: Path, right_dir: Path) -> list[tuple[Path, Path]]:
    left = {p.name: p for p in list_images(left_dir)}
    right = {p.name: p for p in list_images(right_dir)}
    names = sorted(set(left).intersection(right), key=lambda name: numeric_key(Path(name)))
    return [(left[name], right[name]) for name in names]


def make_pattern_order(
    full_pattern: tuple[int, int],
    min_pattern: tuple[int, int],
    preferred: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    full_cols, full_rows = full_pattern
    min_cols, min_rows = min_pattern
    patterns = [
        (cols, rows)
        for cols in range(min_cols, full_cols + 1)
        for rows in range(min_rows, full_rows + 1)
    ]
    patterns.sort(key=lambda p: (p[0] * p[1], min(p[0], p[1])), reverse=True)
    ordered: list[tuple[int, int]] = []
    for pattern in [full_pattern] + list(preferred) + patterns:
        if (
            min_cols <= pattern[0] <= full_cols
            and min_rows <= pattern[1] <= full_rows
            and pattern not in ordered
        ):
            ordered.append(pattern)
    return ordered


def grid_quality(points: np.ndarray, pattern: tuple[int, int]) -> float:
    cols, rows = pattern
    if points.shape[0] != cols * rows:
        return -1.0
    grid = points.reshape(rows, cols, 2)
    steps: list[float] = []
    if cols > 1:
        steps.extend(np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel().tolist())
    if rows > 1:
        steps.extend(np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel().tolist())
    if not steps:
        return -1.0
    steps_arr = np.asarray(steps, dtype=np.float32)
    median_step = float(np.median(steps_arr))
    if median_step < 4.0:
        return -1.0
    spread = float(np.percentile(steps_arr, 90) / max(np.percentile(steps_arr, 10), 1e-6))
    if spread > 4.5:
        return -1.0
    return median_step / spread


def detect_chessboard(
    image: np.ndarray,
    patterns: list[tuple[int, int]],
    detect_scale: float,
) -> BoardDetection | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not math.isclose(detect_scale, 1.0):
        scaled = cv2.resize(gray, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    else:
        scaled = gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    scaled_eq = clahe.apply(scaled)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    for pattern in patterns:
        flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCornersSB(scaled_eq, pattern, flags=flags_sb)
        method = "findChessboardCornersSB"
        if not ok:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(scaled_eq, pattern, flags=flags)
            method = "findChessboardCorners"
            if ok:
                cv2.cornerSubPix(scaled, corners, (7, 7), (-1, -1), criteria)
        if not ok or corners is None:
            continue
        points = corners.reshape(-1, 2).astype(np.float32) / float(detect_scale)
        quality = grid_quality(points, pattern)
        if quality <= 0:
            continue
        # The pattern order is intentional: full board, last good partial board,
        # then descending partial boards. Returning the first valid hit keeps the
        # partial-board path fast while preserving temporal continuity.
        return BoardDetection(pattern=pattern, points=points, method=method, quality=quality)
    return None


def grid_from_points(points: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    cols, rows = pattern
    return points.reshape(rows, cols, 2)


def orient_grid(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "id":
        return grid
    if orientation == "flip_x":
        return grid[:, ::-1]
    if orientation == "flip_y":
        return grid[::-1, :]
    if orientation == "flip_xy":
        return grid[::-1, ::-1]
    raise ValueError(f"unknown orientation: {orientation}")


def grid_variants(grid: np.ndarray, include_transpose: bool = False) -> list[tuple[str, np.ndarray]]:
    variants = [(name, orient_grid(grid, name)) for name in ("id", "flip_x", "flip_y", "flip_xy")]
    if include_transpose:
        transposed = np.transpose(grid, (1, 0, 2))
        variants.extend(
            [
                (f"transpose_{name}", orient_grid(transposed, name))
                for name in ("id", "flip_x", "flip_y", "flip_xy")
            ]
        )
    return variants


def windows_for_pattern(
    grid: np.ndarray,
    sub_pattern: tuple[int, int],
) -> Iterable[tuple[tuple[int, int], np.ndarray]]:
    sub_cols, sub_rows = sub_pattern
    rows, cols = grid.shape[:2]
    if sub_cols > cols or sub_rows > rows:
        return
    for row0 in range(rows - sub_rows + 1):
        for col0 in range(cols - sub_cols + 1):
            window = grid[row0 : row0 + sub_rows, col0 : col0 + sub_cols]
            yield (col0, row0), window.reshape(-1, 2).astype(np.float32)


def common_patterns(
    left_pattern: tuple[int, int],
    right_pattern: tuple[int, int],
    min_pattern: tuple[int, int],
) -> list[tuple[int, int]]:
    max_cols = min(max(left_pattern), max(right_pattern))
    max_rows = min(max(left_pattern), max(right_pattern))
    min_cols, min_rows = min_pattern
    patterns = [
        (cols, rows)
        for cols in range(min_cols, max_cols + 1)
        for rows in range(min_rows, max_rows + 1)
    ]
    patterns.sort(key=lambda p: (p[0] * p[1], min(p[0], p[1])), reverse=True)
    return patterns


def normalize_h(homography: np.ndarray | None) -> np.ndarray | None:
    if homography is None:
        return None
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(h[2, 2]) > 1e-9:
        h = h / h[2, 2]
    return h


def estimate_homography(
    left_points: np.ndarray,
    right_points: np.ndarray,
    ransac_threshold: float = 4.0,
) -> tuple[np.ndarray | None, int, float]:
    if left_points.shape[0] < 4 or right_points.shape[0] < 4:
        return None, 0, float("inf")
    h, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, ransac_threshold)
    h = normalize_h(h)
    if h is None:
        return None, 0, float("inf")
    projected = cv2.perspectiveTransform(left_points.reshape(-1, 1, 2), h).reshape(-1, 2)
    errors = np.linalg.norm(projected - right_points, axis=1)
    if mask is not None and mask.size == errors.size:
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        reproj = float(np.median(errors[inlier_mask])) if inliers else float(np.median(errors))
    else:
        inliers = left_points.shape[0]
        reproj = float(np.median(errors))
    return h, inliers, reproj


def projection_error(homography: np.ndarray | None, src: np.ndarray, dst: np.ndarray) -> float:
    if homography is None or src.shape[0] == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    return float(np.median(errors))


def homography_jacobian_at(homography: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y = float(point[0]), float(point[1])
    h = homography
    denom = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(denom) < 1e-9:
        return np.full((2, 2), np.nan, dtype=np.float64)
    u_num = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v_num = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    du_dx = (h[0, 0] * denom - u_num * h[2, 0]) / (denom * denom)
    du_dy = (h[0, 1] * denom - u_num * h[2, 1]) / (denom * denom)
    dv_dx = (h[1, 0] * denom - v_num * h[2, 0]) / (denom * denom)
    dv_dy = (h[1, 1] * denom - v_num * h[2, 1]) / (denom * denom)
    return np.array([[du_dx, du_dy], [dv_dx, dv_dy]], dtype=np.float64)


def local_rotation_and_det(homography: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    if points.size == 0:
        center = np.array([0.0, 0.0], dtype=np.float64)
    else:
        center = np.median(points, axis=0).astype(np.float64)
    jac = homography_jacobian_at(homography, center)
    if not np.isfinite(jac).all():
        return float("nan"), float("nan")
    det = float(np.linalg.det(jac))
    angle = math.degrees(math.atan2(jac[1, 0] - jac[0, 1], jac[0, 0] + jac[1, 1]))
    if angle > 180:
        angle -= 360
    if angle < -180:
        angle += 360
    return angle, det


def detect_green_points(image: np.ndarray, max_points: int = 3) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([38, 35, 45]), np.array([85, 220, 230]))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 120.0 < area < 3500.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x < 25 or y < 25 or x + w > width - 25 or y + h > height - 25:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        blobs.append((area, cx, cy))
    blobs.sort(reverse=True)
    return np.array([[cx, cy] for _, cx, cy in blobs[:max_points]], dtype=np.float32)


def green_alignment_error(
    homography: np.ndarray,
    left_green_points: np.ndarray,
    right_green_points: np.ndarray,
) -> float:
    if left_green_points.size == 0 or right_green_points.size == 0:
        return float("nan")
    # The largest circular green blob is the intended registration marker in
    # these frames. Smaller green-ish blobs around borders are usually noise.
    left_main = left_green_points[:1]
    right_main = right_green_points[:1]
    projected = cv2.perspectiveTransform(left_main.reshape(-1, 1, 2), homography).reshape(-1, 2)
    distances = np.linalg.norm(projected - right_main, axis=1)
    return float(np.mean(distances))


def candidate_physical_metrics(
    homography: np.ndarray,
    left_points: np.ndarray,
    left_green_points: np.ndarray,
    right_green_points: np.ndarray,
) -> tuple[float, float, float]:
    twist_angle, local_det = local_rotation_and_det(homography, left_points)
    green_error = green_alignment_error(homography, left_green_points, right_green_points)
    return green_error, twist_angle, local_det


def physical_metrics_ok(
    green_error: float,
    twist_angle: float,
    local_det: float,
    args: argparse.Namespace,
) -> bool:
    if not np.isfinite(twist_angle) or not np.isfinite(local_det):
        return False
    if local_det <= args.min_local_det:
        return False
    if abs(twist_angle) > args.max_twist_degrees:
        return False
    if np.isfinite(green_error) and green_error > args.max_green_error:
        return False
    return True


def scaled_homography(homography: np.ndarray, scale: float) -> np.ndarray:
    s = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return s @ homography @ np.linalg.inv(s)


def edge_overlap_error(
    left_image: np.ndarray,
    right_image: np.ndarray,
    homography: np.ndarray,
    scale: float,
) -> float:
    if scale <= 0:
        return 1.0
    left_small = cv2.resize(left_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
    left_edges = cv2.Canny(left_gray, 60, 160)
    right_edges = cv2.Canny(right_gray, 60, 160)
    h_small = scaled_homography(homography, scale)
    warped_edges = cv2.warpPerspective(
        left_edges,
        h_small,
        (right_edges.shape[1], right_edges.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    warped_mask = cv2.warpPerspective(
        np.ones(left_edges.shape, dtype=np.uint8) * 255,
        h_small,
        (right_edges.shape[1], right_edges.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    support = (warped_edges > 0) & (warped_mask > 0)
    support_count = int(support.sum())
    if support_count < 50:
        return 1.0
    right_dilated = cv2.dilate(right_edges, np.ones((3, 3), np.uint8), iterations=1)
    aligned = support & (right_dilated > 0)
    return 1.0 - float(aligned.sum()) / float(support_count)


def transformed_corners(image_shape: tuple[int, int, int], homography: np.ndarray) -> np.ndarray:
    height, width = image_shape[:2]
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, homography).reshape(-1, 2)


def valid_homography_geometry(
    homography: np.ndarray,
    image_shape: tuple[int, int, int],
    max_canvas_factor: float = 4.5,
) -> tuple[bool, str]:
    corners = transformed_corners(image_shape, homography)
    if not np.isfinite(corners).all():
        return False, "nonfinite-corners"
    width = float(corners[:, 0].max() - corners[:, 0].min())
    height = float(corners[:, 1].max() - corners[:, 1].min())
    src_h, src_w = image_shape[:2]
    if width < 0.15 * src_w or height < 0.15 * src_h:
        return False, "collapsed-warp"
    if width > max_canvas_factor * src_w or height > max_canvas_factor * src_h:
        return False, "oversized-warp"
    area = cv2.contourArea(corners.astype(np.float32))
    if area < 0.03 * src_w * src_h:
        return False, "tiny-warp-area"
    return True, "ok"


def build_board_candidates(
    left_detection: BoardDetection,
    right_detection: BoardDetection,
    min_pattern: tuple[int, int],
    last_direct_h: np.ndarray | None,
    max_candidates: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int], str]]:
    left_grid = grid_from_points(left_detection.points, left_detection.pattern)
    right_grid_base = grid_from_points(right_detection.points, right_detection.pattern)
    orientations = ("id", "flip_x", "flip_y", "flip_xy")

    emitted = 0
    for common_pattern in common_patterns(left_detection.pattern, right_detection.pattern, min_pattern):
        left_windows_by_orientation: list[tuple[str, list[tuple[tuple[int, int], np.ndarray]]]] = []
        right_windows_by_orientation: list[tuple[str, list[tuple[tuple[int, int], np.ndarray]]]] = []
        for orientation, left_oriented in grid_variants(left_grid, include_transpose=True):
            left_windows_by_orientation.append(
                (orientation, list(windows_for_pattern(left_oriented, common_pattern)))
            )
        for orientation, right_oriented in grid_variants(right_grid_base, include_transpose=True):
            right_windows_by_orientation.append(
                (orientation, list(windows_for_pattern(right_oriented, common_pattern)))
            )

        raw_pairs: list[tuple[float, tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], str]]] = []
        for left_orientation, left_windows in left_windows_by_orientation:
            for left_window, left_points in left_windows:
                for right_orientation, right_windows in right_windows_by_orientation:
                    for right_window, right_points in right_windows:
                        if left_points.shape[0] != right_points.shape[0]:
                            continue
                        if last_direct_h is None:
                            priority = 0.0
                        else:
                            priority = projection_error(last_direct_h, left_points, right_points)
                        orientation = f"{left_orientation}->{right_orientation}"
                        raw_pairs.append(
                            (priority, (left_points, right_points, left_window, right_window, orientation))
                        )

        raw_pairs.sort(key=lambda item: item[0])
        for _, (left_points, right_points, left_window, right_window, orientation) in raw_pairs:
            yield left_points, right_points, common_pattern, left_window, right_window, orientation
            emitted += 1
            if emitted >= max_candidates:
                return


def select_board_match(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_detection: BoardDetection,
    right_detection: BoardDetection,
    left_green_points: np.ndarray,
    right_green_points: np.ndarray,
    min_pattern: tuple[int, int],
    last_direct_h: np.ndarray | None,
    args: argparse.Namespace,
) -> MatchCandidate | None:
    full_pattern = (args.pattern_cols, args.pattern_rows)
    if left_detection.pattern == full_pattern and right_detection.pattern == full_pattern:
        left_points = left_detection.points.astype(np.float32)
        right_grid_base = grid_from_points(right_detection.points, right_detection.pattern)
        full_candidates: list[MatchCandidate] = []
        for orientation in ("id", "flip_x", "flip_y", "flip_xy"):
            right_points = orient_grid(right_grid_base, orientation).reshape(-1, 2).astype(np.float32)
            h, inliers, reproj = estimate_homography(left_points, right_points)
            if h is None or reproj > args.max_reprojection_error:
                continue
            geometry_ok, _ = valid_homography_geometry(h, left_image.shape)
            if not geometry_ok:
                continue
            prior = projection_error(last_direct_h, left_points, right_points) if last_direct_h is not None else 0.0
            edge = edge_overlap_error(left_image, right_image, h, args.edge_scale)
            green_error, twist_angle, local_det = candidate_physical_metrics(
                h, left_points, left_green_points, right_green_points
            )
            if not physical_metrics_ok(green_error, twist_angle, local_det, args):
                continue
            has_green_check = np.isfinite(green_error) and green_error <= args.max_green_error
            if last_direct_h is None and edge > args.max_edge_error and not has_green_check:
                continue
            green_term = 0.0 if not np.isfinite(green_error) else args.green_score_weight * green_error
            score = (
                reproj
                + 0.018 * prior
                + 0.25 * edge
                + green_term
                + 0.030 * abs(twist_angle)
                - 0.018 * (full_pattern[0] * full_pattern[1])
            )
            full_candidates.append(
                MatchCandidate(
                    left_points=left_points,
                    right_points=right_points,
                    common_pattern=full_pattern,
                    left_window=(0, 0),
                    right_window=(0, 0),
                    right_orientation=orientation,
                    source="board",
                    homography=h,
                    inliers=inliers,
                    reprojection_error=reproj,
                    prior_error=prior,
                    edge_error=edge,
                    green_error=green_error,
                    twist_angle_deg=twist_angle,
                    local_det=local_det,
                    score=score,
                )
            )
        if full_candidates:
            full_candidates.sort(key=lambda c: c.score)
            return full_candidates[0]

    candidates: list[MatchCandidate] = []
    for left_points, right_points, common_pattern, left_window, right_window, orientation in build_board_candidates(
        left_detection,
        right_detection,
        min_pattern,
        last_direct_h,
        args.max_board_candidates,
    ):
        h, inliers, reproj = estimate_homography(left_points, right_points)
        if h is None:
            continue
        if reproj > args.max_reprojection_error:
            continue
        geometry_ok, _ = valid_homography_geometry(h, left_image.shape)
        if not geometry_ok:
            continue
        green_error, twist_angle, local_det = candidate_physical_metrics(
            h, left_points, left_green_points, right_green_points
        )
        if not physical_metrics_ok(green_error, twist_angle, local_det, args):
            continue
        prior = projection_error(last_direct_h, left_points, right_points) if last_direct_h is not None else 0.0
        area = common_pattern[0] * common_pattern[1]
        green_term = 0.0 if not np.isfinite(green_error) else args.green_score_weight * green_error
        preliminary = reproj + 0.018 * prior + green_term + 0.030 * abs(twist_angle) - 0.018 * area
        candidates.append(
            MatchCandidate(
                left_points=left_points,
                right_points=right_points,
                common_pattern=common_pattern,
                left_window=left_window,
                right_window=right_window,
                right_orientation=orientation,
                source="board",
                homography=h,
                inliers=inliers,
                reprojection_error=reproj,
                prior_error=prior,
                edge_error=1.0,
                green_error=green_error,
                twist_angle_deg=twist_angle,
                local_det=local_det,
                score=preliminary,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c.score)

    edge_scored: list[MatchCandidate] = []
    for candidate in candidates[: args.max_edge_candidates]:
        edge = edge_overlap_error(left_image, right_image, candidate.homography, args.edge_scale)
        candidate.edge_error = edge
        green_term = 0.0 if not np.isfinite(candidate.green_error) else args.green_score_weight * candidate.green_error
        candidate.score = (
            candidate.reprojection_error
            + 0.018 * candidate.prior_error
            + 0.25 * candidate.edge_error
            + green_term
            + 0.030 * abs(candidate.twist_angle_deg)
            - 0.018 * (candidate.common_pattern[0] * candidate.common_pattern[1])
        )
        edge_scored.append(candidate)

    if not edge_scored:
        return None
    edge_scored.sort(key=lambda c: c.score)
    best = edge_scored[0]
    if last_direct_h is not None and best.prior_error > args.max_board_prior_error:
        return None
    best_has_green_check = np.isfinite(best.green_error) and best.green_error <= args.max_green_error
    if last_direct_h is None and best.edge_error > args.max_edge_error and not best_has_green_check:
        return None
    if best.edge_error > args.max_edge_error and len(edge_scored) > 1:
        lower_edge = [c for c in edge_scored if c.edge_error <= args.max_edge_error]
        if lower_edge:
            lower_edge.sort(key=lambda c: c.score)
            best = lower_edge[0]
    return best


def sift_match(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_green_points: np.ndarray,
    right_green_points: np.ndarray,
    last_direct_h: np.ndarray | None,
    args: argparse.Namespace,
) -> MatchCandidate | None:
    if not hasattr(cv2, "SIFT_create"):
        return None
    gray_l = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=args.sift_features, contrastThreshold=0.025)
    kp_l, desc_l = sift.detectAndCompute(gray_l, None)
    kp_r, desc_r = sift.detectAndCompute(gray_r, None)
    if desc_l is None or desc_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_l, desc_r, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)
    if len(good) < 10:
        return None

    left_points = np.float32([kp_l[m.queryIdx].pt for m in good])
    right_points = np.float32([kp_r[m.trainIdx].pt for m in good])
    h, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 4.0)
    h = normalize_h(h)
    if h is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 10:
        return None
    left_inliers = left_points[inlier_mask]
    right_inliers = right_points[inlier_mask]
    reproj = projection_error(h, left_inliers, right_inliers)
    prior = projection_error(last_direct_h, left_inliers, right_inliers) if last_direct_h is not None else 0.0
    geometry_ok, _ = valid_homography_geometry(h, left_image.shape)
    if not geometry_ok:
        return None
    green_error, twist_angle, local_det = candidate_physical_metrics(
        h, left_inliers, left_green_points, right_green_points
    )
    if not physical_metrics_ok(green_error, twist_angle, local_det, args):
        return None
    if last_direct_h is not None and prior > args.max_sift_prior_error:
        return None
    edge = edge_overlap_error(left_image, right_image, h, args.edge_scale)
    if edge > args.max_edge_error:
        return None
    green_term = 0.0 if not np.isfinite(green_error) else args.green_score_weight * green_error
    score = reproj + 0.020 * prior + 0.25 * edge + green_term + 0.030 * abs(twist_angle)
    return MatchCandidate(
        left_points=left_inliers,
        right_points=right_inliers,
        common_pattern=(0, 0),
        left_window=(0, 0),
        right_window=(0, 0),
        right_orientation="sift",
        source="sift",
        homography=h,
        inliers=inliers,
        reprojection_error=reproj,
        prior_error=prior,
        edge_error=edge,
        green_error=green_error,
        twist_angle_deg=twist_angle,
        local_det=local_det,
        score=score,
    )


def matrices_for_plane(
    candidate: MatchCandidate,
    stitch_plane: str,
) -> tuple[np.ndarray, np.ndarray]:
    if stitch_plane == "right":
        return candidate.homography.copy(), np.eye(3, dtype=np.float64)
    mid_points = 0.5 * (candidate.left_points + candidate.right_points)
    left_h, _, _ = estimate_homography(candidate.left_points, mid_points, ransac_threshold=3.0)
    right_h, _, _ = estimate_homography(candidate.right_points, mid_points, ransac_threshold=3.0)
    if left_h is None or right_h is None:
        return candidate.homography.copy(), np.eye(3, dtype=np.float64)
    return left_h, right_h


def warp_and_blend(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_h: np.ndarray,
    right_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_corners = transformed_corners(left_image.shape, left_h)
    right_corners = transformed_corners(right_image.shape, right_h)
    all_corners = np.vstack([left_corners, right_corners])
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    out_w = int(max_x - min_x)
    out_h = int(max_y - min_y)
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)
    translate = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    canvas_left_h = translate @ left_h
    canvas_right_h = translate @ right_h

    warped_l = cv2.warpPerspective(left_image, canvas_left_h, (out_w, out_h))
    warped_r = cv2.warpPerspective(right_image, canvas_right_h, (out_w, out_h))
    mask_src_l = np.ones(left_image.shape[:2], dtype=np.uint8) * 255
    mask_src_r = np.ones(right_image.shape[:2], dtype=np.uint8) * 255
    mask_l = cv2.warpPerspective(mask_src_l, canvas_left_h, (out_w, out_h), flags=cv2.INTER_NEAREST)
    mask_r = cv2.warpPerspective(mask_src_r, canvas_right_h, (out_w, out_h), flags=cv2.INTER_NEAREST)

    dist_l = cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    dist_r = cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    weight_l = dist_l / (dist_l + dist_r + 1e-6)
    weight_r = dist_r / (dist_l + dist_r + 1e-6)
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
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    valid = (mask_l > 0) | (mask_r > 0)
    coords = cv2.findNonZero(valid.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        blended = blended[y : y + h, x : x + w]
        crop_translate = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]], dtype=np.float64)
        canvas_left_h = crop_translate @ canvas_left_h
        canvas_right_h = crop_translate @ canvas_right_h
    return blended, canvas_left_h, canvas_right_h


def save_debug_match(
    debug_dir: Path,
    stem: str,
    left_image: np.ndarray,
    right_image: np.ndarray,
    candidate: MatchCandidate,
) -> None:
    left_kp = [cv2.KeyPoint(float(x), float(y), 6) for x, y in candidate.left_points]
    right_kp = [cv2.KeyPoint(float(x), float(y), 6) for x, y in candidate.right_points]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(left_kp))]
    shown = cv2.drawMatches(
        left_image,
        left_kp,
        right_image,
        right_kp,
        matches[:120],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{stem}_matches.jpg"), shown)


def matrix_columns(prefix: str) -> list[str]:
    return [f"{prefix}{r}{c}" for r in range(1, 4) for c in range(1, 4)]


def flatten_matrix(matrix: np.ndarray) -> list[float]:
    return [float(x) for x in matrix.reshape(-1)]


def process_pair(
    left_path: Path,
    right_path: Path,
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_patterns: list[tuple[int, int]],
    right_patterns: list[tuple[int, int]],
    min_pattern: tuple[int, int],
    last_result: ProcessResult | None,
    args: argparse.Namespace,
) -> ProcessResult | None:
    last_direct_h = last_result.direct_h if last_result is not None else None

    left_detection = detect_chessboard(left_image, left_patterns, args.detect_scale)
    right_detection = detect_chessboard(right_image, right_patterns, args.detect_scale)
    left_green_points = detect_green_points(left_image)
    right_green_points = detect_green_points(right_image)
    candidate: MatchCandidate | None = None
    notes: list[str] = []

    if left_detection is not None and right_detection is not None:
        candidate = select_board_match(
            left_image,
            right_image,
            left_detection,
            right_detection,
            left_green_points,
            right_green_points,
            min_pattern,
            last_direct_h,
            args,
        )
        if candidate is None:
            notes.append("board-rejected")
    else:
        if left_detection is None:
            notes.append("left-board-missing")
        if right_detection is None:
            notes.append("right-board-missing")

    if candidate is None:
        candidate = sift_match(left_image, right_image, left_green_points, right_green_points, last_direct_h, args)
        if candidate is not None:
            notes.append("used-sift")

    if candidate is None and last_result is not None:
        stitched, canvas_left_h, canvas_right_h = warp_and_blend(
            left_image,
            right_image,
            last_result.left_h,
            last_result.right_h,
        )
        return ProcessResult(
            status="reused_previous",
            source="previous",
            left_h=last_result.left_h.copy(),
            right_h=last_result.right_h.copy(),
            direct_h=last_result.direct_h.copy(),
            canvas_left_h=canvas_left_h,
            canvas_right_h=canvas_right_h,
            left_points=np.empty((0, 2), dtype=np.float32),
            right_points=np.empty((0, 2), dtype=np.float32),
            left_pattern=left_detection.pattern if left_detection else None,
            right_pattern=right_detection.pattern if right_detection else None,
            common_pattern=None,
            inliers=0,
            reprojection_error=float("inf"),
            prior_error=float("inf"),
            edge_error=float("inf"),
            green_error=float("inf"),
            twist_angle_deg=float("inf"),
            local_det=float("inf"),
            notes=";".join(notes + ["no-valid-new-h"]),
        )

    if candidate is None:
        return None

    left_h, right_h = matrices_for_plane(candidate, args.stitch_plane)
    stitched, canvas_left_h, canvas_right_h = warp_and_blend(left_image, right_image, left_h, right_h)
    return ProcessResult(
        status="accepted",
        source=candidate.source,
        left_h=left_h,
        right_h=right_h,
        direct_h=candidate.homography,
        canvas_left_h=canvas_left_h,
        canvas_right_h=canvas_right_h,
        left_points=candidate.left_points,
        right_points=candidate.right_points,
        left_pattern=left_detection.pattern if left_detection else None,
        right_pattern=right_detection.pattern if right_detection else None,
        common_pattern=candidate.common_pattern,
        inliers=candidate.inliers,
        reprojection_error=candidate.reprojection_error,
        prior_error=candidate.prior_error,
        edge_error=candidate.edge_error,
        green_error=candidate.green_error,
        twist_angle_deg=candidate.twist_angle_deg,
        local_det=candidate.local_det,
        notes=";".join(notes),
    )


def prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    stitched_dir = output_dir / "stitched_frames"
    matrices_dir = output_dir / "matrices"
    debug_dir = output_dir / "debug_matches"
    for folder in (stitched_dir, matrices_dir, debug_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for folder in (stitched_dir, debug_dir):
            for path in folder.glob("*"):
                if path.is_file():
                    path.unlink()
        for path in matrices_dir.glob("*"):
            if path.is_file():
                path.unlink()
    return stitched_dir, matrices_dir, debug_dir


def pattern_to_text(pattern: tuple[int, int] | None) -> str:
    if pattern is None:
        return ""
    return f"{pattern[0]}x{pattern[1]}"


def main() -> int:
    args = parse_args()
    start_time = time.time()
    if not args.left_dir.exists():
        raise FileNotFoundError(f"left-dir does not exist: {args.left_dir}")
    if not args.right_dir.exists():
        raise FileNotFoundError(f"right-dir does not exist: {args.right_dir}")

    stitched_dir, matrices_dir, debug_dir = prepare_output(args.output_dir, args.overwrite)
    pairs = paired_frames(args.left_dir, args.right_dir)
    if args.start_frame is not None:
        pairs = [(l, r) for l, r in pairs if (frame_number(l) or -1) >= args.start_frame]
    if args.end_frame is not None:
        pairs = [(l, r) for l, r in pairs if (frame_number(l) or -1) <= args.end_frame]
    if args.max_frames and args.max_frames > 0:
        pairs = pairs[: args.max_frames]
    if not pairs:
        raise RuntimeError("No matched left/right frames found.")

    full_pattern = (args.pattern_cols, args.pattern_rows)
    min_pattern = (args.min_pattern_cols, args.min_pattern_rows)
    csv_path = matrices_dir / "homography_summary.csv"
    flagged_path = matrices_dir / "flagged_frames.csv"
    npz_path = matrices_dir / "homography_matrices_and_points.npz"
    metadata_path = matrices_dir / "run_metadata.json"

    fieldnames = [
        "frame",
        "left_path",
        "right_path",
        "status",
        "source",
        "left_pattern",
        "right_pattern",
        "common_pattern",
        "inliers",
        "reprojection_error",
        "prior_error",
        "edge_error",
        "green_error",
        "twist_angle_deg",
        "local_det",
        "notes",
    ] + matrix_columns("left_h_") + matrix_columns("right_h_") + matrix_columns("direct_h_") + matrix_columns("canvas_left_h_") + matrix_columns("canvas_right_h_")

    rows: list[dict[str, object]] = []
    flagged: list[dict[str, object]] = []
    npz_data: dict[str, np.ndarray] = {}
    last_result: ProcessResult | None = None
    preferred_left_patterns: list[tuple[int, int]] = []
    preferred_right_patterns: list[tuple[int, int]] = []
    status_counts: dict[str, int] = {}

    for idx, (left_path, right_path) in enumerate(pairs, start=1):
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            status = "read_failed"
            status_counts[status] = status_counts.get(status, 0) + 1
            flagged.append({"frame": left_path.name, "status": status, "notes": "image-read-failed"})
            continue

        left_pattern_order = make_pattern_order(full_pattern, min_pattern, preferred_left_patterns)
        right_pattern_order = make_pattern_order(full_pattern, min_pattern, preferred_right_patterns)

        result = process_pair(
            left_path,
            right_path,
            left_image,
            right_image,
            left_pattern_order,
            right_pattern_order,
            min_pattern,
            last_result,
            args,
        )

        if result is None:
            status = "failed"
            status_counts[status] = status_counts.get(status, 0) + 1
            flagged.append({"frame": left_path.name, "status": status, "notes": "no-homography"})
            continue

        stitched, canvas_left_h, canvas_right_h = warp_and_blend(
            left_image,
            right_image,
            result.left_h,
            result.right_h,
        )
        result.canvas_left_h = canvas_left_h
        result.canvas_right_h = canvas_right_h
        out_path = stitched_dir / left_path.name
        cv2.imwrite(str(out_path), stitched)

        stem = left_path.stem
        npz_data[f"{stem}_left_h"] = result.left_h.astype(np.float64)
        npz_data[f"{stem}_right_h"] = result.right_h.astype(np.float64)
        npz_data[f"{stem}_direct_left_to_right_h"] = result.direct_h.astype(np.float64)
        npz_data[f"{stem}_canvas_left_h"] = result.canvas_left_h.astype(np.float64)
        npz_data[f"{stem}_canvas_right_h"] = result.canvas_right_h.astype(np.float64)
        npz_data[f"{stem}_left_points"] = result.left_points.astype(np.float32)
        npz_data[f"{stem}_right_points"] = result.right_points.astype(np.float32)

        row: dict[str, object] = {
            "frame": left_path.name,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "status": result.status,
            "source": result.source,
            "left_pattern": pattern_to_text(result.left_pattern),
            "right_pattern": pattern_to_text(result.right_pattern),
            "common_pattern": pattern_to_text(result.common_pattern),
            "inliers": result.inliers,
            "reprojection_error": result.reprojection_error,
            "prior_error": result.prior_error,
            "edge_error": result.edge_error,
            "green_error": result.green_error,
            "twist_angle_deg": result.twist_angle_deg,
            "local_det": result.local_det,
            "notes": result.notes,
        }
        for key, value in zip(matrix_columns("left_h_"), flatten_matrix(result.left_h)):
            row[key] = value
        for key, value in zip(matrix_columns("right_h_"), flatten_matrix(result.right_h)):
            row[key] = value
        for key, value in zip(matrix_columns("direct_h_"), flatten_matrix(result.direct_h)):
            row[key] = value
        for key, value in zip(matrix_columns("canvas_left_h_"), flatten_matrix(result.canvas_left_h)):
            row[key] = value
        for key, value in zip(matrix_columns("canvas_right_h_"), flatten_matrix(result.canvas_right_h)):
            row[key] = value
        rows.append(row)

        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        if result.status != "accepted" or result.source != "board" or "rejected" in result.notes:
            flagged.append(
                {
                    "frame": left_path.name,
                    "status": result.status,
                    "source": result.source,
                    "prior_error": result.prior_error,
                    "edge_error": result.edge_error,
                    "green_error": result.green_error,
                    "twist_angle_deg": result.twist_angle_deg,
                    "local_det": result.local_det,
                    "notes": result.notes,
                }
            )
        if args.save_debug_matches and result.left_points.shape[0] > 0:
            n = frame_number(left_path) or 0
            if args.debug_every <= 1 or n % args.debug_every == 0 or result.status != "accepted":
                pseudo_candidate = MatchCandidate(
                    left_points=result.left_points,
                    right_points=result.right_points,
                    common_pattern=result.common_pattern or (0, 0),
                    left_window=(0, 0),
                    right_window=(0, 0),
                    right_orientation="",
                    source=result.source,
                    homography=result.direct_h,
                    inliers=result.inliers,
                    reprojection_error=result.reprojection_error,
                    prior_error=result.prior_error,
                    edge_error=result.edge_error,
                    green_error=result.green_error,
                    twist_angle_deg=result.twist_angle_deg,
                    local_det=result.local_det,
                    score=0.0,
                )
                save_debug_match(debug_dir, stem, left_image, right_image, pseudo_candidate)

        if result.status == "accepted":
            last_result = result
            if result.left_pattern is not None:
                preferred_left_patterns = [result.left_pattern]
            if result.right_pattern is not None:
                preferred_right_patterns = [result.right_pattern]

        if args.progress_every > 0 and (idx == 1 or idx % args.progress_every == 0 or idx == len(pairs)):
            elapsed = time.time() - start_time
            print(
                f"[{idx}/{len(pairs)}] {left_path.name}: {result.status}/{result.source}, "
                f"pattern {pattern_to_text(result.left_pattern)}->{pattern_to_text(result.right_pattern)}, "
                f"common {pattern_to_text(result.common_pattern)}, "
                f"prior={result.prior_error:.2f}, green={result.green_error:.1f}, "
                f"angle={result.twist_angle_deg:.2f}, det={result.local_det:.3f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with flagged_path.open("w", newline="", encoding="utf-8") as f:
        flagged_fields = [
            "frame",
            "status",
            "source",
            "prior_error",
            "edge_error",
            "green_error",
            "twist_angle_deg",
            "local_det",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=flagged_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flagged)

    if npz_data:
        np.savez_compressed(npz_path, **npz_data)

    metadata = {
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "output_dir": str(args.output_dir),
        "stitched_dir": str(stitched_dir),
        "matrices_dir": str(matrices_dir),
        "frame_count_requested": len(pairs),
        "frame_count_written": len(rows),
        "status_counts": status_counts,
        "full_inner_corner_pattern": list(full_pattern),
        "min_inner_corner_pattern": list(min_pattern),
        "stitch_plane": args.stitch_plane,
        "elapsed_seconds": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
