#!/usr/bin/env python
r"""
Estimate per-frame left/right homographies and stitch paired robotic-eye frames.

Default inputs:
  D:\visualization2026\left_downsample10_test1
  D:\visualization2026\right_downsample10_test1

Default outputs:
  D:\visualization2026\downsample10_test1\stitched_frames
  D:\visualization2026\downsample10_test1\matrices
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ChessboardDetection:
    ok: bool
    points: np.ndarray
    pattern: tuple[int, int]
    method: str

    @property
    def count(self) -> int:
        return int(self.points.shape[0]) if self.ok else 0

    @property
    def cols(self) -> int:
        return int(self.pattern[0])

    @property
    def rows(self) -> int:
        return int(self.pattern[1])

    def grid(self) -> np.ndarray:
        return self.points.reshape(self.rows, self.cols, 2)


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def list_frame_pairs(left_dir: Path, right_dir: Path) -> list[tuple[Path, Path, str]]:
    left_files = {
        p.name: p for p in left_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    right_files = {
        p.name: p for p in right_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    names = sorted(left_files.keys() & right_files.keys(), key=lambda n: natural_key(Path(n)))
    return [(left_files[name], right_files[name], name) for name in names]


def pattern_candidates(
    full_pattern: tuple[int, int],
    min_pattern: tuple[int, int],
    try_transposed: bool,
) -> list[tuple[int, int]]:
    full_cols, full_rows = full_pattern
    min_cols, min_rows = min_pattern
    candidates: list[tuple[int, int]] = []
    for rows in range(full_rows, min_rows - 1, -1):
        for cols in range(full_cols, min_cols - 1, -1):
            candidates.append((cols, rows))
    candidates.sort(key=lambda p: (p[0] * p[1], p[0], p[1]), reverse=True)
    if try_transposed:
        for cols, rows in list(candidates):
            if (rows, cols) not in candidates:
                candidates.append((rows, cols))
    return list(dict.fromkeys(candidates))


def find_chessboard(
    gray: np.ndarray,
    pattern: tuple[int, int],
    use_sb: bool,
) -> tuple[bool, np.ndarray, str]:
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = (
            cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        try:
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=sb_flags)
            if ok:
                return True, corners.reshape(-1, 2).astype(np.float32), "sb"
        except cv2.error:
            pass

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, classic_flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners.reshape(-1, 2).astype(np.float32), "classic"

    return False, np.empty((0, 2), dtype=np.float32), "none"


def detect_best_chessboard(
    image: np.ndarray,
    candidates: list[tuple[int, int]],
    detect_scale: float,
    use_sb: bool,
) -> ChessboardDetection:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detect_scale != 1.0:
        scaled = cv2.resize(
            gray,
            None,
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_AREA if detect_scale < 1.0 else cv2.INTER_CUBIC,
        )
    else:
        scaled = gray

    for pattern in candidates:
        ok, points, method = find_chessboard(scaled, pattern, use_sb)
        if ok:
            points = points / float(detect_scale)
            return ChessboardDetection(True, points, pattern, method)

    return ChessboardDetection(False, np.empty((0, 2), dtype=np.float32), (0, 0), "none")


def grid_variants(points: np.ndarray, pattern: tuple[int, int]) -> Iterable[np.ndarray]:
    cols, rows = pattern
    grid = points.reshape(rows, cols, 2)
    variants = [
        grid,
        grid[:, ::-1, :],
        grid[::-1, :, :],
        grid[::-1, ::-1, :],
    ]
    if rows == cols:
        transposed = np.transpose(grid, (1, 0, 2))
        variants.extend(
            [
                transposed,
                transposed[:, ::-1, :],
                transposed[::-1, :, :],
                transposed[::-1, ::-1, :],
            ]
        )
    seen: set[bytes] = set()
    for item in variants:
        flat = np.ascontiguousarray(item.reshape(-1, 2).astype(np.float32))
        key = np.round(flat, 3).tobytes()
        if key not in seen:
            seen.add(key)
            yield flat


def subwindows_from_points(
    points: np.ndarray,
    full_pattern: tuple[int, int],
    sub_pattern: tuple[int, int],
) -> Iterable[np.ndarray]:
    full_cols, full_rows = full_pattern
    sub_cols, sub_rows = sub_pattern
    full_grid = points.reshape(full_rows, full_cols, 2)
    for row0 in range(0, full_rows - sub_rows + 1):
        for col0 in range(0, full_cols - sub_cols + 1):
            patch = full_grid[row0 : row0 + sub_rows, col0 : col0 + sub_cols, :]
            yield np.ascontiguousarray(patch.reshape(-1, 2).astype(np.float32))


def transform_points(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, h_matrix).reshape(-1, 2)


def median_projection_error(src: np.ndarray, dst: np.ndarray, h_matrix: np.ndarray) -> float:
    if h_matrix is None:
        return float("inf")
    projected = transform_points(src, h_matrix)
    return float(np.median(np.linalg.norm(projected - dst, axis=1)))


def estimate_feature_homography(
    left: np.ndarray,
    right: np.ndarray,
    ratio: float,
    ransac_thresh: float,
    min_inliers: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int, int]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=3500)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING

    keypoints_l, desc_l = detector.detectAndCompute(left_gray, None)
    keypoints_r, desc_r = detector.detectAndCompute(right_gray, None)
    if desc_l is None or desc_r is None or len(keypoints_l) < 4 or len(keypoints_r) < 4:
        empty = np.empty((0, 2), dtype=np.float32)
        return None, empty, empty, 0, 0

    matcher = cv2.BFMatcher(norm)
    raw_matches = matcher.knnMatch(desc_l, desc_r, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)

    if len(good) < 4:
        empty = np.empty((0, 2), dtype=np.float32)
        return None, empty, empty, len(good), 0

    left_pts = np.float32([keypoints_l[m.queryIdx].pt for m in good])
    right_pts = np.float32([keypoints_r[m.trainIdx].pt for m in good])
    h_matrix, inlier_mask = cv2.findHomography(
        left_pts, right_pts, cv2.RANSAC, ransac_thresh
    )
    if h_matrix is None or inlier_mask is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return None, empty, empty, len(good), 0

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        empty = np.empty((0, 2), dtype=np.float32)
        return None, empty, empty, len(good), inlier_count

    return (
        h_matrix.astype(np.float64),
        left_pts[inliers].astype(np.float32),
        right_pts[inliers].astype(np.float32),
        len(good),
        inlier_count,
    )


def match_chessboard_points(
    left_det: ChessboardDetection,
    right_det: ChessboardDetection,
    left_to_right_hs: list[np.ndarray],
    max_error: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    empty = np.empty((0, 2), dtype=np.float32)
    if not left_det.ok or not right_det.ok:
        return empty, empty, "none", float("inf")

    best_left = empty
    best_right = empty
    best_method = "none"
    best_error = float("inf")

    def pair_error(left_pts: np.ndarray, right_pts: np.ndarray) -> float:
        if not left_to_right_hs:
            return 0.0
        return min(
            median_projection_error(left_pts, right_pts, h_matrix)
            for h_matrix in left_to_right_hs
        )

    def consider(left_pts: np.ndarray, right_pts: np.ndarray, method: str) -> None:
        nonlocal best_left, best_right, best_method, best_error
        error = pair_error(left_pts, right_pts)
        if error < best_error:
            best_left = left_pts.astype(np.float32)
            best_right = right_pts.astype(np.float32)
            best_method = method
            best_error = error

    if left_det.pattern == right_det.pattern:
        for right_variant in grid_variants(right_det.points, right_det.pattern):
            consider(left_det.points, right_variant, "chessboard_same_pattern")

    if left_det.count > right_det.count:
        for left_full_variant in grid_variants(left_det.points, left_det.pattern):
            for left_window in subwindows_from_points(
                left_full_variant, left_det.pattern, right_det.pattern
            ):
                for left_window_variant in grid_variants(left_window, right_det.pattern):
                    for right_variant in grid_variants(right_det.points, right_det.pattern):
                        consider(
                            left_window_variant,
                            right_variant,
                            "chessboard_left_full_window",
                        )

    if right_det.count > left_det.count:
        for right_full_variant in grid_variants(right_det.points, right_det.pattern):
            for right_window in subwindows_from_points(
                right_full_variant, right_det.pattern, left_det.pattern
            ):
                for right_window_variant in grid_variants(right_window, left_det.pattern):
                    for left_variant in grid_variants(left_det.points, left_det.pattern):
                        consider(
                            left_variant,
                            right_window_variant,
                            "chessboard_right_full_window",
                        )

    if best_left.shape[0] >= 4 and (not left_to_right_hs or best_error <= max_error):
        return best_left, best_right, best_method, best_error

    return empty, empty, "none", best_error


def chessboard_match_candidates(
    left_det: ChessboardDetection,
    right_det: ChessboardDetection,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
    if not left_det.ok or not right_det.ok:
        return candidates

    if left_det.pattern == right_det.pattern:
        for right_variant in grid_variants(right_det.points, right_det.pattern):
            candidates.append(
                (
                    left_det.points.astype(np.float32),
                    right_variant.astype(np.float32),
                    "chessboard_same_pattern",
                )
            )

    if left_det.count > right_det.count:
        for left_full_variant in grid_variants(left_det.points, left_det.pattern):
            for left_window in subwindows_from_points(
                left_full_variant, left_det.pattern, right_det.pattern
            ):
                for left_window_variant in grid_variants(left_window, right_det.pattern):
                    for right_variant in grid_variants(right_det.points, right_det.pattern):
                        candidates.append(
                            (
                                left_window_variant.astype(np.float32),
                                right_variant.astype(np.float32),
                                "chessboard_left_full_window",
                            )
                        )

    if right_det.count > left_det.count:
        for right_full_variant in grid_variants(right_det.points, right_det.pattern):
            for right_window in subwindows_from_points(
                right_full_variant, right_det.pattern, left_det.pattern
            ):
                for right_window_variant in grid_variants(right_window, left_det.pattern):
                    for left_variant in grid_variants(left_det.points, left_det.pattern):
                        candidates.append(
                            (
                                left_variant.astype(np.float32),
                                right_window_variant.astype(np.float32),
                                "chessboard_right_full_window",
                            )
                        )

    if left_det.cols != right_det.cols and left_det.rows != right_det.rows:
        common_pattern = (min(left_det.cols, right_det.cols), min(left_det.rows, right_det.rows))
        if common_pattern[0] >= 3 and common_pattern[1] >= 3:
            for left_full_variant in grid_variants(left_det.points, left_det.pattern):
                for left_window in subwindows_from_points(
                    left_full_variant, left_det.pattern, common_pattern
                ):
                    for left_window_variant in grid_variants(left_window, common_pattern):
                        for right_full_variant in grid_variants(right_det.points, right_det.pattern):
                            for right_window in subwindows_from_points(
                                right_full_variant, right_det.pattern, common_pattern
                            ):
                                for right_window_variant in grid_variants(
                                    right_window, common_pattern
                                ):
                                    candidates.append(
                                        (
                                            left_window_variant.astype(np.float32),
                                            right_window_variant.astype(np.float32),
                                            "chessboard_common_window",
                                        )
                                    )

    return candidates


def chessboard_edge_score(
    left: np.ndarray,
    right: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    ransac_thresh: float,
) -> tuple[float, np.ndarray | None]:
    h_matrix, _ = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, ransac_thresh)
    if h_matrix is None:
        return float("inf"), None

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    left_edges = cv2.Canny(left_gray, 50, 150)
    right_edges = cv2.Canny(right_gray, 50, 150)
    warped_edges = cv2.warpPerspective(
        left_edges, h_matrix, (right.shape[1], right.shape[0])
    )

    x, y, w, h = cv2.boundingRect(right_pts.astype(np.float32).reshape(-1, 1, 2))
    if right_pts.shape[0] > 1:
        margin = int(max(w, h) / max(math.sqrt(right_pts.shape[0]), 1.0))
        margin = max(60, min(180, int(margin * 1.8)))
    else:
        margin = 90
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(right.shape[1], x + w + margin)
    y1 = min(right.shape[0], y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return float("inf"), h_matrix

    kernel = np.ones((5, 5), dtype=np.uint8)
    a = cv2.dilate(warped_edges[y0:y1, x0:x1], kernel) > 0
    b = cv2.dilate(right_edges[y0:y1, x0:x1], kernel) > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return float("inf"), h_matrix
    intersection = int(np.logical_and(a, b).sum())
    return 1.0 - (intersection / union), h_matrix


def match_chessboard_points_photometric(
    left_det: ChessboardDetection,
    right_det: ChessboardDetection,
    left: np.ndarray,
    right: np.ndarray,
    ransac_thresh: float,
    max_score: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    empty = np.empty((0, 2), dtype=np.float32)
    best: tuple[float, np.ndarray, np.ndarray, str] | None = None
    for left_pts, right_pts, method in chessboard_match_candidates(left_det, right_det):
        score, _ = chessboard_edge_score(left, right, left_pts, right_pts, ransac_thresh)
        if not np.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, left_pts, right_pts, method)

    if best is None or best[0] > max_score:
        return empty, empty, "none", float("inf") if best is None else best[0]

    return best[1], best[2], f"{best[3]}_photometric", best[0]


def estimate_mid_homographies(
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    ransac_thresh: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    if left_pts.shape[0] < 4 or right_pts.shape[0] < 4:
        return None, None, 0, 0

    mid_pts = ((left_pts + right_pts) / 2.0).astype(np.float32)
    left_h, left_mask = cv2.findHomography(left_pts, mid_pts, cv2.RANSAC, ransac_thresh)
    right_h, right_mask = cv2.findHomography(right_pts, mid_pts, cv2.RANSAC, ransac_thresh)
    if left_h is None or right_h is None or left_mask is None or right_mask is None:
        return None, None, 0, 0

    left_inliers = int(left_mask.reshape(-1).sum())
    right_inliers = int(right_mask.reshape(-1).sum())
    if left_inliers < 4 or right_inliers < 4:
        return None, None, left_inliers, right_inliers

    return left_h.astype(np.float64), right_h.astype(np.float64), left_inliers, right_inliers


def estimate_right_plane_homographies(
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    ransac_thresh: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    if left_pts.shape[0] < 4 or right_pts.shape[0] < 4:
        return None, None, 0, 0
    left_h, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, ransac_thresh)
    if left_h is None or mask is None:
        return None, None, 0, 0
    inliers = int(mask.reshape(-1).sum())
    if inliers < 4:
        return None, None, inliers, inliers
    return left_h.astype(np.float64), np.eye(3, dtype=np.float64), inliers, right_pts.shape[0]


def estimate_output_homographies(
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    ransac_thresh: float,
    stitch_plane: str,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    if stitch_plane == "right":
        return estimate_right_plane_homographies(left_pts, right_pts, ransac_thresh)
    return estimate_mid_homographies(left_pts, right_pts, ransac_thresh)


def is_valid_homography(h_matrix: np.ndarray | None) -> bool:
    if h_matrix is None:
        return False
    if h_matrix.shape != (3, 3) or not np.isfinite(h_matrix).all():
        return False
    return abs(float(h_matrix[2, 2])) > 1e-12


def normalize_homography(h_matrix: np.ndarray | None) -> np.ndarray | None:
    if not is_valid_homography(h_matrix):
        return None
    return h_matrix.astype(np.float64) / float(h_matrix[2, 2])


def direct_homography(left_h: np.ndarray | None, right_h: np.ndarray | None) -> np.ndarray | None:
    if not (is_valid_homography(left_h) and is_valid_homography(right_h)):
        return None
    try:
        return normalize_homography(np.linalg.inv(right_h) @ left_h)
    except np.linalg.LinAlgError:
        return None


def image_probe_points(shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    h, w = shape[:2]
    return np.float32(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
            [w / 2, h / 2],
            [w / 2, 0],
            [w, h / 2],
            [w / 2, h],
            [0, h / 2],
        ]
    ).reshape(-1, 1, 2)


def homography_delta(
    first_h: np.ndarray,
    second_h: np.ndarray,
    shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[float, float]:
    points = image_probe_points(shape)
    first = cv2.perspectiveTransform(points, first_h).reshape(-1, 2)
    second = cv2.perspectiveTransform(points, second_h).reshape(-1, 2)
    distances = np.linalg.norm(second - first, axis=1)
    return float(np.median(distances)), float(np.max(distances))


def output_geometry_metrics(
    left_h: np.ndarray,
    right_h: np.ndarray,
    shape: tuple[int, int] | tuple[int, int, int],
) -> dict[str, float]:
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    left_corners = cv2.perspectiveTransform(corners, left_h).reshape(-1, 2)
    right_corners = cv2.perspectiveTransform(corners, right_h).reshape(-1, 2)
    all_corners = np.vstack([left_corners, right_corners])
    min_xy = all_corners.min(axis=0)
    max_xy = all_corners.max(axis=0)
    bbox_w = float(max_xy[0] - min_xy[0])
    bbox_h = float(max_xy[1] - min_xy[1])
    aspect = bbox_w / max(bbox_h, 1e-6)
    area_scale = (bbox_w * bbox_h) / max(float(w * h), 1e-6)
    return {
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "aspect": aspect,
        "area_scale": area_scale,
    }


def validate_homography_pair(
    left_h: np.ndarray | None,
    right_h: np.ndarray | None,
    image_shape: tuple[int, int] | tuple[int, int, int],
    last_direct_h: np.ndarray | None,
    source: str,
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, float], np.ndarray | None]:
    left_h = normalize_homography(left_h)
    right_h = normalize_homography(right_h)
    if left_h is None or right_h is None:
        return False, "invalid_matrix", {}, None

    metrics = output_geometry_metrics(left_h, right_h, image_shape)
    h, w = image_shape[:2]
    min_w = args.min_output_width_ratio * w
    max_w = args.max_output_width_ratio * w
    min_h = args.min_output_height_ratio * h
    max_h = args.max_output_height_ratio * h

    if not (min_w <= metrics["bbox_width"] <= max_w):
        return False, "bad_bbox_width", metrics, None
    if not (min_h <= metrics["bbox_height"] <= max_h):
        return False, "bad_bbox_height", metrics, None
    if not (args.min_output_aspect <= metrics["aspect"] <= args.max_output_aspect):
        return False, "bad_aspect", metrics, None

    current_direct_h = direct_homography(left_h, right_h)
    if current_direct_h is None:
        return False, "bad_direct_h", metrics, None

    if last_direct_h is not None:
        temporal_median, temporal_max = homography_delta(
            last_direct_h, current_direct_h, image_shape
        )
        metrics["temporal_median_shift"] = temporal_median
        metrics["temporal_max_shift"] = temporal_max
        if source.startswith("sift") and temporal_median > args.max_sift_temporal_shift:
            return False, "sift_temporal_jump", metrics, current_direct_h
        if temporal_median > args.max_temporal_shift:
            return False, "temporal_jump", metrics, current_direct_h

    return True, "ok", metrics, current_direct_h


def warp_and_blend(
    left: np.ndarray,
    right: np.ndarray,
    left_h: np.ndarray,
    right_h: np.ndarray,
    padding: int,
    crop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    h, w = left.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    left_corners = cv2.perspectiveTransform(corners, left_h).reshape(-1, 2)
    right_corners = cv2.perspectiveTransform(corners, right_h).reshape(-1, 2)
    all_corners = np.vstack([left_corners, right_corners])

    min_xy = np.floor(all_corners.min(axis=0)).astype(int) - padding
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int) + padding
    out_w = int(max(1, max_xy[0] - min_xy[0]))
    out_h = int(max(1, max_xy[1] - min_xy[1]))

    shift = np.array(
        [[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]]
    )
    left_canvas_h = shift @ left_h
    right_canvas_h = shift @ right_h

    left_warp = cv2.warpPerspective(left, left_canvas_h, (out_w, out_h))
    right_warp = cv2.warpPerspective(right, right_canvas_h, (out_w, out_h))

    mask = np.ones((h, w), dtype=np.uint8) * 255
    left_mask = cv2.warpPerspective(mask, left_canvas_h, (out_w, out_h))
    right_mask = cv2.warpPerspective(mask, right_canvas_h, (out_w, out_h))

    left_weight = cv2.distanceTransform((left_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    right_weight = cv2.distanceTransform((right_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    weight_sum = left_weight + right_weight
    weight_sum[weight_sum == 0] = 1.0

    blended = (
        left_warp.astype(np.float32) * left_weight[..., None]
        + right_warp.astype(np.float32) * right_weight[..., None]
    ) / weight_sum[..., None]
    stitched = np.clip(blended, 0, 255).astype(np.uint8)

    crop_box = {"x": 0, "y": 0, "width": stitched.shape[1], "height": stitched.shape[0]}
    if crop:
        combined_mask = ((left_mask > 0) | (right_mask > 0)).astype(np.uint8)
        coords = cv2.findNonZero(combined_mask)
        if coords is not None:
            x, y, cw, ch = cv2.boundingRect(coords)
            stitched = stitched[y : y + ch, x : x + cw]
            crop_shift = np.array(
                [[1.0, 0.0, -float(x)], [0.0, 1.0, -float(y)], [0.0, 0.0, 1.0]]
            )
            left_canvas_h = crop_shift @ left_canvas_h
            right_canvas_h = crop_shift @ right_canvas_h
            crop_box = {"x": int(x), "y": int(y), "width": int(cw), "height": int(ch)}

    return stitched, left_canvas_h, right_canvas_h, crop_box


def matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{r + 1}{c + 1}": float(matrix[r, c]) for r in range(3) for c in range(3)}


def write_debug_image(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    max_lines: int = 80,
) -> None:
    if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
        return
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    step = max(1, left_pts.shape[0] // max_lines)
    for lp, rp in zip(left_pts[::step], right_pts[::step]):
        p1 = tuple(np.round(lp).astype(int))
        p2 = tuple(np.round(rp + np.array([left.shape[1], 0])).astype(int))
        color = (0, 255, 0)
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.circle(canvas, p2, 4, color, -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate left/right homographies and stitch downsampled robotic-eye frames."
    )
    parser.add_argument("--left-dir", type=Path, default=Path(r"D:\visualization2026\left_downsample10_test1"))
    parser.add_argument("--right-dir", type=Path, default=Path(r"D:\visualization2026\right_downsample10_test1"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\visualization2026\downsample10_test1"))
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=6,
        help="Number of inner chessboard corners along columns. Use 6 for a 7x9-square board.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=8,
        help="Number of inner chessboard corners along rows. Use 8 for a 7x9-square board.",
    )
    parser.add_argument("--min-pattern-cols", type=int, default=3)
    parser.add_argument("--min-pattern-rows", type=int, default=3)
    parser.add_argument("--try-transposed", action="store_true")
    parser.add_argument("--detect-scale", type=float, default=0.5)
    parser.add_argument("--no-sb", action="store_true", help="Disable findChessboardCornersSB.")
    parser.add_argument("--match-ratio", type=float, default=0.78)
    parser.add_argument("--ransac-thresh", type=float, default=4.0)
    parser.add_argument("--min-feature-inliers", type=int, default=12)
    parser.add_argument(
        "--stitch-plane",
        choices=("right", "middle"),
        default="right",
        help="Common output plane. 'right' warps left to the right frame; 'middle' uses the older midpoint plane.",
    )
    parser.add_argument(
        "--chessboard-max-error",
        type=float,
        default=15.0,
        help="Reject partial-board subwindow matches whose median error to the trusted prior H is above this many pixels.",
    )
    parser.add_argument(
        "--max-chessboard-edge-score",
        type=float,
        default=0.58,
        help="Accept photometric chessboard subwindow matches below this edge mismatch score.",
    )
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-reuse-last", action="store_true")
    parser.add_argument("--min-output-width-ratio", type=float, default=0.65)
    parser.add_argument("--max-output-width-ratio", type=float, default=2.6)
    parser.add_argument("--min-output-height-ratio", type=float, default=0.65)
    parser.add_argument("--max-output-height-ratio", type=float, default=2.6)
    parser.add_argument("--min-output-aspect", type=float, default=0.55)
    parser.add_argument("--max-output-aspect", type=float, default=2.6)
    parser.add_argument(
        "--max-temporal-shift",
        type=float,
        default=900.0,
        help="Reject any candidate whose direct left-to-right H jumps this many pixels from the last accepted H.",
    )
    parser.add_argument(
        "--max-sift-temporal-shift",
        type=float,
        default=260.0,
        help="Stricter temporal jump limit for SIFT fallback candidates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.left_dir.exists():
        raise FileNotFoundError(f"Left directory not found: {args.left_dir}")
    if not args.right_dir.exists():
        raise FileNotFoundError(f"Right directory not found: {args.right_dir}")

    stitched_dir = args.output_dir / "stitched_frames"
    matrix_dir = args.output_dir / "matrices"
    debug_dir = args.output_dir / "debug_matches"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_frame_pairs(args.left_dir, args.right_dir)
    if args.start_index:
        pairs = pairs[args.start_index :]
    if args.max_frames is not None:
        pairs = pairs[: args.max_frames]
    if not pairs:
        raise RuntimeError("No matching image pairs found.")

    candidates = pattern_candidates(
        (args.pattern_cols, args.pattern_rows),
        (args.min_pattern_cols, args.min_pattern_rows),
        args.try_transposed,
    )

    summary_rows: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}
    last_left_h: np.ndarray | None = None
    last_right_h: np.ndarray | None = None
    last_direct_h: np.ndarray | None = None
    last_status = ""

    print(f"Processing {len(pairs)} frame pairs")
    print(f"Chessboard inner-corner pattern: {args.pattern_cols}x{args.pattern_rows}")
    print(f"Output: {args.output_dir}")

    for index, (left_path, right_path, name) in enumerate(pairs, start=1):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            print(f"[{index}/{len(pairs)}] {name}: failed to read image")
            continue
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

        left_det = detect_best_chessboard(left, candidates, args.detect_scale, not args.no_sb)
        right_det = detect_best_chessboard(right, candidates, args.detect_scale, not args.no_sb)

        feature_direct_h, feature_left, feature_right, good_matches, feature_inliers = estimate_feature_homography(
            left,
            right,
            args.match_ratio,
            args.ransac_thresh,
            args.min_feature_inliers,
        )

        # For partial chessboards the repeated grid is ambiguous. A current
        # SIFT estimate can be wrong in exactly the same way as a wrong
        # subwindow, so prefer temporal continuity whenever it is available.
        prior_hs = []
        if last_direct_h is not None:
            prior_hs.append(last_direct_h)
        elif feature_direct_h is not None:
            prior_hs.append(feature_direct_h)

        chess_left, chess_right, pair_method, chess_error = match_chessboard_points(
            left_det, right_det, prior_hs, args.chessboard_max_error
        )
        photo_left, photo_right, photo_method, photo_score = match_chessboard_points_photometric(
            left_det,
            right_det,
            left,
            right,
            args.ransac_thresh,
            args.max_chessboard_edge_score,
        )

        candidate_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
        if photo_left.shape[0] >= 4:
            candidate_specs.append((photo_method, photo_left, photo_right))
        if chess_left.shape[0] >= 4:
            candidate_specs.append((pair_method, chess_left, chess_right))
        if feature_left.shape[0] >= 4:
            candidate_specs.append(("sift_ransac", feature_left, feature_right))

        fit_left = np.empty((0, 2), dtype=np.float32)
        fit_right = np.empty((0, 2), dtype=np.float32)
        source = "failed"
        validation_reason = "no_candidate_points"
        validation_metrics: dict[str, float] = {}
        left_h: np.ndarray | None = None
        right_h: np.ndarray | None = None
        left_mid_inliers = 0
        right_mid_inliers = 0
        current_direct_h: np.ndarray | None = None
        rejected_reasons: list[str] = []

        for candidate_source, candidate_left, candidate_right in candidate_specs:
            candidate_left_h, candidate_right_h, candidate_left_inliers, candidate_right_inliers = (
                estimate_output_homographies(
                    candidate_left,
                    candidate_right,
                    args.ransac_thresh,
                    args.stitch_plane,
                )
            )
            ok, reason, metrics, candidate_direct_h = validate_homography_pair(
                candidate_left_h,
                candidate_right_h,
                left.shape,
                last_direct_h,
                candidate_source,
                args,
            )
            if ok:
                fit_left = candidate_left
                fit_right = candidate_right
                source = candidate_source
                validation_reason = reason
                validation_metrics = metrics
                left_h = normalize_homography(candidate_left_h)
                right_h = normalize_homography(candidate_right_h)
                left_mid_inliers = candidate_left_inliers
                right_mid_inliers = candidate_right_inliers
                current_direct_h = candidate_direct_h
                break
            rejected_reasons.append(f"{candidate_source}:{reason}")
            validation_reason = reason
            validation_metrics = metrics

        reused_last = False
        if not (is_valid_homography(left_h) and is_valid_homography(right_h)):
            if not args.no_reuse_last and is_valid_homography(last_left_h) and is_valid_homography(last_right_h):
                left_h = last_left_h.copy()
                right_h = last_right_h.copy()
                current_direct_h = last_direct_h.copy() if last_direct_h is not None else None
                reused_last = True
                detail = rejected_reasons[-1].split(":", 1)[-1] if rejected_reasons else source
                source = f"reuse_last_after_{detail}"
                validation_reason = "reused_last"
            else:
                print(
                    f"[{index}/{len(pairs)}] {name}: no usable homography "
                    f"(left board {left_det.pattern}/{left_det.method}, "
                    f"right board {right_det.pattern}/{right_det.method}, "
                    f"feature inliers {feature_inliers})"
                )
                continue

        assert left_h is not None and right_h is not None
        last_left_h = left_h.copy()
        last_right_h = right_h.copy()
        if current_direct_h is not None:
            last_direct_h = current_direct_h.copy()
        last_status = source

        stitched, left_canvas_h, right_canvas_h, crop_box = warp_and_blend(
            left,
            right,
            left_h,
            right_h,
            args.padding,
            crop=not args.no_crop,
        )
        output_path = stitched_dir / name
        cv2.imwrite(str(output_path), stitched)

        stem = Path(name).stem
        matrices[f"{stem}_left_mid"] = left_h
        matrices[f"{stem}_right_mid"] = right_h
        matrices[f"{stem}_left_canvas"] = left_canvas_h
        matrices[f"{stem}_right_canvas"] = right_canvas_h

        row: dict[str, object] = {
            "frame": name,
            "status": source,
            "reused_last": int(reused_last),
            "left_pattern": f"{left_det.cols}x{left_det.rows}" if left_det.ok else "",
            "right_pattern": f"{right_det.cols}x{right_det.rows}" if right_det.ok else "",
            "left_detector": left_det.method,
            "right_detector": right_det.method,
            "left_chess_points": left_det.count,
            "right_chess_points": right_det.count,
            "paired_points": int(fit_left.shape[0]),
            "chessboard_pair_error_px": "" if math.isinf(chess_error) else f"{chess_error:.3f}",
            "chessboard_edge_score": "" if math.isinf(photo_score) else f"{photo_score:.6f}",
            "feature_good_matches": good_matches,
            "feature_inliers": feature_inliers,
            "left_mid_inliers": left_mid_inliers,
            "right_mid_inliers": right_mid_inliers,
            "stitch_plane": args.stitch_plane,
            "validation_reason": validation_reason,
            "rejected_reasons": ";".join(rejected_reasons),
            "bbox_width": f"{validation_metrics.get('bbox_width', '')}",
            "bbox_height": f"{validation_metrics.get('bbox_height', '')}",
            "bbox_aspect": f"{validation_metrics.get('aspect', '')}",
            "bbox_area_scale": f"{validation_metrics.get('area_scale', '')}",
            "temporal_median_shift": f"{validation_metrics.get('temporal_median_shift', '')}",
            "temporal_max_shift": f"{validation_metrics.get('temporal_max_shift', '')}",
            "output_width": int(stitched.shape[1]),
            "output_height": int(stitched.shape[0]),
            "crop_x": crop_box["x"],
            "crop_y": crop_box["y"],
            "crop_width": crop_box["width"],
            "crop_height": crop_box["height"],
        }
        row.update(matrix_fields("left_mid_h", left_h))
        row.update(matrix_fields("right_mid_h", right_h))
        row.update(matrix_fields("left_canvas_h", left_canvas_h))
        row.update(matrix_fields("right_canvas_h", right_canvas_h))
        summary_rows.append(row)

        if args.debug:
            debug_path = debug_dir / name
            write_debug_image(debug_path, left, right, fit_left, fit_right)

        print(
            f"[{index}/{len(pairs)}] {name}: {source}, "
            f"L {left_det.pattern}/{left_det.method}, R {right_det.pattern}/{right_det.method}, "
            f"pairs {fit_left.shape[0]}, feature inliers {feature_inliers}, "
            f"valid {validation_reason}"
        )

    if summary_rows:
        csv_path = matrix_dir / "homography_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        npz_path = matrix_dir / "homography_matrices.npz"
        np.savez_compressed(npz_path, **matrices)

        metadata = {
            "left_dir": str(args.left_dir),
            "right_dir": str(args.right_dir),
            "output_dir": str(args.output_dir),
            "pattern_cols": args.pattern_cols,
            "pattern_rows": args.pattern_rows,
            "min_pattern_cols": args.min_pattern_cols,
            "min_pattern_rows": args.min_pattern_rows,
            "detect_scale": args.detect_scale,
            "stitch_plane": args.stitch_plane,
            "processed_pairs": len(summary_rows),
            "last_status": last_status,
        }
        with (matrix_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved stitched frames: {stitched_dir}")
        print(f"Saved matrix CSV: {csv_path}")
        print(f"Saved matrix NPZ: {npz_path}")
    else:
        print("No frames were stitched.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
