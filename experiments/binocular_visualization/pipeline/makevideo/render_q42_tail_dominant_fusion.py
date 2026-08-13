#!/usr/bin/env python3
"""Render Q42 tail with dominant-eye fusion to avoid chessboard ghosting.

When Q42's final upward-gaze segment is adjusted to near-zero misalignment,
the visual scene should look fused. Even tiny residual perspective differences
between the two warped camera views make a 50/50 blend show chessboard ghosts.
This renderer keeps the tail-adjusted homographies, but fades the overlap
region from normal blending to left-dominant fusion after 10 seconds. It also
fades right-eye-only pixels to a left-eye extrapolated background in the same
tail segment, so the far edge does not keep a second chessboard sliver.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from refine_hess_match_from_raw_green_markers import (  # noqa: E402
    list_frames,
    matrix_frame_names,
    read_image,
    write_image,
)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def fast_left_extrapolated_fill(warped_l: np.ndarray, valid_l: np.ndarray) -> np.ndarray:
    """Fill left-invalid pixels from smoothed bright left-edge colors.

    The tail artifact is a narrow right-eye-only strip. Sampling the bright
    part of the nearby left-eye edge avoids copying black chess squares, and a
    strong vertical smoothing prevents row-wise stripe artifacts.
    """
    height, width = valid_l.shape
    if np.any(valid_l):
        global_color = np.median(warped_l[valid_l], axis=0).astype(np.float32)
    else:
        global_color = np.array([190, 190, 190], dtype=np.float32)

    row_colors = np.zeros((height, 1, 3), dtype=np.float32)
    for y in range(height):
        xs = np.flatnonzero(valid_l[y])
        if xs.size == 0:
            row_colors[y, 0] = global_color
            continue
        band_xs = xs[max(0, xs.size - 320) :]
        band = warped_l[y, band_xs].astype(np.float32)
        gray = 0.114 * band[:, 0] + 0.587 * band[:, 1] + 0.299 * band[:, 2]
        cutoff = np.percentile(gray, 60)
        bright_band = band[gray >= cutoff]
        if bright_band.size == 0:
            bright_band = band
        row_colors[y, 0] = np.median(bright_band, axis=0)

    row_colors = cv2.GaussianBlur(
        row_colors,
        (1, 301),
        0,
        borderType=cv2.BORDER_REPLICATE,
    )
    fill = np.zeros_like(warped_l)
    for y in range(height):
        fill[y, :] = np.clip(row_colors[y, 0], 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(fill, (0, 0), 4)


def dominant_blend(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_h: np.ndarray,
    right_h: np.ndarray,
    output_size: tuple[int, int],
    dominance: float,
) -> np.ndarray:
    out_w, out_h = output_size
    warped_l = cv2.warpPerspective(left_image, left_h, (out_w, out_h))
    warped_r = cv2.warpPerspective(right_image, right_h, (out_w, out_h))
    mask_src_l = np.ones(left_image.shape[:2], dtype=np.uint8) * 255
    mask_src_r = np.ones(right_image.shape[:2], dtype=np.uint8) * 255
    mask_l = cv2.warpPerspective(mask_src_l, left_h, (out_w, out_h), flags=cv2.INTER_NEAREST)
    mask_r = cv2.warpPerspective(mask_src_r, right_h, (out_w, out_h), flags=cv2.INTER_NEAREST)
    valid_l = mask_l > 0
    valid_r = mask_r > 0
    both = valid_l & valid_r
    right_only = valid_r & ~valid_l

    result = np.zeros_like(warped_l)
    result[valid_l & ~valid_r] = warped_l[valid_l & ~valid_r]
    if np.any(right_only):
        if dominance <= 1e-6:
            result[right_only] = warped_r[right_only]
        else:
            left_fill = fast_left_extrapolated_fill(warped_l, valid_l)
            mixed_right = (
                warped_r.astype(np.float32) * (1.0 - dominance)
                + left_fill.astype(np.float32) * dominance
            )
            result[right_only] = np.clip(mixed_right[right_only], 0, 255).astype(np.uint8)
    if np.any(both):
        dist_l = cv2.distanceTransform(valid_l.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        dist_r = cv2.distanceTransform(valid_r.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        denom = np.maximum(dist_l + dist_r, 1e-6)
        normal_wr = dist_r / denom
        # dominance=0 keeps the normal distance-weighted blend; dominance=1
        # makes left eye the fused view wherever both eyes are valid.
        wr = normal_wr * (1.0 - dominance)
        wl = 1.0 - wr
        blended = (
            warped_l.astype(np.float32) * wl[:, :, None]
            + warped_r.astype(np.float32) * wr[:, :, None]
        )
        result[both] = np.clip(blended[both], 0, 255).astype(np.uint8)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(r"D:\visualization2026\dowsample2_Q42_hess_matched")
    parser = argparse.ArgumentParser(description="Render Q42 with left-dominant fused tail.")
    parser.add_argument(
        "--left-dir",
        type=Path,
        default=Path(r"D:\visualization2026\visualization\left_downsample2_Q42"),
    )
    parser.add_argument(
        "--right-dir",
        type=Path,
        default=Path(r"D:\visualization2026\visualization\right_downsample2_Q42"),
    )
    parser.add_argument("--matrix-npz", type=Path, default=root / "matrices" / "hess_matched_homographies_Q42_q23style_tail.npz")
    parser.add_argument("--base-stitched-dir", type=Path, default=Path(r"D:\visualization2026\dowsample2_Q42\stitched_frames"))
    parser.add_argument("--output-dir", type=Path, default=root / "stitched_frames_q23style_tail_fused")
    parser.add_argument("--report-csv", type=Path, default=root / "tail_dominant_fusion_report.csv")
    parser.add_argument("--metadata-json", type=Path, default=root / "tail_dominant_fusion_metadata.json")
    parser.add_argument("--fade-start-frame", type=int, default=600)
    parser.add_argument("--full-frame", type=int, default=690)
    parser.add_argument("--full-dominance", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    matrices = np.load(args.matrix_npz, allow_pickle=True)
    stems = [
        stem
        for stem in matrix_frame_names(matrices)
        if stem in left_frames and stem in right_frames and (args.base_stitched_dir / f"{stem}.jpg").exists()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, stem in enumerate(stems):
        base = read_image(args.base_stitched_dir / f"{stem}.jpg")
        out_h, out_w = base.shape[:2]
        dominance = args.full_dominance * smoothstep(args.fade_start_frame, args.full_frame, index)
        rendered = dominant_blend(
            read_image(left_frames[stem]),
            read_image(right_frames[stem]),
            matrices[f"{stem}_canvas_left_h"].astype(np.float64),
            matrices[f"{stem}_canvas_right_h"].astype(np.float64),
            output_size=(out_w, out_h),
            dominance=dominance,
        )
        output_path = args.output_dir / f"{stem}.jpg"
        write_image(output_path, rendered, args.jpeg_quality)
        rows.append(
            {
                "frame": f"{stem}.jpg",
                "video_frame_index": index,
                "dominance": dominance,
                "output_path": str(output_path),
                "output_width": out_w,
                "output_height": out_h,
            }
        )
        if index == 0 or (index + 1) % args.progress_every == 0 or index == len(stems) - 1:
            print(f"[{index + 1}/{len(stems)}] {stem}.jpg dominance={dominance:.3f}")
    write_csv(args.report_csv, rows)
    metadata = {
        "matrix_npz": str(args.matrix_npz),
        "output_dir": str(args.output_dir),
        "frame_count": len(rows),
        "fade_start_frame": args.fade_start_frame,
        "full_frame": args.full_frame,
        "full_dominance": args.full_dominance,
        "method": "Normal blend before the tail; left-dominant overlap plus bright-edge-filled right-only pixels after the tail fade.",
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
