#!/usr/bin/env python3
"""Rebuild Q42 with Q23-style smooth homography corrections.

The direct Q42 pass matched the Hess vector, but it inherited noisy per-frame
green-marker reference jumps. Those jumps made the homography translations
change too abruptly and created visible camera shake. This script keeps the
Hess target vector, but smooths the inferred base marker vector first, then
applies the resulting split translation to the repaired health_second base
homographies.
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
    blend_warped_pair,
    flatten_matrix,
    list_frames,
    matrix_frame_names,
    read_image,
    translation_matrix,
    write_image,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_crop_report(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: dict[str, dict[str, float]] = {}
        for row in csv.DictReader(handle):
            rows[row["frame"]] = {key: float(value) for key, value in row.items() if key != "frame"}
        return rows


def moving_average(values: np.ndarray, window: int, iterations: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = values.copy()
    for _ in range(max(1, iterations)):
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.empty_like(out, dtype=np.float64)
        for dim in range(out.shape[1]):
            smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        out = smoothed
    return out


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def tail_adjusted_target(
    target: np.ndarray,
    frame_index: int,
    enabled: bool,
    fade_start_frame: int,
    full_frame: int,
    horizontal_scale: float,
    vertical_scale: float,
    max_vertical_abs: float,
) -> tuple[np.ndarray, float]:
    if not enabled:
        return target.copy(), 0.0
    alpha = smoothstep(float(fade_start_frame), float(full_frame), float(frame_index))
    damped = np.array(
        [
            target[0] * horizontal_scale,
            float(np.clip(target[1] * vertical_scale, -max_vertical_abs, max_vertical_abs)),
        ],
        dtype=np.float64,
    )
    return (1.0 - alpha) * target + alpha * damped, alpha


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
    parser = argparse.ArgumentParser(description="Smooth Q42 homographies in the style of Q23.")
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
    parser.add_argument(
        "--base-matrix-npz",
        type=Path,
        default=Path(r"D:\visualization2026\dowsample2_Q42\matrices\applied_health_second_homographies_Q42.npz"),
    )
    parser.add_argument(
        "--base-stitched-dir",
        type=Path,
        default=Path(r"D:\visualization2026\dowsample2_Q42\stitched_frames"),
    )
    parser.add_argument("--reference-report-csv", type=Path, default=root / "hess_match_report.csv")
    parser.add_argument("--target-crop-report-csv", type=Path, default=root / "flip_crop_report.csv")
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "stitched_frames_q23style")
    parser.add_argument("--matrix-output-dir", type=Path, default=root / "matrices")
    parser.add_argument("--matrix-name", default="hess_matched_homographies_Q42_q23style.npz")
    parser.add_argument("--report-csv", type=Path, default=root / "hess_match_report_q23style.csv")
    parser.add_argument("--metadata-json", type=Path, default=root / "hess_match_metadata_q23style.json")
    parser.add_argument("--case-id", default="Q42")
    parser.add_argument("--smooth-window", type=int, default=61)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--tail-override", action="store_true")
    parser.add_argument("--tail-fade-start-frame", type=int, default=600)
    parser.add_argument("--tail-full-frame", type=int, default=690)
    parser.add_argument("--tail-horizontal-scale", type=float, default=0.0)
    parser.add_argument("--tail-vertical-scale", type=float, default=0.35)
    parser.add_argument("--tail-max-vertical-abs", type=float, default=18.0)
    parser.add_argument(
        "--tail-shift-full-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the split homography translation after the tail fade. "
            "Use 0 to return fully to the base health homography in the tail."
        ),
    )
    parser.add_argument("--final-width", type=int, default=1920)
    parser.add_argument("--final-height", type=int, default=1080)
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
    base_matrices = np.load(args.base_matrix_npz, allow_pickle=True)
    stems = matrix_frame_names(base_matrices)
    reference_rows = read_csv_rows(args.reference_report_csv)
    reference_by_frame = {row["frame"]: row for row in reference_rows}
    crop_rows = read_crop_report(args.target_crop_report_csv)

    active_stems = [
        stem
        for stem in stems
        if f"{stem}.jpg" in reference_by_frame
        and f"{stem}.jpg" in crop_rows
        and stem in left_frames
        and stem in right_frames
    ]
    if not active_stems:
        raise RuntimeError("No active stems found.")

    ref_ordered = [reference_by_frame[f"{stem}.jpg"] for stem in active_stems]
    total_left = np.array(
        [[float(row["left_translation_x"]), float(row["left_translation_y"])] for row in ref_ordered],
        dtype=np.float64,
    )
    total_right = np.array(
        [[float(row["right_translation_x"]), float(row["right_translation_y"])] for row in ref_ordered],
        dtype=np.float64,
    )
    current_preflip_target = np.array(
        [[float(row["target_dx"]), float(row["target_dy"])] for row in ref_ordered],
        dtype=np.float64,
    )
    noisy_base_vec = current_preflip_target - (total_left - total_right)
    smooth_base_vec = moving_average(noisy_base_vec, args.smooth_window, args.smooth_iterations)

    output_npz: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    left_steps: list[float] = []
    visual_errors: list[float] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matrix_output_dir.mkdir(parents=True, exist_ok=True)

    previous_left_shift: np.ndarray | None = None
    for index, stem in enumerate(active_stems, start=1):
        frame_name = f"{stem}.jpg"
        ref = reference_by_frame[frame_name]
        crop = crop_rows[frame_name]
        sx = args.final_width / float(crop["crop_width"])
        sy = args.final_height / float(crop["crop_height"])
        desired_visual = np.array(
            [float(ref["direct_visual_target_dx"]), float(ref["direct_visual_target_dy"])],
            dtype=np.float64,
        )
        original_visual = desired_visual.copy()
        # The rendered video contains one stitched frame per row. Use the video
        # frame index here, not the raw source frame number embedded in
        # "frame_####", because Q42 source frames are downsampled by 2.
        frame_index = index - 1
        desired_visual, tail_alpha = tail_adjusted_target(
            desired_visual,
            frame_index=frame_index,
            enabled=args.tail_override,
            fade_start_frame=args.tail_fade_start_frame,
            full_frame=args.tail_full_frame,
            horizontal_scale=args.tail_horizontal_scale,
            vertical_scale=args.tail_vertical_scale,
            max_vertical_abs=args.tail_max_vertical_abs,
        )
        desired_preflip = np.array([desired_visual[0] / sx, -desired_visual[1] / sy], dtype=np.float64)
        left_shift = 0.5 * (desired_preflip - smooth_base_vec[index - 1])
        if args.tail_override:
            shift_scale = (1.0 - tail_alpha) + tail_alpha * args.tail_shift_full_scale
            left_shift = left_shift * shift_scale
        right_shift = -left_shift
        if previous_left_shift is not None:
            left_steps.append(float(np.linalg.norm(left_shift - previous_left_shift)))
        previous_left_shift = left_shift.copy()

        base_left_h = base_matrices[f"{stem}_canvas_left_h"].astype(np.float64)
        base_right_h = base_matrices[f"{stem}_canvas_right_h"].astype(np.float64)
        corrected_left_h = translation_matrix(float(left_shift[0]), float(left_shift[1])) @ base_left_h
        corrected_right_h = translation_matrix(float(right_shift[0]), float(right_shift[1])) @ base_right_h

        base_stitched_path = args.base_stitched_dir / frame_name
        if not base_stitched_path.exists():
            raise FileNotFoundError(base_stitched_path)
        base_stitched = read_image(base_stitched_path)
        out_h, out_w = base_stitched.shape[:2]
        stitched = blend_warped_pair(
            read_image(left_frames[stem]),
            read_image(right_frames[stem]),
            corrected_left_h,
            corrected_right_h,
            output_size=(out_w, out_h),
        )
        output_path = args.output_dir / frame_name
        write_image(output_path, stitched, args.jpeg_quality)

        final_preflip = smooth_base_vec[index - 1] + (left_shift - right_shift)
        final_visual = np.array([final_preflip[0] * sx, -final_preflip[1] * sy], dtype=np.float64)
        visual_error = float(np.linalg.norm(final_visual - desired_visual))
        visual_errors.append(visual_error)

        output_npz[f"{stem}_canvas_left_h"] = corrected_left_h
        output_npz[f"{stem}_canvas_right_h"] = corrected_right_h
        output_npz[f"{stem}_base_canvas_left_h"] = base_left_h
        output_npz[f"{stem}_base_canvas_right_h"] = base_right_h
        output_npz[f"{stem}_smooth_base_left_minus_right_vec"] = smooth_base_vec[index - 1]
        output_npz[f"{stem}_target_left_minus_right_vec"] = desired_preflip
        output_npz[f"{stem}_direct_visual_target_vec"] = desired_visual
        output_npz[f"{stem}_left_translation"] = left_shift
        output_npz[f"{stem}_right_translation"] = right_shift

        row: dict[str, Any] = dict(ref)
        row.update(
            {
                "output_path": str(output_path),
                "q23style_smoothed": 1,
                "smooth_window": args.smooth_window,
                "smooth_iterations": args.smooth_iterations,
                "target_dx": float(desired_preflip[0]),
                "target_dy": float(desired_preflip[1]),
                "smooth_base_dx": float(smooth_base_vec[index - 1, 0]),
                "smooth_base_dy": float(smooth_base_vec[index - 1, 1]),
                "left_translation_x": float(left_shift[0]),
                "left_translation_y": float(left_shift[1]),
                "right_translation_x": float(right_shift[0]),
                "right_translation_y": float(right_shift[1]),
                "final_visual_dx": float(final_visual[0]),
                "final_visual_dy": float(final_visual[1]),
                "final_visual_error_to_direct_target": visual_error,
                "original_direct_visual_target_dx": float(original_visual[0]),
                "original_direct_visual_target_dy": float(original_visual[1]),
                "direct_visual_target_dx": float(desired_visual[0]),
                "direct_visual_target_dy": float(desired_visual[1]),
                "tail_override_alpha": float(tail_alpha),
                "tail_shift_full_scale": float(args.tail_shift_full_scale),
            }
        )
        row.update(flatten_matrix("canvas_left_h", corrected_left_h))
        row.update(flatten_matrix("canvas_right_h", corrected_right_h))
        rows.append(row)

        if index == 1 or index % args.progress_every == 0 or index == len(active_stems):
            print(
                f"[{index}/{len(active_stems)}] {frame_name}: "
                f"Lshift=({left_shift[0]:.1f},{left_shift[1]:.1f}) "
                f"target=({desired_visual[0]:.1f},{desired_visual[1]:.1f}) "
                f"err={visual_error:.2f}"
            )

    matrix_out = args.matrix_output_dir / args.matrix_name
    output_npz["frame_names"] = np.array(active_stems)
    output_npz["smooth_window"] = np.array([args.smooth_window], dtype=np.int32)
    output_npz["smooth_iterations"] = np.array([args.smooth_iterations], dtype=np.int32)
    output_npz["smooth_base_vectors"] = smooth_base_vec
    np.savez_compressed(matrix_out, **output_npz)
    write_csv(args.report_csv, rows)

    left_steps_arr = np.array(left_steps, dtype=np.float64)
    visual_errors_arr = np.array(visual_errors, dtype=np.float64)
    metadata = {
        "case_id": args.case_id,
        "base_matrix_npz": str(args.base_matrix_npz),
        "reference_report_csv": str(args.reference_report_csv),
        "target_crop_report_csv": str(args.target_crop_report_csv),
        "output_dir": str(args.output_dir),
        "matrix_output_npz": str(matrix_out),
        "report_csv": str(args.report_csv),
        "frame_count": len(rows),
        "smooth_window": args.smooth_window,
        "smooth_iterations": args.smooth_iterations,
        "tail_override": bool(args.tail_override),
        "tail_fade_start_frame": args.tail_fade_start_frame,
        "tail_full_frame": args.tail_full_frame,
        "tail_horizontal_scale": args.tail_horizontal_scale,
        "tail_vertical_scale": args.tail_vertical_scale,
        "tail_max_vertical_abs": args.tail_max_vertical_abs,
        "tail_shift_full_scale": args.tail_shift_full_scale,
        "left_translation_step_median_px": float(np.median(left_steps_arr)),
        "left_translation_step_p90_px": float(np.percentile(left_steps_arr, 90)),
        "left_translation_step_max_px": float(np.max(left_steps_arr)),
        "final_visual_error_median_px": float(np.median(visual_errors_arr)),
        "final_visual_error_p90_px": float(np.percentile(visual_errors_arr, 90)),
        "final_visual_error_max_px": float(np.max(visual_errors_arr)),
        "method": (
            "Q23-style: repaired health_second base homographies plus split translations "
            "computed from a temporally smoothed inferred base marker vector and the direct Hess target."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} frames to {args.output_dir}")
    print(f"Matrices: {matrix_out}")
    print(f"Report: {args.report_csv}")
    print(f"Metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
