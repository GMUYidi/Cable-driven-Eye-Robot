#!/usr/bin/env python3
"""Apply a direct visual Hess-direction correction to Q42 stitched frames.

This is a second-pass refinement on top of the marker-refined Q42 matrices.
The previous pass made the measured marker vector match a fitted image-space
mapping. This pass instead makes the final flipped/cropped video-space L-R
marker direction follow the Hess overlay direction directly:

    final_video_L_minus_R = (scale_x * dH, scale_y * dV)

where dH/dV are taken from SGazeAngleSmoothed_cell as left minus right.
Because the published frames are vertically flipped before video generation,
the requested final-video vector is converted back to the unflipped stitched
canvas before it is applied to the homographies.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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


def read_csv_by_frame(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["frame"]: row for row in csv.DictReader(handle)}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def backup_file(path: Path, suffix: str) -> None:
    if path.exists():
        backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not backup.exists():
            shutil.copy2(path, backup)


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
    parser = argparse.ArgumentParser(
        description="Directly align Q42 final-video green-dot direction to the Hess trajectory."
    )
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
        "--input-matrix-npz",
        type=Path,
        default=root / "matrices" / "hess_matched_homographies_Q42.npz",
    )
    parser.add_argument("--input-report-csv", type=Path, default=root / "hess_match_report.csv")
    parser.add_argument("--flip-crop-report-csv", type=Path, default=root / "flip_crop_report.csv")
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "stitched_frames")
    parser.add_argument(
        "--matrix-output-dir",
        type=Path,
        default=root / "matrices",
    )
    parser.add_argument("--report-csv", type=Path, default=root / "hess_match_report.csv")
    parser.add_argument("--metadata-json", type=Path, default=root / "hess_match_metadata.json")
    parser.add_argument("--case-id", default="Q42")
    parser.add_argument("--final-width", type=int, default=1920)
    parser.add_argument("--final-height", type=int, default=1080)
    parser.add_argument("--scale-x", type=float, default=44.0)
    parser.add_argument("--scale-y", type=float, default=18.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for folder in (args.left_dir, args.right_dir):
        if not folder.exists():
            raise FileNotFoundError(folder)
    for file_path in (args.input_matrix_npz, args.input_report_csv, args.flip_crop_report_csv):
        if not file_path.exists():
            raise FileNotFoundError(file_path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matrix_output_dir.mkdir(parents=True, exist_ok=True)
    backup_file(args.report_csv, "_before_direct_visual_refine")
    backup_file(args.metadata_json, "_before_direct_visual_refine")
    backup_file(args.matrix_output_dir / f"hess_matched_homographies_{args.case_id}.npz", "_before_direct_visual_refine")

    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    matrices = np.load(args.input_matrix_npz, allow_pickle=True)
    previous_rows = read_csv_by_frame(args.input_report_csv)
    crop_rows = read_csv_by_frame(args.flip_crop_report_csv)
    stems = [
        stem
        for stem in matrix_frame_names(matrices)
        if f"{stem}.jpg" in previous_rows and f"{stem}.jpg" in crop_rows
    ]
    if not stems:
        raise RuntimeError("No real frame stems were found in the matrix/report intersection.")

    output_npz: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    visual_errors: list[float] = []
    added_norms: list[float] = []

    for index, stem in enumerate(stems, start=1):
        frame_name = f"{stem}.jpg"
        if frame_name not in previous_rows:
            raise KeyError(f"Missing report row for {frame_name}")
        if frame_name not in crop_rows:
            raise KeyError(f"Missing flip/crop row for {frame_name}")
        if stem not in left_frames or stem not in right_frames:
            raise KeyError(f"Missing raw frame pair for {stem}")

        previous = previous_rows[frame_name]
        crop = crop_rows[frame_name]
        crop_width = as_float(crop, "crop_width")
        crop_height = as_float(crop, "crop_height")
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(f"Invalid crop size for {frame_name}")

        sx = args.final_width / crop_width
        sy = args.final_height / crop_height
        angle_dx = as_float(previous, "angle_split_dx_deg")
        angle_dy = as_float(previous, "angle_split_dy_deg")

        desired_final_vec = np.array(
            [args.scale_x * angle_dx, args.scale_y * angle_dy],
            dtype=np.float64,
        )
        desired_preflip_vec = np.array(
            [desired_final_vec[0] / sx, -desired_final_vec[1] / sy],
            dtype=np.float64,
        )
        current_preflip_vec = np.array(
            [as_float(previous, "target_dx"), as_float(previous, "target_dy")],
            dtype=np.float64,
        )
        additional_correction = desired_preflip_vec - current_preflip_vec
        left_add = 0.5 * additional_correction
        right_add = -0.5 * additional_correction

        base_left_h = matrices[f"{stem}_canvas_left_h"].astype(np.float64)
        base_right_h = matrices[f"{stem}_canvas_right_h"].astype(np.float64)
        corrected_left_h = translation_matrix(float(left_add[0]), float(left_add[1])) @ base_left_h
        corrected_right_h = translation_matrix(float(right_add[0]), float(right_add[1])) @ base_right_h

        base_stitched_path = Path(previous["output_path"])
        if not base_stitched_path.exists():
            base_stitched_path = args.output_dir / frame_name
        if not base_stitched_path.exists():
            raise FileNotFoundError(f"Need stitched frame for canvas size: {base_stitched_path}")
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

        final_visual_vec = np.array(
            [desired_preflip_vec[0] * sx, -desired_preflip_vec[1] * sy],
            dtype=np.float64,
        )
        visual_error = float(np.linalg.norm(final_visual_vec - desired_final_vec))
        visual_errors.append(visual_error)
        added_norms.append(float(np.linalg.norm(additional_correction)))

        output_npz[f"{stem}_canvas_left_h"] = corrected_left_h
        output_npz[f"{stem}_canvas_right_h"] = corrected_right_h
        output_npz[f"{stem}_input_canvas_left_h"] = base_left_h
        output_npz[f"{stem}_input_canvas_right_h"] = base_right_h
        output_npz[f"{stem}_direct_visual_target_vec"] = desired_final_vec
        output_npz[f"{stem}_direct_preflip_target_vec"] = desired_preflip_vec
        output_npz[f"{stem}_additional_correction"] = additional_correction
        output_npz[f"{stem}_left_direct_translation"] = left_add
        output_npz[f"{stem}_right_direct_translation"] = right_add

        row: dict[str, Any] = dict(previous)
        row.update(
            {
                "direct_visual_scale_x": args.scale_x,
                "direct_visual_scale_y": args.scale_y,
                "direct_visual_target_dx": float(desired_final_vec[0]),
                "direct_visual_target_dy": float(desired_final_vec[1]),
                "direct_preflip_target_dx": float(desired_preflip_vec[0]),
                "direct_preflip_target_dy": float(desired_preflip_vec[1]),
                "previous_target_dx": float(current_preflip_vec[0]),
                "previous_target_dy": float(current_preflip_vec[1]),
                "target_dx": float(desired_preflip_vec[0]),
                "target_dy": float(desired_preflip_vec[1]),
                "additional_correction_dx": float(additional_correction[0]),
                "additional_correction_dy": float(additional_correction[1]),
                "additional_correction_norm": float(np.linalg.norm(additional_correction)),
                "left_translation_x": float(as_float(previous, "left_translation_x") + left_add[0]),
                "left_translation_y": float(as_float(previous, "left_translation_y") + left_add[1]),
                "right_translation_x": float(as_float(previous, "right_translation_x") + right_add[0]),
                "right_translation_y": float(as_float(previous, "right_translation_y") + right_add[1]),
                "final_visual_dx": float(final_visual_vec[0]),
                "final_visual_dy": float(final_visual_vec[1]),
                "final_visual_error_to_direct_target": visual_error,
                "direct_visual_refined": 1,
            }
        )
        row.update(flatten_matrix("canvas_left_h", corrected_left_h))
        row.update(flatten_matrix("canvas_right_h", corrected_right_h))
        rows.append(row)

        if index == 1 or index % args.progress_every == 0 or index == len(stems):
            print(
                f"[{index}/{len(stems)}] {frame_name}: "
                f"direct_final=({desired_final_vec[0]:.1f},{desired_final_vec[1]:.1f}) "
                f"add=({additional_correction[0]:.1f},{additional_correction[1]:.1f})"
            )

    matrix_out = args.matrix_output_dir / f"hess_matched_homographies_{args.case_id}.npz"
    output_npz["direct_visual_scale"] = np.array([args.scale_x, args.scale_y], dtype=np.float64)
    output_npz["frame_names"] = np.array(stems)
    np.savez_compressed(matrix_out, **output_npz)
    write_csv(args.report_csv, rows)

    metadata = {
        "case_id": args.case_id,
        "input_matrix_npz": str(args.input_matrix_npz),
        "input_report_csv": str(args.input_report_csv),
        "flip_crop_report_csv": str(args.flip_crop_report_csv),
        "output_dir": str(args.output_dir),
        "matrix_output_npz": str(matrix_out),
        "report_csv": str(args.report_csv),
        "frame_count": len(rows),
        "direct_visual_scale_x": args.scale_x,
        "direct_visual_scale_y": args.scale_y,
        "final_visual_error_median_px": float(np.median(visual_errors)),
        "final_visual_error_max_px": float(np.max(visual_errors)),
        "additional_correction_median_px": float(np.median(added_norms)),
        "additional_correction_p90_px": float(np.percentile(added_norms, 90)),
        "additional_correction_max_px": float(np.max(added_norms)),
        "method": (
            "Second-pass split translations applied to Q42 homographies so the "
            "final flipped/cropped video-space left-minus-right marker vector "
            "follows SGazeAngleSmoothed_cell left-minus-right direction directly."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} frames to {args.output_dir}")
    print(f"Matrices: {matrix_out}")
    print(f"Report: {args.report_csv}")


if __name__ == "__main__":
    main()
