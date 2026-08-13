#!/usr/bin/env python3
"""
Increase Q45 horizontal Hess-matched image split from the angle trajectory.

The automatic Q45 fit underestimates the 5-10 s horizontal split. This script
takes the raw-green Q45 preview as the stable base and applies an additional
left/right symmetric canvas translation driven by SGazeAngleSmoothed_cell
horizontal L-R angle difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stitch_q3_hess_consistent import (
    blend_warped_pair,
    matrix_frame_names,
    read_image,
    translation_matrix,
    write_image,
)


DEFAULT_LEFT_DIR = Path(r"D:\visualization2026\visualization\left_downsample2_Q45")
DEFAULT_RIGHT_DIR = Path(r"D:\visualization2026\visualization\right_downsample2_Q45")
DEFAULT_PREVIEW_ROOT = Path(r"D:\visualization2026\dowsample2_Q45_hess_matched_rawgreen_preview")
DEFAULT_OUTPUT_ROOT = Path(r"D:\visualization2026\dowsample2_Q45_hess_matched")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def frame_number(stem: str) -> int:
    return int(stem.split("_")[-1])


def list_frames(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(folder.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def read_report(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows, {row["stem"]: row for row in rows}


def ramp_alpha(frame_index: int, ramp_start: int, full_start: int) -> float:
    if frame_index <= ramp_start:
        return 0.0
    if frame_index >= full_start:
        return 1.0
    return (frame_index - ramp_start) / max(1.0, float(full_start - ramp_start))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_RIGHT_DIR)
    parser.add_argument("--preview-root", type=Path, default=DEFAULT_PREVIEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", default="Q45")
    parser.add_argument("--horizontal-px-per-deg", type=float, default=35.0)
    parser.add_argument("--ramp-start-frame", type=int, default=540)
    parser.add_argument("--full-start-frame", type=int, default=600)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--backup-suffix", default="_before_q45_horizontal_scale_fix")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preview_npz_path = (
        args.preview_root / "matrices" / f"hess_matched_homographies_{args.case_id}.npz"
    )
    preview_report_path = args.preview_root / "hess_match_report.csv"
    preview_metadata_path = args.preview_root / "hess_match_metadata.json"
    preview_frames_dir = args.preview_root / "stitched_frames"

    output_frames_dir = args.output_root / "stitched_frames"
    output_matrices_dir = args.output_root / "matrices"
    output_npz_path = output_matrices_dir / f"hess_matched_homographies_{args.case_id}.npz"
    output_report_path = args.output_root / "hess_match_report.csv"
    output_metadata_path = args.output_root / "hess_match_metadata.json"
    debug_path = args.output_root / "debug" / "q45_horizontal_scale_fix_samples.jpg"

    for path in (
        args.left_dir,
        args.right_dir,
        preview_npz_path,
        preview_report_path,
        preview_metadata_path,
        preview_frames_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if output_frames_dir.exists() and any(output_frames_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_frames_dir}")
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    output_matrices_dir.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        backup_file(output_npz_path, args.backup_suffix)
        backup_file(output_report_path, args.backup_suffix)
        backup_file(output_metadata_path, args.backup_suffix)

    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    preview_npz = np.load(preview_npz_path, allow_pickle=True)
    preview_data = {key: preview_npz[key] for key in preview_npz.files}
    report_rows, rows_by_stem = read_report(preview_report_path)
    stems = [
        stem
        for stem in matrix_frame_names(preview_npz)
        if not stem.endswith("_base") and stem in rows_by_stem
    ]

    output_npz: dict[str, np.ndarray] = dict(preview_data)
    updated_rows: list[dict[str, Any]] = []
    sample_images: list[np.ndarray] = []
    sample_stems = {"frame_0540", "frame_0600", "frame_0720", "frame_0840", "frame_0960", "frame_1080", "frame_1200", "frame_1500", "frame_1798"}

    for index, stem in enumerate(stems, start=1):
        row = dict(rows_by_stem[stem])
        frame_index = frame_number(stem)
        alpha = ramp_alpha(frame_index, args.ramp_start_frame, args.full_start_frame)

        current_target = np.array(
            [float(row["target_dx"]), float(row["target_dy"])],
            dtype=np.float64,
        )
        angle_dx = float(row["angle_split_dx_deg"])
        scaled_dx = args.horizontal_px_per_deg * angle_dx
        if abs(scaled_dx) < abs(current_target[0]):
            scaled_dx = current_target[0]
        desired_target = np.array(
            [
                current_target[0] + alpha * (scaled_dx - current_target[0]),
                current_target[1],
            ],
            dtype=np.float64,
        )
        extra = desired_target - current_target

        left_h = preview_data[f"{stem}_canvas_left_h"].astype(np.float64)
        right_h = preview_data[f"{stem}_canvas_right_h"].astype(np.float64)
        corrected_left_h = translation_matrix(float(0.5 * extra[0]), float(0.5 * extra[1])) @ left_h
        corrected_right_h = translation_matrix(float(-0.5 * extra[0]), float(-0.5 * extra[1])) @ right_h

        left_image = read_image(left_frames[stem])
        right_image = read_image(right_frames[stem])
        preview_frame = read_image(preview_frames_dir / f"{stem}.jpg")
        out_h, out_w = preview_frame.shape[:2]
        stitched = blend_warped_pair(
            left_image,
            right_image,
            corrected_left_h,
            corrected_right_h,
            output_size=(out_w, out_h),
        )
        output_path = output_frames_dir / left_frames[stem].name
        write_image(output_path, stitched, args.jpeg_quality)

        output_npz[f"{stem}_canvas_left_h"] = corrected_left_h
        output_npz[f"{stem}_canvas_right_h"] = corrected_right_h
        output_npz[f"{stem}_q45_horizontal_scale_alpha"] = np.array(alpha, dtype=np.float64)
        output_npz[f"{stem}_q45_horizontal_scale_extra"] = extra.astype(np.float64)
        output_npz[f"{stem}_q45_horizontal_scaled_target"] = desired_target.astype(np.float64)

        row["target_dx_before_horizontal_scale"] = row["target_dx"]
        row["target_dy_before_horizontal_scale"] = row["target_dy"]
        row["target_dx"] = f"{desired_target[0]:.12g}"
        row["target_dy"] = f"{desired_target[1]:.12g}"
        row["horizontal_scale_alpha"] = f"{alpha:.6f}"
        row["horizontal_px_per_deg"] = f"{args.horizontal_px_per_deg:.6f}"
        row["horizontal_scale_extra_dx"] = f"{extra[0]:.12g}"
        row["horizontal_scale_extra_dy"] = f"{extra[1]:.12g}"
        row["output_path"] = str(output_path)
        updated_rows.append(row)

        if stem in sample_stems:
            small = cv2.resize(stitched, (520, 360), interpolation=cv2.INTER_AREA)
            label = (
                f"{stem} target=({desired_target[0]:.0f},{desired_target[1]:.0f}) "
                f"a={alpha:.2f}"
            )
            cv2.putText(
                small,
                label,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            sample_images.append(small)

        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(stems)
        ):
            print(
                f"[{index}/{len(stems)}] {stem}: "
                f"dx {current_target[0]:.1f}->{desired_target[0]:.1f}, alpha={alpha:.2f}"
            )

    np.savez_compressed(output_npz_path, **output_npz)

    if updated_rows:
        with output_report_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(updated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(updated_rows)

    preview_metadata = json.loads(preview_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **preview_metadata,
        "output_dir": str(output_frames_dir),
        "matrix_output_npz": str(output_npz_path),
        "report_csv": str(output_report_path),
        "debug_sheet": str(debug_path),
        "q45_horizontal_scale_fix": {
            "horizontal_px_per_deg": args.horizontal_px_per_deg,
            "ramp_start_frame": args.ramp_start_frame,
            "full_start_frame": args.full_start_frame,
            "source_preview_root": str(args.preview_root),
            "method": (
                "Additional symmetric canvas translation. Horizontal target "
                "uses max(auto target, horizontal_px_per_deg * SGazeAngle "
                "left-minus-right horizontal split), ramped in before 5 s."
            ),
        },
    }
    output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if sample_images:
        rows = []
        for i in range(0, len(sample_images), 3):
            group = sample_images[i : i + 3]
            while len(group) < 3:
                group.append(np.zeros_like(sample_images[0]))
            rows.append(np.hstack(group))
        write_image(debug_path, np.vstack(rows), args.jpeg_quality)

    print(f"Done. Wrote Q45 horizontal-scale frames to {output_frames_dir}")
    print(f"Matrices: {output_npz_path}")
    print(f"Debug: {debug_path}")


if __name__ == "__main__":
    main()
