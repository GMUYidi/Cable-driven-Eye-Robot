#!/usr/bin/env python3
"""
Apply the repaired health_second canvas homographies to another left/right case.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from stitch_q3_hess_consistent import blend_warped_pair, matrix_frame_names, read_image, write_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def frame_number(stem: str) -> int:
    return int(stem.split("_")[-1])


def list_frames(input_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument(
        "--health-matrix-npz",
        type=Path,
        default=Path(
            r"D:\visualization2026\dowsample2_health_second\matrices\homography_matrices_and_points.npz"
        ),
    )
    parser.add_argument(
        "--health-stitched-dir",
        type=Path,
        default=Path(r"D:\visualization2026\dowsample2_health_second\stitched_frames"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for path in (args.left_dir, args.right_dir, args.health_matrix_npz, args.health_stitched_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_root / "stitched_frames"
    matrices_dir = args.output_root / "matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    matrix_output = matrices_dir / f"applied_health_second_homographies_{args.case_id}.npz"
    report_csv = args.output_root / "apply_health_second_h_report.csv"
    metadata_json = args.output_root / "apply_health_second_h_metadata.json"

    if any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_dir}")

    if args.overwrite:
        backup_file(matrix_output, "_before_repaired_health_second")
        backup_file(report_csv, "_before_repaired_health_second")
        backup_file(metadata_json, "_before_repaired_health_second")

    left_frames = list_frames(args.left_dir)
    right_frames = list_frames(args.right_dir)
    health_npz = np.load(args.health_matrix_npz, allow_pickle=True)
    stems = matrix_frame_names(health_npz)

    output_npz: dict[str, np.ndarray] = {}
    rows: list[dict[str, int | str | float]] = []
    written = 0
    for index, stem in enumerate(stems, start=1):
        if stem not in left_frames or stem not in right_frames:
            continue
        health_frame = args.health_stitched_dir / f"{stem}.jpg"
        if not health_frame.exists():
            continue

        left_image = read_image(left_frames[stem])
        right_image = read_image(right_frames[stem])
        health_image = read_image(health_frame)
        output_size = (health_image.shape[1], health_image.shape[0])

        canvas_left_h = health_npz[f"{stem}_canvas_left_h"].astype(np.float64)
        canvas_right_h = health_npz[f"{stem}_canvas_right_h"].astype(np.float64)
        stitched = blend_warped_pair(
            left_image,
            right_image,
            canvas_left_h,
            canvas_right_h,
            output_size,
        )
        output_path = output_dir / left_frames[stem].name
        write_image(output_path, stitched, args.jpeg_quality)

        output_npz[f"{stem}_canvas_left_h"] = canvas_left_h
        output_npz[f"{stem}_canvas_right_h"] = canvas_right_h
        output_npz[f"{stem}_health_output_size"] = np.array(output_size, dtype=np.int32)
        for suffix in (
            "green_repair_left_shift",
            "green_repair_right_shift",
            "green_repair_original_error",
            "green_repair_corrected_error",
        ):
            key = f"{stem}_{suffix}"
            if key in health_npz.files:
                output_npz[key] = health_npz[key]

        rows.append(
            {
                "frame": left_frames[stem].name,
                "status": "applied",
                "output_width": int(output_size[0]),
                "output_height": int(output_size[1]),
                "left_source": str(left_frames[stem]),
                "right_source": str(right_frames[stem]),
                "health_frame": str(health_frame),
            }
        )
        written += 1
        if args.progress_every > 0 and (
            written == 1 or written % args.progress_every == 0 or index == len(stems)
        ):
            print(f"[{written}] {stem}: {output_size[0]}x{output_size[1]}")

    if not rows:
        raise RuntimeError("No frames were written.")

    np.savez_compressed(matrix_output, **output_npz)
    with report_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "case_id": args.case_id,
        "left_dir": str(args.left_dir),
        "right_dir": str(args.right_dir),
        "health_matrix_npz": str(args.health_matrix_npz),
        "health_stitched_dir": str(args.health_stitched_dir),
        "output_dir": str(output_dir),
        "frame_count": len(rows),
        "matrix_output_npz": str(matrix_output),
        "report_csv": str(report_csv),
        "jpeg_quality": args.jpeg_quality,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} frames to {output_dir}")
    print(f"Matrix output: {matrix_output}")


if __name__ == "__main__":
    main()
