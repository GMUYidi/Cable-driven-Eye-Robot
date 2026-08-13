from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import scipy.io as scio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_VIDEO = Path(
    r"D:\visualization2026\dowsample2_Q23"
    r"\stitched_frames_flipped_cropped_60fps.mp4"
)
DEFAULT_DATA_FILE = (
    SCRIPT_DIR / "strabismus data" / "simulated_strabismus_Q23.mat"
)
DEFAULT_OUTPUT_VIDEO = (
    DEFAULT_INPUT_VIDEO.parent
    / "stitched_frames_flipped_cropped_60fps_with_gaze_Q23_lead0p5s.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place an animated simulated gaze-angle trajectory in the "
            "bottom-right corner of a video."
        )
    )
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--mat-key", default="SGazeAngleSmoothed_cell")
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--overlay-width", type=int, default=480)
    parser.add_argument("--overlay-height", type=int, default=360)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument(
        "--animation-lead-s",
        type=float,
        default=0.5,
        help=(
            "Advance the gaze animation by this many seconds to compensate "
            "for stationary samples at the start."
        ),
    )
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} was not found on PATH.")
    return executable


def probe_video(video_path: Path, ffprobe: str) -> dict[str, float | int]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stream = json.loads(result.stdout)["streams"][0]
    frame_rate = float(Fraction(stream["avg_frame_rate"]))
    duration = float(stream["duration"])
    frame_count = int(stream.get("nb_frames") or round(duration * frame_rate))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "duration": duration,
    }


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0

    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    value_range = value_max - value_min
    padding = max(abs(value_min) * 0.05, 1.0) if value_range == 0 else value_range * 0.05
    return value_min - padding, value_max + padding


def load_simulated_angles(data_file: Path, mat_key: str) -> np.ndarray:
    with data_file.open("rb") as file:
        data = scio.loadmat(file)

    if mat_key not in data:
        available = ", ".join(key for key in data if not key.startswith("__"))
        raise KeyError(
            f"{mat_key!r} was not found in {data_file}. Available: {available}"
        )

    angles = np.asarray(data[mat_key], dtype=float)
    if angles.ndim != 2 or angles.shape[1] < 4:
        raise ValueError(f"{mat_key} must be an N-by-4 array, got {angles.shape}.")
    if angles.shape[0] == 0:
        raise ValueError(f"{mat_key} is empty.")
    return angles


def render_overlay_video(
    angles: np.ndarray,
    output_path: Path,
    frame_rate: float,
    frame_count: int,
    width: int,
    height: int,
    animation_lead_s: float,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Overlay width and height must be positive.")

    left_angles = angles[:, 0:2]
    right_angles = angles[:, 2:4]

    dpi = 100
    figure = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    axes = figure.add_subplot(111)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.88)

    left_line, = axes.plot(
        [],
        [],
        color=(0.85, 0.1, 0.1),
        linewidth=1.6,
        label="Simulated left",
    )
    right_line, = axes.plot(
        [],
        [],
        color=(0.1, 0.25, 0.85),
        linewidth=1.6,
        label="Simulated right",
    )

    axes.set_title("Gaze angle trajectory", fontsize=10)
    axes.set_xlabel("Horizontal gaze angle", fontsize=8)
    axes.set_ylabel("Vertical gaze angle", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.grid(True, linewidth=0.5, alpha=0.55)
    axes.set_aspect("equal", adjustable="box")
    axes.legend(loc="center", fontsize=7, framealpha=0.9)

    all_horizontal = np.concatenate((left_angles[:, 0], right_angles[:, 0]))
    all_vertical = np.concatenate((left_angles[:, 1], right_angles[:, 1]))
    axes.set_xlim(*padded_limits(all_horizontal))
    axes.set_ylim(*padded_limits(all_vertical))
    axes.invert_yaxis()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        plt.close(figure)
        raise RuntimeError(f"Could not create temporary overlay video: {output_path}")

    sample_count = angles.shape[0]
    sample_rate = sample_count / (frame_count / frame_rate)
    lead_samples = round(animation_lead_s * sample_rate)
    progress_interval = max(frame_count // 10, 1)

    try:
        for frame_index in range(frame_count):
            video_time_s = frame_index / frame_rate
            source_time_s = video_time_s + animation_lead_s
            last_sample = 1 + round(source_time_s * sample_rate)
            last_sample = min(max(last_sample, 1), sample_count)

            left_line.set_data(
                left_angles[:last_sample, 0],
                left_angles[:last_sample, 1],
            )
            right_line.set_data(
                right_angles[:last_sample, 0],
                right_angles[:last_sample, 1],
            )

            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            cv2.rectangle(bgr, (0, 0), (width - 1, height - 1), (45, 45, 45), 2)
            writer.write(bgr)

            if (
                frame_index == 0
                or frame_index + 1 == frame_count
                or (frame_index + 1) % progress_interval == 0
            ):
                print(f"Rendered overlay frame {frame_index + 1}/{frame_count}")
        print(
            f"Animation lead: {animation_lead_s:.3f} s "
            f"(approximately {lead_samples} trajectory samples)"
        )
    finally:
        writer.release()
        plt.close(figure)


def compose_video(
    input_video: Path,
    overlay_video: Path,
    output_video: Path,
    margin: int,
    crf: int,
    ffmpeg: str,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"[1:v]setsar=1[overlay];"
        f"[0:v][overlay]overlay=W-w-{margin}:H-h-{margin}:shortest=1[outv]"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(overlay_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")

    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input_video}")
    if not args.data_file.is_file():
        raise FileNotFoundError(f"MAT file does not exist: {args.data_file}")
    if args.margin < 0:
        raise ValueError("Margin cannot be negative.")
    if args.animation_lead_s < 0:
        raise ValueError("Animation lead cannot be negative.")

    metadata = probe_video(args.input_video, ffprobe)
    if (
        args.overlay_width + args.margin > metadata["width"]
        or args.overlay_height + args.margin > metadata["height"]
    ):
        raise ValueError("The overlay and margin do not fit inside the source video.")

    angles = load_simulated_angles(args.data_file, args.mat_key)
    print(
        "Source video: "
        f"{metadata['width']}x{metadata['height']}, "
        f"{metadata['frame_rate']:.3f} fps, "
        f"{metadata['frame_count']} frames, "
        f"{metadata['duration']:.3f} s"
    )
    print(f"Trajectory samples: {angles.shape[0]}")

    with tempfile.TemporaryDirectory(prefix="gaze_overlay_") as temp_dir:
        overlay_path = Path(temp_dir) / "gaze_overlay.mp4"
        render_overlay_video(
            angles=angles,
            output_path=overlay_path,
            frame_rate=float(metadata["frame_rate"]),
            frame_count=int(metadata["frame_count"]),
            width=args.overlay_width,
            height=args.overlay_height,
            animation_lead_s=args.animation_lead_s,
        )
        compose_video(
            input_video=args.input_video,
            overlay_video=overlay_path,
            output_video=args.output_video,
            margin=args.margin,
            crf=args.crf,
            ffmpeg=ffmpeg,
        )

    print(f"Created: {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
