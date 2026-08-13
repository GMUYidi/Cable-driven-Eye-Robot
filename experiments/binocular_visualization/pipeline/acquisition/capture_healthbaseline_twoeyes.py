from __future__ import annotations

import argparse
from pathlib import Path

from replay_healthbaseline_twoeyes import (
    DEFAULT_CAMERA_BAUD,
    DEFAULT_DXL_BAUD,
    DEFAULT_DXL_PORT,
    DEFAULT_FRAME_DIR,
    DEFAULT_JOURNAL_DIR,
    DEFAULT_LEFT_CAMERA_PORT,
    DEFAULT_LEFT_SAVE_DIR,
    DEFAULT_MAT_PATH,
    DEFAULT_RIGHT_CAMERA_PORT,
    DEFAULT_RIGHT_SAVE_DIR,
    FINE_ANGLE_TOLERANCE_DEG,
    IMU_ZERO_SAMPLES,
    run_capture_replay,
)

# Change this when running directly in Spyder/runfile.
# Valid range: 0-1799. The first target gaze and first displayed frame will
# start from this original healthbaseline/frame index.
CAPTURE_START_INDEX = 0

# Change this to preview a downsampled route. Valid values: 1, 2, 3, 6, 10.
# 1 means the original full 1800-frame capture. 10 means 0, 10, 20, ...
CAPTURE_DOWNSAMPLE_STEP = 2
ALLOWED_DOWNSAMPLE_STEPS = (1, 2, 3, 6, 10)

# Fine tuning knobs for direct Spyder/runfile use.
CAPTURE_FINE_TOLERANCE_DEG = FINE_ANGLE_TOLERANCE_DEG
CAPTURE_FIRST_FINE_TIMEOUT_S = 60.0
CAPTURE_LATER_FINE_TIMEOUT_S = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay healthbaseline gaze targets and capture the corresponding "
            "fullscreen frame with both eye cameras."
        )
    )
    parser.add_argument("--mat-path", type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument("--mat-key", default="SGazeAngleSmoothed_cell")
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--left-save-dir", type=Path)
    parser.add_argument("--right-save-dir", type=Path)
    parser.add_argument("--left-camera-port", default=DEFAULT_LEFT_CAMERA_PORT)
    parser.add_argument("--right-camera-port", default=DEFAULT_RIGHT_CAMERA_PORT)
    parser.add_argument("--camera-baud", type=int, default=DEFAULT_CAMERA_BAUD)
    parser.add_argument("--dxl-port", default=DEFAULT_DXL_PORT)
    parser.add_argument("--dxl-baud", type=int, default=DEFAULT_DXL_BAUD)
    parser.add_argument("--start-index", type=int, default=CAPTURE_START_INDEX)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--sample-step",
        type=int,
        choices=ALLOWED_DOWNSAMPLE_STEPS,
        default=CAPTURE_DOWNSAMPLE_STEP,
    )
    parser.add_argument(
        "--first-fine-timeout-s",
        type=float,
        default=CAPTURE_FIRST_FINE_TIMEOUT_S,
    )
    parser.add_argument("--fine-timeout-s", type=float, default=CAPTURE_LATER_FINE_TIMEOUT_S)
    parser.add_argument("--fine-tolerance-deg", type=float, default=CAPTURE_FINE_TOLERANCE_DEG)
    parser.add_argument("--display-settle-s", type=float, default=0.05)
    parser.add_argument("--first-frame-extra-settle-s", type=float, default=3.0)
    parser.add_argument(
        "--display-index",
        type=int,
        default=1,
        help="Pygame display index for fullscreen frames. Usually 0=main, 1=extended.",
    )
    parser.add_argument("--capture-retries", type=int, default=1)
    parser.add_argument("--feedback-print-every-steps", type=int, default=0)
    parser.add_argument("--imu-zero-samples", type=int, default=IMU_ZERO_SAMPLES)
    parser.add_argument(
        "--print-calibration",
        action="store_true",
        help="Print IMU zero calibration values during startup.",
    )
    parser.add_argument(
        "--print-tuning",
        action="store_true",
        help="Print coarse/fine tuning details during capture.",
    )
    parser.add_argument(
        "--no-bring-display-to-front",
        dest="bring_display_to_front",
        action="store_false",
        help="Open the fullscreen display without forcing the window to the foreground.",
    )
    parser.set_defaults(bring_display_to_front=True)
    parser.add_argument(
        "--no-reset-perfect",
        action="store_true",
        help="Use current motor positions as baseline instead of moving to PERFECT_POSITIONS.",
    )
    parser.add_argument(
        "--no-imu-zero",
        action="store_true",
        help="Use absolute IMU yaw/pitch/roll instead of zeroing both IMUs at startup.",
    )
    parser.add_argument(
        "--no-joystick-emergency",
        action="store_true",
        help="Disable triangle-button emergency return-to-initial handling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and IK pulses without opening motors, cameras, or fullscreen display.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help="Folder for the human-readable per-capture journal.",
    )
    args = parser.parse_args()
    if args.start_index < 0 or args.start_index > 1799:
        parser.error("--start-index must be in the range 0-1799")
    if args.sample_step not in ALLOWED_DOWNSAMPLE_STEPS:
        parser.error("--sample-step must be one of 1, 2, 3, 6, 10")
    if args.fine_timeout_s <= 0:
        parser.error("--fine-timeout-s must be positive")
    if args.fine_tolerance_deg <= 0:
        parser.error("--fine-tolerance-deg must be positive")

    if args.left_save_dir is None:
        args.left_save_dir = (
            DEFAULT_LEFT_SAVE_DIR
            if args.sample_step == 1
            else Path(f"D:/visualization2026/left_downsample{args.sample_step}")
        )
    if args.right_save_dir is None:
        args.right_save_dir = (
            DEFAULT_RIGHT_SAVE_DIR
            if args.sample_step == 1
            else Path(f"D:/visualization2026/right_downsample{args.sample_step}")
        )
    if args.log_path is None:
        args.log_path = (
            Path("D:/visualization2026/capture_replay_log.csv")
            if args.sample_step == 1
            else Path(f"D:/visualization2026/capture_replay_downsample{args.sample_step}_log.csv")
        )
    return args


def main() -> int:
    run_capture_replay(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
