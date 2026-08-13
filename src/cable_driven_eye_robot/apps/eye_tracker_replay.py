from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from ..config import CameraConfig, RobotConfig
from ..data.eye_tracker import EyeTrackerTarget, load_eye_tracker_targets
from ..hardware.dynamixel import DynamixelController
from ..kinematics import motor_pulses_from_pose
from ..sensors.frame_capture import StillFrameCapture


CSV_FIELDS = [
    "target_index",
    "timestamp_s",
    "left_yaw_deg",
    "left_pitch_deg",
    "left_roll_deg",
    "right_yaw_deg",
    "right_pitch_deg",
    "right_roll_deg",
    "safety_limited",
    "left_image",
    "right_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay eye-tracker gaze targets with open-loop inverse kinematics and capture camera frames."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mat-key", default="GazeAngleSmoothed_cell")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=500)
    parser.add_argument("--target-sign", type=float, default=-1.0)
    parser.add_argument("--dxl-port", default="COM12")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--left-camera", default="http://192.168.4.1/stream")
    parser.add_argument("--right-camera", default="http://192.168.4.200/stream")
    parser.add_argument("--settle-time-s", type=float, default=0.35)
    parser.add_argument("--camera-warmup-frames", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eye_tracker_replay"))
    parser.add_argument("--no-motors", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RobotConfig:
    base = RobotConfig()
    return replace(
        base,
        dynamixel=replace(base.dynamixel, device_name=args.dxl_port, baudrate=args.dxl_baud),
        camera=CameraConfig(left_url=args.left_camera, right_url=args.right_camera),
    )


def make_row(
    target: EyeTrackerTarget,
    safety_limited: bool,
    left_image: Path | None,
    right_image: Path | None,
) -> dict[str, object]:
    return {
        "target_index": target.index,
        "timestamp_s": time.time(),
        "left_yaw_deg": target.left_pose.yaw_deg,
        "left_pitch_deg": target.left_pose.pitch_deg,
        "left_roll_deg": target.left_pose.roll_deg,
        "right_yaw_deg": target.right_pose.yaw_deg,
        "right_pitch_deg": target.right_pose.pitch_deg,
        "right_roll_deg": target.right_pose.roll_deg,
        "safety_limited": int(safety_limited),
        "left_image": str(left_image) if left_image is not None else "",
        "right_image": str(right_image) if right_image is not None else "",
    }


def run() -> int:
    args = parse_args()
    config = build_config(args)
    targets = load_eye_tracker_targets(
        args.data_path,
        key=args.mat_key,
        start_index=args.start_index,
        max_targets=args.max_targets,
        sign=args.target_sign,
    )
    if not targets:
        raise RuntimeError("No eye-tracker targets were loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eye_tracker_replay_log.csv"
    left_dir = args.output_dir / "left"
    right_dir = args.output_dir / "right"

    controller = DynamixelController(config.dynamixel, config.profile)
    left_camera = StillFrameCapture("left", config.camera.left_url, args.camera_warmup_frames)
    right_camera = StillFrameCapture("right", config.camera.right_url, args.camera_warmup_frames)

    try:
        if not args.no_motors:
            controller.open()
            print(f"Dynamixel baseline: {controller.initial_positions}")
        if not args.no_cameras:
            left_camera.open()
            right_camera.open()

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for count, target in enumerate(targets, start=1):
                pulses = motor_pulses_from_pose(target.left_pose, target.right_pose)
                safety_limited = any(abs(value) > config.safety.max_abs_pulse for value in pulses)
                if safety_limited:
                    print(f"[skip] target {target.index}: pulse command exceeds safety limit")
                    writer.writerow(make_row(target, True, None, None))
                    continue

                if not args.no_motors:
                    controller.command_position_diffs(pulses)
                time.sleep(max(0.0, args.settle_time_s))

                left_image = left_dir / f"target_{target.index:04d}.jpg"
                right_image = right_dir / f"target_{target.index:04d}.jpg"
                if not args.no_cameras:
                    left_camera.capture(left_image)
                    right_camera.capture(right_image)
                else:
                    left_image = None
                    right_image = None

                writer.writerow(make_row(target, False, left_image, right_image))
                print(f"[{count}/{len(targets)}] target {target.index} captured")

    finally:
        try:
            if not args.no_motors and controller.is_open:
                controller.reset_to_baseline()
                time.sleep(0.2)
                controller.enable_torque(False)
        finally:
            controller.close()
            left_camera.close()
            right_camera.close()

    print(f"Replay log written to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
