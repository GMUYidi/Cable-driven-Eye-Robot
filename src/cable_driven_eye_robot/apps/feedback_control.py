from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

from ..config import FeedbackConfig, RobotConfig
from ..control.feedback import EyePoseFeedbackController
from ..hardware.dynamixel import DynamixelController
from ..input_devices.joystick import JoystickReader
from ..kinematics import EyePose
from ..math_utils import clamp, limit_step
from ..sensors.imu import IMUReader


CSV_FIELDS = [
    "time_s",
    "target_yaw_deg",
    "target_pitch_deg",
    "command_yaw_deg",
    "command_pitch_deg",
    "command_roll_deg",
    "left_actual_yaw_deg",
    "left_actual_pitch_deg",
    "left_actual_roll_deg",
    "right_actual_yaw_deg",
    "right_actual_pitch_deg",
    "right_actual_roll_deg",
    "left_error_yaw_deg",
    "left_error_pitch_deg",
    "left_error_roll_deg",
    "right_error_yaw_deg",
    "right_error_pitch_deg",
    "right_error_roll_deg",
    "left_reached",
    "right_reached",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joystick + IMU feedback controller.")
    parser.add_argument("--dxl-port", default="COM12")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--left-imu-port", default="COM11")
    parser.add_argument("--right-imu-port", default="COM6")
    parser.add_argument("--imu-baud", type=int, default=921_600)
    parser.add_argument("--loop-period-s", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("runningdata"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def keyboard_stop_requested() -> bool:
    if msvcrt is None:
        return False
    if msvcrt.kbhit():
        msvcrt.getch()
        return True
    return False


def build_config(args: argparse.Namespace) -> RobotConfig:
    base = RobotConfig()
    return replace(
        base,
        dynamixel=replace(base.dynamixel, device_name=args.dxl_port, baudrate=args.dxl_baud),
        imu=replace(
            base.imu,
            left_port=args.left_imu_port,
            right_port=args.right_imu_port,
            baudrate=args.imu_baud,
        ),
        feedback=replace(base.feedback, loop_period_s=args.loop_period_s),
    )


def pose_value(sample, attr: str) -> float:
    if sample is None:
        return float("nan")
    return getattr(sample.pose, attr)


def run() -> int:
    args = parse_args()
    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"feedback_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    joystick = JoystickReader(config.joystick)
    controller = DynamixelController(config.dynamixel, config.profile)
    feedback = EyePoseFeedbackController(config.feedback)
    left_imu = IMUReader("Left", config.imu.left_port, config.imu, zero_on_open=True)
    right_imu = IMUReader("Right", config.imu.right_port, config.imu, zero_on_open=True)

    x_cmd = 0.0
    y_cmd = 0.0
    start_time = time.monotonic()

    try:
        joystick.open()
        print(f"Detected joystick: {joystick.name}")
        if not args.dry_run:
            controller.open()
            print(f"Dynamixel baseline: {controller.initial_positions}")

        left_imu.start()
        right_imu.start()

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            print("Feedback control started. Press any key in the terminal to stop.")

            while True:
                loop_start = time.monotonic()
                if keyboard_stop_requested():
                    print("Keyboard stop requested.")
                    break

                target = joystick.read_gaze_target()
                x_cmd = limit_step(x_cmd, target.yaw_deg, config.joystick.max_yaw_step_deg)
                y_cmd = limit_step(y_cmd, target.pitch_deg, config.joystick.max_pitch_step_deg)
                x_cmd = clamp(x_cmd, -config.joystick.max_yaw_deg, config.joystick.max_yaw_deg)
                y_cmd = clamp(y_cmd, -config.joystick.max_pitch_deg, config.joystick.max_pitch_deg)
                command_pose = EyePose.from_yaw_pitch(x_cmd, y_cmd)

                left_sample, _left_age, left_valid = left_imu.latest(config.imu.max_age_s)
                right_sample, _right_age, right_valid = right_imu.latest(config.imu.max_age_s)
                result = feedback.build_command(
                    command_pose,
                    command_pose,
                    left_sample if left_valid else None,
                    right_sample if right_valid else None,
                )

                safety_limited = any(abs(value) > config.safety.max_abs_pulse for value in result.command_pulses)
                command_pulses = [0.0] * 12 if safety_limited else result.command_pulses
                if not args.dry_run:
                    controller.command_position_diffs(command_pulses)

                writer.writerow(
                    {
                        "time_s": time.monotonic() - start_time,
                        "target_yaw_deg": target.yaw_deg,
                        "target_pitch_deg": target.pitch_deg,
                        "command_yaw_deg": command_pose.yaw_deg,
                        "command_pitch_deg": command_pose.pitch_deg,
                        "command_roll_deg": command_pose.roll_deg,
                        "left_actual_yaw_deg": pose_value(left_sample, "yaw_deg"),
                        "left_actual_pitch_deg": pose_value(left_sample, "pitch_deg"),
                        "left_actual_roll_deg": pose_value(left_sample, "roll_deg"),
                        "right_actual_yaw_deg": pose_value(right_sample, "yaw_deg"),
                        "right_actual_pitch_deg": pose_value(right_sample, "pitch_deg"),
                        "right_actual_roll_deg": pose_value(right_sample, "roll_deg"),
                        "left_error_yaw_deg": result.left_error.yaw_deg,
                        "left_error_pitch_deg": result.left_error.pitch_deg,
                        "left_error_roll_deg": result.left_error.roll_deg,
                        "right_error_yaw_deg": result.right_error.yaw_deg,
                        "right_error_pitch_deg": result.right_error.pitch_deg,
                        "right_error_roll_deg": result.right_error.roll_deg,
                        "left_reached": int(result.left_reached),
                        "right_reached": int(result.right_reached),
                    }
                )
                csv_file.flush()

                sleep_s = config.feedback.loop_period_s - (time.monotonic() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        return 0
    finally:
        try:
            if controller.is_open:
                controller.reset_to_baseline()
                controller.enable_torque(False)
        finally:
            left_imu.close()
            right_imu.close()
            controller.close()
            joystick.close()
            print(f"CSV saved at: {csv_path}")


if __name__ == "__main__":
    raise SystemExit(run())
