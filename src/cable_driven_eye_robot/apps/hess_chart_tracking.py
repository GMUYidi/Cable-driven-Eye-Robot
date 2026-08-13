from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import replace
from pathlib import Path

from ..config import CameraConfig, FeedbackConfig, RobotConfig
from ..control.feedback import EyePoseFeedbackController, FeedbackResult
from ..data.hess_chart import (
    HessChartTarget,
    generate_grid_targets,
    generate_hess9_targets,
    parse_degree_list,
)
from ..hardware.dynamixel import DynamixelController
from ..kinematics import EyePose, motor_pulses_from_pose
from ..sensors.frame_capture import StillFrameCapture
from ..sensors.imu import IMUReader, IMUSample


SUMMARY_FIELDS = [
    "target_index",
    "label",
    "status",
    "duration_s",
    "loop_count",
    "stable_samples",
    "safety_limited",
    "target_left_yaw_deg",
    "target_left_pitch_deg",
    "target_left_roll_deg",
    "target_right_yaw_deg",
    "target_right_pitch_deg",
    "target_right_roll_deg",
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
    "left_image",
    "right_image",
]


SAMPLE_FIELDS = [
    "target_index",
    "label",
    "loop_index",
    "elapsed_s",
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
    "safety_limited",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a Hess-chart gaze sequence with inverse kinematics, IMU feedback, and eye-camera capture."
    )
    parser.add_argument("--pattern", choices=("hess9", "grid"), default="hess9")
    parser.add_argument("--amplitude-deg", type=float, default=8.0)
    parser.add_argument("--yaw-points", default="-10,-5,0,5,10")
    parser.add_argument("--pitch-points", default="-8,0,8")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--dxl-port", default="COM12")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--left-imu-port", default="COM11")
    parser.add_argument("--right-imu-port", default="COM6")
    parser.add_argument("--imu-baud", type=int, default=921_600)
    parser.add_argument("--left-camera", default="http://192.168.4.1/stream")
    parser.add_argument("--right-camera", default="http://192.168.4.200/stream")
    parser.add_argument("--loop-period-s", type=float, default=0.05)
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--target-timeout-s", type=float, default=5.0)
    parser.add_argument("--settle-without-imu-s", type=float, default=0.35)
    parser.add_argument("--imu-ready-timeout-s", type=float, default=8.0)
    parser.add_argument("--camera-warmup-frames", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hess_chart_tracking"))
    parser.add_argument("--capture-on-timeout", action="store_true")
    parser.add_argument("--no-motors", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    return parser.parse_args()


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
        camera=CameraConfig(left_url=args.left_camera, right_url=args.right_camera),
        feedback=replace(
            base.feedback,
            loop_period_s=args.loop_period_s,
            tolerance_deg=args.tolerance_deg,
        ),
    )


def build_targets(args: argparse.Namespace) -> list[HessChartTarget]:
    if args.pattern == "hess9":
        targets = generate_hess9_targets(args.amplitude_deg)
    else:
        targets = generate_grid_targets(
            parse_degree_list(args.yaw_points),
            parse_degree_list(args.pitch_points),
        )

    if args.max_targets is not None:
        targets = targets[: max(0, args.max_targets)]
    if not targets:
        raise RuntimeError("No Hess-chart targets were generated.")
    return targets


def wait_for_imus(
    left_imu: IMUReader,
    right_imu: IMUReader,
    config: RobotConfig,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _left_sample, _left_age, left_valid = left_imu.latest(config.imu.max_age_s)
        _right_sample, _right_age, right_valid = right_imu.latest(config.imu.max_age_s)
        if left_valid and right_valid:
            return
        time.sleep(0.05)
    raise TimeoutError("IMU streams did not produce fresh left/right samples before the ready timeout.")


def track_with_feedback(
    target: HessChartTarget,
    args: argparse.Namespace,
    config: RobotConfig,
    controller: DynamixelController,
    feedback: EyePoseFeedbackController,
    left_imu: IMUReader,
    right_imu: IMUReader,
    sample_writer: csv.DictWriter,
) -> tuple[str, int, int, bool, FeedbackResult | None, IMUSample | None, IMUSample | None, float]:
    start = time.monotonic()
    loop_count = 0
    stable_count = 0
    safety_limited = False
    last_result: FeedbackResult | None = None
    last_left_sample: IMUSample | None = None
    last_right_sample: IMUSample | None = None

    while time.monotonic() - start <= args.target_timeout_s:
        loop_start = time.monotonic()
        left_sample, _left_age, left_valid = left_imu.latest(config.imu.max_age_s)
        right_sample, _right_age, right_valid = right_imu.latest(config.imu.max_age_s)
        last_left_sample = left_sample if left_valid else last_left_sample
        last_right_sample = right_sample if right_valid else last_right_sample

        result = feedback.build_command(
            target.left_pose,
            target.right_pose,
            left_sample if left_valid else None,
            right_sample if right_valid else None,
        )
        last_result = result
        safety_limited = any(abs(value) > config.safety.max_abs_pulse for value in result.command_pulses)

        if not safety_limited and not args.no_motors:
            controller.command_position_diffs(result.command_pulses)

        sample_writer.writerow(
            make_sample_row(
                target=target,
                loop_index=loop_count,
                elapsed_s=time.monotonic() - start,
                left_sample=left_sample if left_valid else None,
                right_sample=right_sample if right_valid else None,
                result=result,
                safety_limited=safety_limited,
            )
        )

        if safety_limited:
            return "safety_limited", loop_count + 1, stable_count, True, result, last_left_sample, last_right_sample, time.monotonic() - start

        if result.left_reached and result.right_reached:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= args.stable_samples:
            return "reached", loop_count + 1, stable_count, False, result, last_left_sample, last_right_sample, time.monotonic() - start

        loop_count += 1
        sleep_s = config.feedback.loop_period_s - (time.monotonic() - loop_start)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return "timeout", loop_count, stable_count, safety_limited, last_result, last_left_sample, last_right_sample, time.monotonic() - start


def command_without_imu(
    target: HessChartTarget,
    args: argparse.Namespace,
    config: RobotConfig,
    controller: DynamixelController,
) -> tuple[str, int, bool, float]:
    start = time.monotonic()
    pulses = motor_pulses_from_pose(target.left_pose, target.right_pose)
    safety_limited = any(abs(value) > config.safety.max_abs_pulse for value in pulses)
    if safety_limited:
        return "safety_limited", 0, True, time.monotonic() - start
    if not args.no_motors:
        controller.command_position_diffs(pulses)
    time.sleep(max(0.0, args.settle_without_imu_s))
    return "settled_without_imu", 1, False, time.monotonic() - start


def capture_target_images(
    target: HessChartTarget,
    args: argparse.Namespace,
    left_camera: StillFrameCapture,
    right_camera: StillFrameCapture,
    left_dir: Path,
    right_dir: Path,
    status: str,
) -> tuple[Path | None, Path | None]:
    should_capture = status in {"reached", "settled_without_imu"} or (
        args.capture_on_timeout and status == "timeout"
    )
    if args.no_cameras or not should_capture:
        return None, None

    image_name = f"target_{target.index:04d}_{target.label}.jpg"
    left_image = left_dir / image_name
    right_image = right_dir / image_name
    left_camera.capture(left_image)
    right_camera.capture(right_image)
    return left_image, right_image


def make_sample_row(
    target: HessChartTarget,
    loop_index: int,
    elapsed_s: float,
    left_sample: IMUSample | None,
    right_sample: IMUSample | None,
    result: FeedbackResult,
    safety_limited: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target_index": target.index,
        "label": target.label,
        "loop_index": loop_index,
        "elapsed_s": f"{elapsed_s:.4f}",
        "left_reached": int(result.left_reached),
        "right_reached": int(result.right_reached),
        "safety_limited": int(safety_limited),
    }
    row.update(sample_pose_fields("left_actual", left_sample))
    row.update(sample_pose_fields("right_actual", right_sample))
    row.update(pose_fields("left_error", result.left_error))
    row.update(pose_fields("right_error", result.right_error))
    return row


def make_summary_row(
    target: HessChartTarget,
    status: str,
    duration_s: float,
    loop_count: int,
    stable_samples: int,
    safety_limited: bool,
    result: FeedbackResult | None,
    left_sample: IMUSample | None,
    right_sample: IMUSample | None,
    left_image: Path | None,
    right_image: Path | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target_index": target.index,
        "label": target.label,
        "status": status,
        "duration_s": f"{duration_s:.4f}",
        "loop_count": loop_count,
        "stable_samples": stable_samples,
        "safety_limited": int(safety_limited),
        "left_image": str(left_image) if left_image is not None else "",
        "right_image": str(right_image) if right_image is not None else "",
    }
    row.update(pose_fields("target_left", target.left_pose))
    row.update(pose_fields("target_right", target.right_pose))
    row.update(sample_pose_fields("left_actual", left_sample))
    row.update(sample_pose_fields("right_actual", right_sample))
    row.update(pose_fields("left_error", result.left_error if result is not None else nan_pose()))
    row.update(pose_fields("right_error", result.right_error if result is not None else nan_pose()))
    return row


def pose_fields(prefix: str, pose: EyePose) -> dict[str, float]:
    return {
        f"{prefix}_yaw_deg": pose.yaw_deg,
        f"{prefix}_pitch_deg": pose.pitch_deg,
        f"{prefix}_roll_deg": pose.roll_deg,
    }


def sample_pose_fields(prefix: str, sample: IMUSample | None) -> dict[str, float]:
    return pose_fields(prefix, sample.pose if sample is not None else nan_pose())


def nan_pose() -> EyePose:
    return EyePose(math.nan, math.nan, math.nan)


def run() -> int:
    args = parse_args()
    config = build_config(args)
    targets = build_targets(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = args.output_dir / "left"
    right_dir = args.output_dir / "right"
    summary_path = args.output_dir / "hess_chart_tracking_summary.csv"
    samples_path = args.output_dir / "hess_chart_tracking_samples.csv"

    controller = DynamixelController(config.dynamixel, config.profile)
    feedback = EyePoseFeedbackController(config.feedback)
    left_imu = IMUReader("Left", config.imu.left_port, config.imu, zero_on_open=True)
    right_imu = IMUReader("Right", config.imu.right_port, config.imu, zero_on_open=True)
    left_camera = StillFrameCapture("left", config.camera.left_url, args.camera_warmup_frames)
    right_camera = StillFrameCapture("right", config.camera.right_url, args.camera_warmup_frames)

    try:
        if not args.no_motors:
            controller.open()
            print(f"Dynamixel baseline: {controller.initial_positions}")
        if not args.no_imu:
            left_imu.start()
            right_imu.start()
            wait_for_imus(left_imu, right_imu, config, args.imu_ready_timeout_s)
            print("Left/right IMU streams are ready.")
        if not args.no_cameras:
            left_camera.open()
            right_camera.open()

        with summary_path.open("w", newline="", encoding="utf-8") as summary_file, samples_path.open(
            "w", newline="", encoding="utf-8"
        ) as samples_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            sample_writer = csv.DictWriter(samples_file, fieldnames=SAMPLE_FIELDS)
            summary_writer.writeheader()
            sample_writer.writeheader()

            for count, target in enumerate(targets, start=1):
                if args.no_imu:
                    status, loop_count, safety_limited, duration_s = command_without_imu(
                        target, args, config, controller
                    )
                    stable_count = 0
                    result = None
                    left_sample = None
                    right_sample = None
                else:
                    (
                        status,
                        loop_count,
                        stable_count,
                        safety_limited,
                        result,
                        left_sample,
                        right_sample,
                        duration_s,
                    ) = track_with_feedback(
                        target,
                        args,
                        config,
                        controller,
                        feedback,
                        left_imu,
                        right_imu,
                        sample_writer,
                    )

                left_image, right_image = capture_target_images(
                    target,
                    args,
                    left_camera,
                    right_camera,
                    left_dir,
                    right_dir,
                    status,
                )
                summary_writer.writerow(
                    make_summary_row(
                        target,
                        status,
                        duration_s,
                        loop_count,
                        stable_count,
                        safety_limited,
                        result,
                        left_sample,
                        right_sample,
                        left_image,
                        right_image,
                    )
                )
                summary_file.flush()
                samples_file.flush()
                print(f"[{count}/{len(targets)}] {target.label}: {status}")

    finally:
        try:
            if not args.no_motors and controller.is_open:
                controller.reset_to_baseline()
                time.sleep(0.2)
                controller.enable_torque(False)
        finally:
            left_imu.close()
            right_imu.close()
            controller.close()
            left_camera.close()
            right_camera.close()

    print(f"Summary log written to {summary_path}")
    print(f"Sample log written to {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
