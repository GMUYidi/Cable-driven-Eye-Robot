from __future__ import annotations

import argparse
import threading
import tkinter as tk
from dataclasses import replace

from ..config import CameraConfig, DynamixelConfig, IMUConfig, RobotConfig
from ..control.open_loop import OpenLoopJoystickController
from ..sensors.camera import CameraStream
from ..sensors.imu import IMUReader
from ..ui.dashboard import EyeCameraDashboard
from ..ui.state import SharedRobotState


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must look like 1280x900") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joystick open-loop control + dual IMU + dual ESP32 camera dashboard."
    )
    parser.add_argument("--dxl-port", default="COM12")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--left-imu-port", default="COM11")
    parser.add_argument("--right-imu-port", default="COM6")
    parser.add_argument("--imu-baud", type=int, default=921_600)
    parser.add_argument("--left-camera", default="http://192.168.4.1/stream")
    parser.add_argument("--right-camera", default="http://192.168.4.200/stream")
    parser.add_argument("--no-motors", action="store_true")
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--yaw-gain", type=float, default=1.65)
    parser.add_argument("--pitch-gain", type=float, default=1.65)
    parser.add_argument("--roll-gain", type=float, default=1.65)
    parser.add_argument("--window-size", type=parse_size, default=(1280, 900))
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
    )


def run() -> int:
    args = parse_args()
    config = build_config(args)
    state = SharedRobotState()
    stop_event = threading.Event()

    if not args.no_serial:
        left_imu = IMUReader("Left", config.imu.left_port, config.imu, callback=state.update_imu)
        right_imu = IMUReader("Right", config.imu.right_port, config.imu, callback=state.update_imu)
        left_imu.start()
        right_imu.start()
    else:
        left_imu = None
        right_imu = None
        state.add_log("IMU serial disabled")

    if not args.no_cameras:
        left_camera = CameraStream("Left", config.camera.left_url, callback=state.update_camera)
        right_camera = CameraStream("Right", config.camera.right_url, callback=state.update_camera)
        left_camera.start()
        right_camera.start()
    else:
        left_camera = None
        right_camera = None
        state.add_log("camera streams disabled")

    controller = OpenLoopJoystickController(
        config,
        stop_event,
        state_callback=lambda sample: state.update_control(sample, controller.joystick.name),
        log_callback=state.add_log,
        enable_motors=not args.no_motors,
    )
    controller.start()

    def stop_all() -> None:
        stop_event.set()

    root = tk.Tk()
    width, height = args.window_size
    root.geometry(f"{width}x{height}")
    EyeCameraDashboard(
        root,
        state,
        stop_all,
        yaw_gain=args.yaw_gain,
        pitch_gain=args.pitch_gain,
        roll_gain=args.roll_gain,
    )

    try:
        root.mainloop()
    finally:
        stop_event.set()
        controller.join(timeout=4.0)
        for resource in (left_imu, right_imu, left_camera, right_camera):
            if resource is not None:
                resource.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
