from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..control.open_loop import OpenLoopState
from ..sensors.camera import CameraFrame
from ..sensors.imu import IMUSample


@dataclass
class EyeDisplayState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    target_yaw_deg: float = 0.0
    target_pitch_deg: float = 0.0
    target_roll_deg: float = 0.0
    has_data: bool = False
    connected: bool = False
    last_update_s: float = 0.0


@dataclass
class CameraDisplayState:
    frame: object = None
    connected: bool = False
    fps: float = 0.0
    last_update_s: float = 0.0
    last_error: str = ""


@dataclass
class ControlDisplayState:
    active: bool = False
    joystick_name: str = ""
    yaw_cmd_deg: float = 0.0
    pitch_cmd_deg: float = 0.0
    roll_cmd_deg: float = 0.0
    safety_limited: bool = False
    message: str = "not started"
    last_update_s: float = 0.0


@dataclass
class DashboardSnapshot:
    left_eye: EyeDisplayState
    right_eye: EyeDisplayState
    left_camera: CameraDisplayState
    right_camera: CameraDisplayState
    control: ControlDisplayState
    logs: list[str] = field(default_factory=list)


class SharedRobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.left_eye = EyeDisplayState()
        self.right_eye = EyeDisplayState()
        self.left_camera = CameraDisplayState()
        self.right_camera = CameraDisplayState()
        self.control = ControlDisplayState()
        self.logs: list[str] = []

    def add_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"{stamp} {message}")
            self.logs = self.logs[-8:]

    def update_imu(self, side: str, sample: IMUSample) -> None:
        with self.lock:
            state = self.left_eye if side.lower().startswith("left") else self.right_eye
            state.target_yaw_deg = sample.pose.yaw_deg
            state.target_pitch_deg = sample.pose.pitch_deg
            state.target_roll_deg = sample.pose.roll_deg
            state.has_data = True
            state.connected = True
            state.last_update_s = time.time()

    def update_camera(self, side: str, frame: CameraFrame) -> None:
        with self.lock:
            state = self.left_camera if side.lower().startswith("left") else self.right_camera
            state.frame = frame.frame
            state.fps = frame.fps
            state.connected = True
            state.last_update_s = time.time()
            state.last_error = ""

    def update_control(self, state: OpenLoopState, joystick_name: str = "") -> None:
        with self.lock:
            self.control.active = True
            self.control.joystick_name = joystick_name or self.control.joystick_name
            self.control.yaw_cmd_deg = state.command_pose.yaw_deg
            self.control.pitch_cmd_deg = state.command_pose.pitch_deg
            self.control.roll_cmd_deg = state.command_pose.roll_deg
            self.control.safety_limited = state.safety_limited
            self.control.message = "safety limited" if state.safety_limited else "running"
            self.control.last_update_s = time.time()

    def set_control_message(self, message: str, active: bool | None = None) -> None:
        with self.lock:
            self.control.message = message
            if active is not None:
                self.control.active = active
            self.control.last_update_s = time.time()

    def snapshot(self, alpha: float = 0.25) -> DashboardSnapshot:
        with self.lock:
            for state in (self.left_eye, self.right_eye):
                state.yaw_deg += (state.target_yaw_deg - state.yaw_deg) * alpha
                state.pitch_deg += (state.target_pitch_deg - state.pitch_deg) * alpha
                state.roll_deg += (state.target_roll_deg - state.roll_deg) * alpha

            return DashboardSnapshot(
                left_eye=EyeDisplayState(**self.left_eye.__dict__),
                right_eye=EyeDisplayState(**self.right_eye.__dict__),
                left_camera=CameraDisplayState(**self.left_camera.__dict__),
                right_camera=CameraDisplayState(**self.right_camera.__dict__),
                control=ControlDisplayState(**self.control.__dict__),
                logs=list(self.logs),
            )
