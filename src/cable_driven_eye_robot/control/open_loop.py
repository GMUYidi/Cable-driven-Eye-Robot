from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from ..config import RobotConfig
from ..hardware.dynamixel import DynamixelController
from ..input_devices.joystick import GazeTarget, JoystickReader
from ..kinematics import EyePose, motor_pulses_from_gaze
from ..math_utils import clamp, limit_step


@dataclass(frozen=True)
class OpenLoopState:
    target: GazeTarget
    command_pose: EyePose
    pulses: list[float]
    safety_limited: bool


class OpenLoopJoystickController(threading.Thread):
    def __init__(
        self,
        config: RobotConfig,
        stop_event: threading.Event,
        state_callback: Callable[[OpenLoopState], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        enable_motors: bool = True,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.state_callback = state_callback
        self.log_callback = log_callback
        self.enable_motors = enable_motors
        self.joystick = JoystickReader(config.joystick)
        self.controller = DynamixelController(config.dynamixel, config.profile)
        self.x_cmd = 0.0
        self.y_cmd = 0.0
        self._last_triangle = False

    def log(self, message: str) -> None:
        if self.log_callback is not None:
            self.log_callback(message)
        else:
            print(message)

    def run(self) -> None:
        try:
            self.joystick.open()
            self.log(f"[joystick] {self.joystick.name}")

            if self.enable_motors:
                self.controller.open()
                self.log(f"[dxl] opened {self.config.dynamixel.device_name}")
                self.log(f"[dxl] baseline: {self.controller.initial_positions}")
            else:
                self.log("[dxl] motor control disabled")

            while not self.stop_event.is_set():
                self.control_once()
                time.sleep(self.config.safety.control_period_s)
        except Exception as exc:
            self.log(f"[open-loop] stopped: {exc}")
        finally:
            self.shutdown()

    def control_once(self) -> None:
        if self.joystick.triangle_pressed() and not self._last_triangle:
            self.x_cmd = 0.0
            self.y_cmd = 0.0
            if self.enable_motors and self.controller.is_open:
                self.controller.reset_to_baseline()
            self.log("[open-loop] triangle reset to baseline")
            self._last_triangle = True
            return
        self._last_triangle = self.joystick.triangle_pressed()

        target = self.joystick.read_gaze_target()
        joystick_config = self.config.joystick
        self.x_cmd = limit_step(self.x_cmd, target.yaw_deg, joystick_config.max_yaw_step_deg)
        self.y_cmd = limit_step(self.y_cmd, target.pitch_deg, joystick_config.max_pitch_step_deg)
        self.x_cmd = clamp(self.x_cmd, -joystick_config.max_yaw_deg, joystick_config.max_yaw_deg)
        self.y_cmd = clamp(self.y_cmd, -joystick_config.max_pitch_deg, joystick_config.max_pitch_deg)

        pulses, pose = motor_pulses_from_gaze(self.x_cmd, self.y_cmd)
        safety_limited = any(abs(value) > self.config.safety.max_abs_pulse for value in pulses)
        if safety_limited:
            pulses = [0.0] * len(self.config.dynamixel.motor_ids)

        if self.enable_motors and self.controller.is_open:
            self.controller.command_position_diffs(pulses)

        if self.state_callback is not None:
            self.state_callback(OpenLoopState(target, pose, pulses, safety_limited))

    def shutdown(self) -> None:
        try:
            if self.enable_motors and self.controller.is_open:
                self.controller.reset_to_baseline()
                time.sleep(0.2)
                self.controller.enable_torque(False)
        except Exception as exc:
            self.log(f"[open-loop] shutdown reset failed: {exc}")
        finally:
            self.controller.close()
            self.joystick.close()
            self.log("[open-loop] shutdown complete")
