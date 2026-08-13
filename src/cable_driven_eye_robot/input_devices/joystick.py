from __future__ import annotations

from dataclasses import dataclass

from ..config import JoystickConfig
from ..math_utils import apply_deadband, clamp


@dataclass(frozen=True)
class GazeTarget:
    raw_yaw_axis: float
    raw_pitch_axis: float
    yaw_deg: float
    pitch_deg: float


class JoystickReader:
    def __init__(self, config: JoystickConfig):
        self.config = config
        self.pygame = None
        self.joystick = None

    def open(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            pygame.quit()
            raise RuntimeError("No joystick detected.")
        self.joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self.joystick.init()

    def close(self) -> None:
        if self.pygame is not None:
            self.pygame.quit()

    @property
    def name(self) -> str:
        if self.joystick is None:
            return ""
        return self.joystick.get_name()

    def pump(self) -> None:
        self._require_open()
        self.pygame.event.pump()

    def read_gaze_target(self) -> GazeTarget:
        self.pump()
        raw_yaw = self.safe_get_axis(self.config.yaw_axis)
        raw_pitch = self.safe_get_axis(self.config.pitch_axis)

        yaw_axis = apply_deadband(raw_yaw, self.config.deadband)
        pitch_axis = apply_deadband(-raw_pitch, self.config.deadband)

        yaw = self.config.yaw_sign * yaw_axis * self.config.max_yaw_deg
        pitch = self.config.pitch_sign * pitch_axis * self.config.max_pitch_deg
        yaw = clamp(yaw, -self.config.max_yaw_deg, self.config.max_yaw_deg)
        pitch = clamp(pitch, -self.config.max_pitch_deg, self.config.max_pitch_deg)
        return GazeTarget(raw_yaw, raw_pitch, yaw, pitch)

    def triangle_pressed(self) -> bool:
        self.pump()
        return self.safe_get_button(self.config.triangle_button)

    def safe_get_axis(self, axis_index: int) -> float:
        self._require_open()
        if 0 <= axis_index < self.joystick.get_numaxes():
            return float(self.joystick.get_axis(axis_index))
        return 0.0

    def safe_get_button(self, button_index: int) -> bool:
        self._require_open()
        if 0 <= button_index < self.joystick.get_numbuttons():
            return bool(self.joystick.get_button(button_index))
        return False

    def safe_get_hat(self, hat_index: int) -> tuple[int, int]:
        self._require_open()
        if 0 <= hat_index < self.joystick.get_numhats():
            return self.joystick.get_hat(hat_index)
        return (0, 0)

    def _require_open(self) -> None:
        if self.pygame is None or self.joystick is None:
            raise RuntimeError("JoystickReader is not open.")
