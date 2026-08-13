from __future__ import annotations

from dataclasses import dataclass

from ..config import FeedbackConfig
from ..kinematics import EyePose, motor_pulses_from_pose
from ..math_utils import angle_error_deg, clamp
from ..sensors.imu import IMUSample


@dataclass(frozen=True)
class FeedbackResult:
    command_pulses: list[float]
    correction_pulses: list[float]
    left_error: EyePose
    right_error: EyePose
    left_reached: bool
    right_reached: bool


class EyePoseFeedbackController:
    """Small PD-style correction around the open-loop inverse-kinematic command.

    The open-loop model produces the main motor command. This controller adds a
    bounded correction based on IMU pose error. The pair/sign mapping is kept
    explicit so it can be tuned without changing the rest of the system.
    """

    # Motor order follows config.MOTOR_CHANNELS:
    # left LR,MR,SR,IR,SO,IO and right MR,LR,SR,IR,SO,IO.
    LEFT_SIGNS = {
        "yaw": (-1.0, 1.0),
        "pitch": (1.0, -1.0),
        "roll": (-1.0, -1.0),
    }
    RIGHT_SIGNS = {
        "yaw": (1.0, -1.0),
        "pitch": (-1.0, 1.0),
        "roll": (-1.0, -1.0),
    }

    def __init__(self, config: FeedbackConfig):
        self.config = config

    def build_command(
        self,
        target_left: EyePose,
        target_right: EyePose,
        left_sample: IMUSample | None,
        right_sample: IMUSample | None,
        control_left: bool = True,
        control_right: bool = True,
    ) -> FeedbackResult:
        feedforward = motor_pulses_from_pose(target_left, target_right)
        correction = [0.0] * len(feedforward)

        left_error = self._pose_error(left_sample.pose if left_sample else None, target_left)
        right_error = self._pose_error(right_sample.pose if right_sample else None, target_right)
        left_reached = self._is_reached(left_error) if left_sample is not None else False
        right_reached = self._is_reached(right_error) if right_sample is not None else False

        if control_left and left_sample is not None and not left_reached:
            self._add_eye_correction(correction, 0, left_error, left_sample, self.LEFT_SIGNS)
        if control_right and right_sample is not None and not right_reached:
            self._add_eye_correction(correction, 6, right_error, right_sample, self.RIGHT_SIGNS)

        command = [base + delta for base, delta in zip(feedforward, correction)]
        return FeedbackResult(command, correction, left_error, right_error, left_reached, right_reached)

    def _pose_error(self, actual: EyePose | None, target: EyePose) -> EyePose:
        if actual is None:
            return EyePose(float("nan"), float("nan"), float("nan"))
        return EyePose(
            angle_error_deg(actual.yaw_deg, target.yaw_deg),
            target.pitch_deg - actual.pitch_deg,
            angle_error_deg(actual.roll_deg, target.roll_deg),
        )

    def _is_reached(self, error: EyePose) -> bool:
        return (
            abs(error.yaw_deg) <= self.config.tolerance_deg
            and abs(error.pitch_deg) <= self.config.tolerance_deg
            and abs(error.roll_deg) <= self.config.tolerance_deg
        )

    def _add_eye_correction(
        self,
        correction: list[float],
        offset: int,
        error: EyePose,
        sample: IMUSample,
        signs: dict[str, tuple[float, float]],
    ) -> None:
        self._add_axis(
            correction,
            offset,
            axis="yaw",
            error_deg=error.yaw_deg,
            rate_dps=sample.angular_velocity.yaw_deg,
            gain=self.config.kp_yaw_ticks_per_deg,
            derivative_gain=self.config.kd_yaw_ticks_per_dps,
            signs=signs["yaw"],
        )
        self._add_axis(
            correction,
            offset + 2,
            axis="pitch",
            error_deg=error.pitch_deg,
            rate_dps=sample.angular_velocity.pitch_deg,
            gain=self.config.kp_pitch_ticks_per_deg,
            derivative_gain=self.config.kd_pitch_ticks_per_dps,
            signs=signs["pitch"],
        )
        self._add_axis(
            correction,
            offset + 4,
            axis="roll",
            error_deg=error.roll_deg,
            rate_dps=sample.angular_velocity.roll_deg,
            gain=self.config.kp_roll_ticks_per_deg,
            derivative_gain=self.config.kd_roll_ticks_per_dps,
            signs=signs["roll"],
        )

    def _add_axis(
        self,
        correction: list[float],
        first_index: int,
        axis: str,
        error_deg: float,
        rate_dps: float,
        gain: float,
        derivative_gain: float,
        signs: tuple[float, float],
    ) -> None:
        del axis
        if abs(error_deg) < self.config.min_error_deg:
            return

        if rate_dps != rate_dps:
            rate_dps = 0.0

        raw_step = gain * error_deg - derivative_gain * rate_dps
        step_mag = abs(raw_step)
        if step_mag < self.config.min_step_ticks:
            step_mag = self.config.min_step_ticks
        step_mag = clamp(step_mag, 0.0, self.config.max_step_ticks)
        step = step_mag if raw_step >= 0 else -step_mag

        correction[first_index] += signs[0] * step
        correction[first_index + 1] += signs[1] * step
