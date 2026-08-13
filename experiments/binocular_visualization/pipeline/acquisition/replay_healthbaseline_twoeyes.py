from __future__ import annotations

import argparse
import csv
import ctypes
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as scio


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =========================
# Default paths
# =========================
ROBOTIC_EYE_ROOT = (
    Path.home()
    / "OneDrive"
    / "\u684c\u9762"
    / "GMU research"
    / "robotic eye"
)

DEFAULT_MAT_PATH = (
    ROBOTIC_EYE_ROOT
    / "Hess chart"
    / "bilinear interpolation"
    / "strabismus data"
    / "simulated_strabismus_Q22.mat"
)

DEFAULT_FRAME_DIR = Path("D:/frames")
DEFAULT_LEFT_SAVE_DIR = Path("D:/visualization2026/left")
DEFAULT_RIGHT_SAVE_DIR = Path("D:/visualization2026/right")
DEFAULT_JOURNAL_DIR = Path("D:/visualization2026/journal")


# =========================
# Data column convention
# =========================
# The selected target-angle matrix columns:
#   0 left horizontal/yaw, 1 left vertical/pitch,
#   2 right horizontal/yaw, 3 right vertical/pitch.
LEFT_YAW_COL = 0
LEFT_PITCH_COL = 1
RIGHT_YAW_COL = 2
RIGHT_PITCH_COL = 3

# Only scale the recorded yaw/pitch targets. Roll is then recomputed from the
# scaled yaw/pitch by Listing's law, not scaled directly.
TARGET_GAZE_SCALE = 0.6

# Change these signs if the physical eye convention is opposite to the
# Hess-chart convention.
LEFT_YAW_SIGN = 1.0
LEFT_PITCH_SIGN = 1.0
RIGHT_YAW_SIGN = 1.0
RIGHT_PITCH_SIGN = 1.0

# These only affect the motor IK command. The target/IMU comparison above keeps
# the real gaze convention from healthbaseline.mat.
LEFT_IK_YAW_SIGN = -1.0
LEFT_IK_PITCH_SIGN = -1.0
LEFT_IK_ROLL_SIGN = 1.0
RIGHT_IK_YAW_SIGN = -1.0
RIGHT_IK_PITCH_SIGN = -1.0
RIGHT_IK_ROLL_SIGN = 1.0


# =========================
# Dynamixel config
# =========================
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VEL = 112
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116

EXTENDED_POSITION_MODE = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_IDS = list(range(1, 13))
TICKS_PER_REV = 4096
PROTOCOL_VERSION = 2.0
DEFAULT_DXL_PORT = "COM12"
DEFAULT_DXL_BAUD = 1_000_000

PROFILE_ACCEL_LEFT = 100
PROFILE_VEL_LEFT = 75
PROFILE_ACCEL_RIGHT = 100
PROFILE_VEL_RIGHT = 75

MAX_GOAL_STEP_TICKS = 12_000
MOTOR_SETTLE_TOLERANCE_TICKS = 25
MOTOR_SETTLE_TIMEOUT_S = 6.0
MOTOR_SETTLE_POLL_S = 0.01
USE_MOTOR_PRESENT_POSITION_WAIT = False
USE_DYNAMIXEL_PRESENT_POSITION_READ = False
MOTOR_IK_COMMAND_DELAY_S = 4.0
COARSE_IK_FEEDBACK_DELAY_S = 0.12
FINE_PD_LOOP_DELAY_S = 0.05
SAFETY_PULSE_LIMIT = 9192
KINEMATICS_RATE = 25
DEBUG_MOTOR_COMMANDS = False
DEBUG_MOTOR_COMMAND_LIMIT = 5
PRINT_POSE_FEEDBACK = True

EYEBALL_RADIUS = 15.0
USE_SPHERE_TANGENT_IK = False
USE_CALIBRATED_KINEMATICS = True
CALIBRATED_KINEMATICS_SOURCE = "kinematic_calibration_grid_20260723_174745_summary.csv"
# Zero-centered fitted geometric model order:
# x_lr, y_lr, rectus_x, rectus_z, oblique_x, oblique_y, oblique_z,
# posterior_pulley_x, lateral_pulley_y, superior_pulley_z,
# oblique_pulley_x, oblique_pulley_y, oblique_pulley_z.
# The model was fit from motor pulses vs IMU yaw/pitch using target Listing roll,
# so it remains a cable-geometry model rather than a per-angle lookup table.
# Constant offsets are intentionally zero so target yaw=0,pitch=0 is the startup
# primary motor position.
CALIBRATED_LEFT_GEOMETRY = np.array(
    [0.54, 12.84, -4.81, 11.45, -5.61, 2.44, 9.93, -24.45, 5.81, 5.11, 14.94, 19.94, 5.23],
    dtype=float,
)
CALIBRATED_RIGHT_GEOMETRY = np.array(
    [0.58, 14.41, -5.48, 11.13, -8.08, 2.08, 10.56, -23.53, 5.30, 3.56, 13.66, 23.55, 3.96],
    dtype=float,
)
CALIBRATED_MOTOR_GAINS = np.array([2.2, 2.2, 2.2, 2.2, 2.0, 2.0], dtype=float)
CALIBRATED_LEFT_SCALE = np.array([1.00, 1.00, 0.95, 0.93, 0.77, 0.83], dtype=float)
CALIBRATED_RIGHT_SCALE = np.array([1.04, 0.99, 0.98, 0.98, 1.06, 1.01], dtype=float)
CALIBRATED_LEFT_OFFSETS = np.zeros(6, dtype=float)
CALIBRATED_RIGHT_OFFSETS = np.zeros(6, dtype=float)
CALIBRATED_BASE_RATE = 25.0
PERFECT_POSITIONS = [3599, 1194, 367, 869, 3408, 2876, 1344, 4016, 366, 2014, 3861, 1133]


# =========================
# Trajectory / controller config
# =========================
IK_THRESHOLD_DEG = 5.0
ENABLE_IMU_FINE_TUNING = True
COARSE_ANGLE_TOLERANCE_DEG = 1.5
COARSE_ACCEPT_MAX_ERROR_DEG = 2.5
COARSE_MAX_ITERATIONS = 400
COARSE_IK_FEEDBACK_GAIN = 0.15
COARSE_IK_FEEDBACK_MAX_STEP_TICKS = 100
FINE_ANGLE_TOLERANCE_DEG = 0.8
FINE_MAX_ITERATIONS = 800
FINE_PD_KP_YAW = 24.0
FINE_PD_KP_PITCH = 24.0
FINE_PD_KP_ROLL = 20.0
FINE_PD_KD_YAW = 5.0
FINE_PD_KD_PITCH = 5.0
FINE_PD_KD_ROLL = 3.0
FINE_PD_MAX_STEP_TICKS = 15
FINE_PD_MIN_STEP_TICKS = 5
FINE_PD_MIN_STEP_ERROR_DEG = 0.5
FINE_IK_PD_KP = 0.15
FINE_IK_PD_KD = 0.03
FINE_IK_PD_MAX_STEP_TICKS = 80
FINE_IK_PD_MIN_STEP_TICKS = 5
FINE_IK_PD_MIN_STEP_ERROR_TICKS = 8.0
CAPTURE_MAX_CONSECUTIVE_MISSES = 3
FINE_PD_MOTOR_PAIRS = ((0, 1), (2, 3), (4, 5))
LEFT_FINE_PD_SIGNS = (
    (-1.0, 1.0),   # +yaw: ID1 down, ID2 up
    (1.0, -1.0),   # +pitch: ID3 up, ID4 down
    (-1.0, -1.0),  # +roll: ID5 down, ID6 down
)
RIGHT_FINE_PD_SIGNS = (
    (1.0, -1.0),   # +yaw: ID7 up, ID8 down
    (-1.0, 1.0),   # +pitch: ID9 down, ID10 up
    (-1.0, -1.0),  # +roll: ID11 down, ID12 down
)
DISPLAY_SETTLE_DELAY_S = 0.05
ENABLE_FRAME_DISPLAY_AND_CAPTURE = False
IK_STEP_TEST_HOLD_S = 6.0


# =========================
# Camera serial config
# =========================
DEFAULT_LEFT_CAMERA_PORT = "COM11"
DEFAULT_RIGHT_CAMERA_PORT = "COM6"
DEFAULT_CAMERA_BAUD = 921_600
CAMERA_STARTUP_TIMEOUT_S = 12.0
CAMERA_CAPTURE_TIMEOUT_S = 120.0
IMU_READ_TIMEOUT_S = 2.0
IMU_ZERO_SAMPLES = 8
IMU_ROLL_CAMERA_ZERO_OFFSETS_DEG = {
    # When the camera image is roll-level, the IMU reports these roll values.
    # Corrected/camera roll = raw IMU roll - offset.
    "COM11": 0.1,
    "COM6": -8.25,
}
TRIANGLE_BUTTON = 3
JOYSTICK_POLL_INTERVAL_S = 0.02

# IMUtest.ino prints plain "yaw,pitch,roll".
# Some older sketches print "LEFT,yaw,roll,pitch,sys,gyro,accel,mag";
# the parser below supports both.
PLAIN_IMU_ORDER = "yaw,pitch,roll"


def imu_roll_camera_zero_offset_deg(port: str) -> float:
    return IMU_ROLL_CAMERA_ZERO_OFFSETS_DEG.get(port.upper(), 0.0)


@dataclass
class EyePose:
    yaw: float
    pitch: float
    roll: float

    def as_array(self) -> np.ndarray:
        return np.array([self.yaw, self.pitch, self.roll], dtype=float)

    @classmethod
    def from_yaw_pitch(cls, yaw: float, pitch: float) -> "EyePose":
        yaw = normalize_angle_deg(yaw)
        pitch = float(pitch)
        return cls(yaw, pitch, compute_listing_roll(yaw, pitch))


@dataclass
class TargetPose:
    left: EyePose
    right: EyePose


@dataclass
class ReplayState:
    left_command: EyePose
    right_command: EyePose
    left_actual: EyePose
    right_actual: EyePose
    left_last_error: np.ndarray
    right_last_error: np.ndarray


@dataclass
class BestCaptureResult:
    captured: bool
    elapsed_s: float
    capture_count: int
    total_error_deg: float
    left_pose: EyePose
    right_pose: EyePose


class WarningAbort(RuntimeError):
    """Abort the current run so the finally block can return to initial pose."""


class EmergencyStop(RuntimeError):
    """Raised when the joystick emergency button is pressed."""


def warning_abort(message: str) -> None:
    raise WarningAbort(f"Warning: {message}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle_deg(angle: float) -> float:
    normalized = (float(angle) + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def angle_error_deg(target: float, current: float) -> float:
    return normalize_angle_deg(target - current)


def circular_mean_deg(values: list[float]) -> float:
    radians = np.radians(values)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    return normalize_angle_deg(np.degrees(np.arctan2(mean_sin, mean_cos)))


class JoystickEmergency:
    def __init__(self, enabled: bool = True, triangle_button: int = TRIANGLE_BUTTON) -> None:
        self.enabled = enabled
        self.triangle_button = triangle_button
        self.pygame = None
        self.joystick = None
        self.active = False

    def open(self) -> None:
        if not self.enabled:
            return

        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("Info: no joystick detected; triangle emergency stop is disabled.")
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.active = True
        print(
            f"Joystick emergency enabled: {self.joystick.get_name()}, "
            f"triangle button index={self.triangle_button}"
        )

    def check(self) -> None:
        if not self.active or self.pygame is None or self.joystick is None:
            return

        try:
            for event in self.pygame.event.get([self.pygame.JOYBUTTONDOWN]):
                if getattr(event, "button", None) == self.triangle_button:
                    raise EmergencyStop("Triangle button pressed")
        except self.pygame.error:
            pass

        self.pygame.event.pump()
        if self.joystick.get_button(self.triangle_button):
            raise EmergencyStop("Triangle button pressed")

    def close(self) -> None:
        if self.pygame is not None:
            try:
                self.pygame.joystick.quit()
            except Exception:
                pass


def sleep_with_emergency(seconds: float, emergency: JoystickEmergency | None = None) -> None:
    deadline = time.monotonic() + seconds
    while True:
        if emergency is not None:
            emergency.check()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(JOYSTICK_POLL_INTERVAL_S, remaining))


def compute_listing_roll(yaw_deg: float, pitch_deg: float) -> float:
    numerator = math.sin(math.radians(yaw_deg)) * math.sin(math.radians(pitch_deg))
    denominator = 1.0 + (
        math.cos(math.radians(yaw_deg)) * math.cos(math.radians(pitch_deg))
    )

    if abs(denominator) < 1e-9:
        return 0.0

    value = clamp(numerator / denominator, -1.0, 1.0)
    roll_deg = math.degrees(math.asin(value))
    if math.isnan(roll_deg):
        return 0.0
    return normalize_angle_deg(roll_deg)


def load_targets(mat_path: Path, key: str) -> list[TargetPose]:
    with mat_path.open("rb") as file:
        data = scio.loadmat(file)

    if key not in data:
        available = ", ".join(k for k in data if not k.startswith("__"))
        raise KeyError(f"{key!r} not found in {mat_path}. Available keys: {available}")

    angles = np.asarray(data[key], dtype=float)
    if angles.ndim != 2 or angles.shape[1] < 4:
        raise ValueError(f"{key} must be an Nx4 numeric array, got shape {angles.shape}")

    targets: list[TargetPose] = []
    for row in angles:
        left_yaw = TARGET_GAZE_SCALE * LEFT_YAW_SIGN * row[LEFT_YAW_COL]
        left_pitch = TARGET_GAZE_SCALE * LEFT_PITCH_SIGN * row[LEFT_PITCH_COL]
        right_yaw = TARGET_GAZE_SCALE * RIGHT_YAW_SIGN * row[RIGHT_YAW_COL]
        right_pitch = TARGET_GAZE_SCALE * RIGHT_PITCH_SIGN * row[RIGHT_PITCH_COL]

        targets.append(
            TargetPose(
                left=EyePose.from_yaw_pitch(left_yaw, left_pitch),
                right=EyePose.from_yaw_pitch(right_yaw, right_pitch),
            )
        )

    return targets


def sphere_tangent_arc_cable_length(
    insertion: np.ndarray,
    pulley: np.ndarray,
    radius: float = EYEBALL_RADIUS,
) -> float:
    """Shortest cable path: insertion -> spherical arc -> tangent point -> pulley."""
    insertion = np.asarray(insertion, dtype=float)
    pulley = np.asarray(pulley, dtype=float)
    insertion_norm = float(np.linalg.norm(insertion))
    pulley_norm = float(np.linalg.norm(pulley))

    if insertion_norm < 1e-9:
        raise ValueError("Insertion point cannot be at the eyeball center")
    if pulley_norm <= radius:
        raise ValueError(
            f"Pulley must be outside eyeball sphere: |pulley|={pulley_norm:.3f}, "
            f"radius={radius:.3f}"
        )

    insertion_hat = insertion / insertion_norm
    pulley_hat = pulley / pulley_norm

    # Tangency points satisfy unit_t dot pulley_hat = radius / |pulley|.
    # The straight tangent segment length is constant for that pulley; choose
    # the tangent point that minimizes spherical arc distance from insertion.
    tangent_cos = clamp(radius / pulley_norm, -1.0, 1.0)
    tangent_sin = math.sqrt(max(0.0, 1.0 - tangent_cos * tangent_cos))
    insertion_along_pulley = clamp(float(np.dot(insertion_hat, pulley_hat)), -1.0, 1.0)
    insertion_perp = math.sqrt(max(0.0, 1.0 - insertion_along_pulley**2))

    best_tangent_dot = clamp(
        tangent_cos * insertion_along_pulley + tangent_sin * insertion_perp,
        -1.0,
        1.0,
    )
    arc_angle = math.acos(best_tangent_dot)
    arc_length = radius * arc_angle
    tangent_length = math.sqrt(max(0.0, pulley_norm * pulley_norm - radius * radius))
    return tangent_length + arc_length


def cable_length_difference(
    base_insertion: np.ndarray,
    rotated_insertion: np.ndarray,
    pulley: np.ndarray,
) -> float:
    if not USE_SPHERE_TANGENT_IK:
        return float(
            np.linalg.norm(base_insertion - pulley)
            - np.linalg.norm(rotated_insertion - pulley)
        )

    return sphere_tangent_arc_cable_length(
        base_insertion,
        pulley,
    ) - sphere_tangent_arc_cable_length(
        rotated_insertion,
        pulley,
    )


def rotation_matrix_from_zyx_deg(x: float, y: float, z: float) -> np.ndarray:
    yaw = np.radians(x)
    pitch = np.radians(y)
    roll = np.radians(z)

    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    return Rz @ (Ry @ Rx)


def calibrated_kinematic_points(
    geometry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    (
        x_lr,
        y_lr,
        rectus_x,
        rectus_z,
        oblique_x,
        oblique_y,
        oblique_z,
        posterior_pulley_x,
        lateral_pulley_y,
        superior_pulley_z,
        oblique_pulley_x,
        oblique_pulley_y,
        oblique_pulley_z,
    ) = geometry

    insertions = np.array(
        [
            [x_lr, y_lr, 0.0],
            [x_lr, -y_lr, 0.0],
            [rectus_x, 0.0, rectus_z],
            [rectus_x, 0.0, -rectus_z],
            [oblique_x, oblique_y, oblique_z],
            [oblique_x, oblique_y, -oblique_z],
        ],
        dtype=float,
    )
    pulleys = np.array(
        [
            [posterior_pulley_x, lateral_pulley_y, 0.0],
            [posterior_pulley_x, -lateral_pulley_y, 0.0],
            [posterior_pulley_x, 0.0, superior_pulley_z],
            [posterior_pulley_x, 0.0, -superior_pulley_z],
            [oblique_pulley_x, oblique_pulley_y, oblique_pulley_z],
            [oblique_pulley_x, oblique_pulley_y, -oblique_pulley_z],
        ],
        dtype=float,
    )
    return insertions, pulleys


def calc_angle_calibrated(
    x: float,
    y: float,
    z: float,
    side: str,
) -> tuple[float, float, float, float, float, float]:
    if side == "left":
        geometry = CALIBRATED_LEFT_GEOMETRY
        scale = CALIBRATED_LEFT_SCALE
        offsets = CALIBRATED_LEFT_OFFSETS
        signs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=float)
    elif side == "right":
        geometry = CALIBRATED_RIGHT_GEOMETRY
        scale = CALIBRATED_RIGHT_SCALE
        offsets = CALIBRATED_RIGHT_OFFSETS
        signs = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0], dtype=float)
    else:
        raise ValueError(f"Unknown calibrated kinematic side: {side}")

    insertions, pulleys = calibrated_kinematic_points(geometry)
    R = rotation_matrix_from_zyx_deg(x, y, z)
    rotated = (R @ insertions.T).T
    length_diff = np.linalg.norm(insertions - pulleys, axis=1) - np.linalg.norm(
        rotated - pulleys,
        axis=1,
    )
    ticks_per_mm = (
        signs
        * TICKS_PER_REV
        / CALIBRATED_BASE_RATE
        * CALIBRATED_MOTOR_GAINS
        * scale
    )
    pulses = length_diff * ticks_per_mm + offsets
    return tuple(float(value) for value in pulses)


def calc_angle_left(x: float, y: float, z: float) -> tuple[float, float, float, float, float, float]:
    if USE_CALIBRATED_KINEMATICS:
        return calc_angle_calibrated(x, y, z, "left")

    rate = KINEMATICS_RATE
    radius = EYEBALL_RADIUS

    SR = np.array([-6, 0, 12.5])
    IR = np.array([-6, 0, -12.5])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 12.5])
    IO = np.array([-6, 0, -12.5])
    S_SO = np.array([12, 24, 7])
    S_IO = np.array([12, 24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, 4.5, 0])
    S_LR = np.array([-21, -4.5, 0])

    yaw = np.radians(x)
    pitch = np.radians(y)
    roll = np.radians(z)

    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    R = Rz @ (Ry @ Rx)

    R_SO = R @ SO
    R_IO = R @ IO
    R_SR = R @ SR
    R_IR = R @ IR
    R_MR = R @ MR
    R_LR = R @ LR

    SR_R_differ = cable_length_difference(SR, R_SR, S_SR)
    IR_R_differ = cable_length_difference(IR, R_IR, S_IR)
    SO_R_differ = cable_length_difference(SO, R_SO, S_SO)
    IO_R_differ = cable_length_difference(IO, R_IO, S_IO)
    MR_R_differ = cable_length_difference(MR, R_MR, S_MR)
    LR_R_differ = cable_length_difference(LR, R_LR, S_LR)

    SR_R_pulse = SR_R_differ / rate * 4096 * 2.2
    IR_R_pulse = IR_R_differ / rate * 4096 * 2.2
    SO_R_pulse = SO_R_differ / rate * 4096 * 2 - 50
    IO_R_pulse = -IO_R_differ / rate * 4096 * 2 - 50
    MR_R_pulse = MR_R_differ / rate * 4096 * 2.2
    LR_R_pulse = LR_R_differ / rate * 4096 * 2.2

    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse


def calc_angle_right(x: float, y: float, z: float) -> tuple[float, float, float, float, float, float]:
    if USE_CALIBRATED_KINEMATICS:
        return calc_angle_calibrated(x, y, z, "right")

    rate = KINEMATICS_RATE
    radius = EYEBALL_RADIUS

    SR = np.array([-6, 0, 12.5])
    IR = np.array([-6, 0, -12.5])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 12.5])
    IO = np.array([-6, 0, -12.5])
    S_SO = np.array([12, -24, 7])
    S_IO = np.array([12, -24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, -4.5, 0])
    S_LR = np.array([-21, 4.5, 0])

    yaw = np.radians(x)
    pitch = np.radians(y)
    roll = np.radians(z)

    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    R = Rz @ (Ry @ Rx)

    R_SO = R @ SO
    R_IO = R @ IO
    R_SR = R @ SR
    R_IR = R @ IR
    R_MR = R @ MR
    R_LR = R @ LR

    SR_R_differ = cable_length_difference(SR, R_SR, S_SR)
    IR_R_differ = cable_length_difference(IR, R_IR, S_IR)
    SO_R_differ = cable_length_difference(SO, R_SO, S_SO)
    IO_R_differ = cable_length_difference(IO, R_IO, S_IO)
    MR_R_differ = cable_length_difference(MR, R_MR, S_MR)
    LR_R_differ = cable_length_difference(LR, R_LR, S_LR)

    SR_R_pulse = -SR_R_differ / rate * 4096 * 2.2
    IR_R_pulse = -IR_R_differ / rate * 4096 * 2.2
    SO_R_pulse = -SO_R_differ / rate * 4096 * 2 - 50
    IO_R_pulse = IO_R_differ / rate * 4096 * 2 - 50
    MR_R_pulse = -MR_R_differ / rate * 4096 * 2.2
    LR_R_pulse = -LR_R_differ / rate * 4096 * 2.2

    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse


def pose_to_pulses(target: TargetPose) -> list[float]:
    left = calc_angle_left(
        LEFT_IK_YAW_SIGN * target.left.yaw,
        LEFT_IK_PITCH_SIGN * target.left.pitch,
        LEFT_IK_ROLL_SIGN * target.left.roll,
    )
    right = calc_angle_right(
        RIGHT_IK_YAW_SIGN * target.right.yaw,
        RIGHT_IK_PITCH_SIGN * target.right.pitch,
        RIGHT_IK_ROLL_SIGN * target.right.roll,
    )
    pulses = [*left, *right]

    max_abs = max(abs(value) for value in pulses)
    if max_abs > SAFETY_PULSE_LIMIT:
        raise ValueError(
            f"IK pulse safety limit exceeded: max abs pulse {max_abs:.1f} > "
            f"{SAFETY_PULSE_LIMIT}"
        )

    return pulses


def unwrap_to_reference(desired_abs: int, ref_abs: int) -> int:
    k = round((ref_abs - desired_abs) / TICKS_PER_REV)
    return int(desired_abs + k * TICKS_PER_REV)


def fix_angle(value: int) -> int:
    if value > 2**31:
        value -= 2**32
    return value


class DynamixelTwoEyesController:
    def __init__(
        self,
        port: str,
        baud: int,
        do_unwrap: bool = True,
        max_goal_step: int = MAX_GOAL_STEP_TICKS,
    ) -> None:
        import dynamixel_sdk as dxl

        self.dxl = dxl
        self.port = port
        self.baud = baud
        self.do_unwrap = do_unwrap
        self.max_goal_step = max_goal_step
        self.port_handler = dxl.PortHandler(port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.group_sync_read = None
        self.initial_positions: list[int] = []
        self.prev_goal_positions: list[int] = []
        self.command_count = 0

    def open(self) -> None:
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set Dynamixel baudrate {self.baud}")

        if USE_DYNAMIXEL_PRESENT_POSITION_READ or USE_MOTOR_PRESENT_POSITION_WAIT:
            self.group_sync_read = self.dxl.GroupSyncRead(
                self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, 4
            )
            for dxl_id in DXL_IDS:
                if not self.group_sync_read.addParam(dxl_id):
                    raise RuntimeError(f"GroupSyncRead addParam failed for ID {dxl_id}")

    def check_comm(self, result: int, error: int, context: str) -> None:
        if result != self.dxl.COMM_SUCCESS:
            raise RuntimeError(f"{context}: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise RuntimeError(f"{context}: {self.packet_handler.getRxPacketError(error)}")

    def configure_extended_position_mode(self) -> None:
        for dxl_id in DXL_IDS:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE
            )
            self.check_comm(result, error, f"Set ID {dxl_id} operating mode")

    def set_profile(self, dxl_id: int, accel: int, vel: int) -> None:
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PROFILE_ACCEL, int(accel)
        )
        self.check_comm(result, error, f"Set ID {dxl_id} profile accel")
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PROFILE_VEL, int(vel)
        )
        self.check_comm(result, error, f"Set ID {dxl_id} profile vel")

    def init_profiles_all(self) -> None:
        for dxl_id in DXL_IDS:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
            if 1 <= dxl_id <= 6:
                self.set_profile(dxl_id, PROFILE_ACCEL_LEFT, PROFILE_VEL_LEFT)
            else:
                self.set_profile(dxl_id, PROFILE_ACCEL_RIGHT, PROFILE_VEL_RIGHT)
        print(
            "Dynamixel profile set: "
            f"ID1-6 accel={PROFILE_ACCEL_LEFT}, vel={PROFILE_VEL_LEFT}; "
            f"ID7-12 accel={PROFILE_ACCEL_RIGHT}, vel={PROFILE_VEL_RIGHT}"
        )

    def enable_torque(self, enable: bool) -> None:
        value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        failures: list[str] = []
        for dxl_id in DXL_IDS:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, value
            )
            if result != self.dxl.COMM_SUCCESS:
                failures.append(
                    f"ID {dxl_id}: {self.packet_handler.getTxRxResult(result)}"
                )
            elif error != 0:
                failures.append(
                    f"ID {dxl_id}: {self.packet_handler.getRxPacketError(error)}"
                )

        if failures:
            raise RuntimeError("Set torque failed for " + "; ".join(failures))

    def read_all_present_positions(self) -> list[int]:
        if self.group_sync_read is None:
            raise RuntimeError("Dynamixel present-position reads are disabled")
        current_positions = [0] * len(DXL_IDS)

        result = self.group_sync_read.txRxPacket()
        if result != self.dxl.COMM_SUCCESS:
            for i, dxl_id in enumerate(DXL_IDS):
                val, read_result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, dxl_id, ADDR_PRESENT_POSITION
                )
                self.check_comm(read_result, error, f"Read ID {dxl_id} position")
                current_positions[i] = fix_angle(val)
            return current_positions

        for i, dxl_id in enumerate(DXL_IDS):
            if self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                val = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                current_positions[i] = fix_angle(val)
            else:
                val, read_result, error = self.packet_handler.read4ByteTxRx(
                    self.port_handler, dxl_id, ADDR_PRESENT_POSITION
                )
                self.check_comm(read_result, error, f"Read ID {dxl_id} position")
                current_positions[i] = fix_angle(val)

        return current_positions

    def _int32_to_param(self, value: int) -> list[int]:
        unsigned = int(value) & 0xFFFFFFFF
        return [
            self.dxl.DXL_LOBYTE(self.dxl.DXL_LOWORD(unsigned)),
            self.dxl.DXL_HIBYTE(self.dxl.DXL_LOWORD(unsigned)),
            self.dxl.DXL_LOBYTE(self.dxl.DXL_HIWORD(unsigned)),
            self.dxl.DXL_HIBYTE(self.dxl.DXL_HIWORD(unsigned)),
        ]

    def command_absolute_positions(self, positions: list[int]) -> list[int]:
        group_sync_write = self.dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, 4
        )
        for dxl_id, goal_pos in zip(DXL_IDS, positions):
            if not group_sync_write.addParam(dxl_id, self._int32_to_param(goal_pos)):
                raise RuntimeError(f"GroupSyncWrite addParam failed for ID {dxl_id}")

        result = group_sync_write.txPacket()
        group_sync_write.clearParam()
        if result != self.dxl.COMM_SUCCESS:
            raise RuntimeError(
                f"SyncWrite failed: {self.packet_handler.getTxRxResult(result)}"
            )
        return positions

    def command_pulses(self, diff_list: list[float]) -> list[int]:
        if len(diff_list) != len(DXL_IDS):
            raise ValueError(f"Expected {len(DXL_IDS)} pulse values, got {len(diff_list)}")

        target_positions: list[int] = []
        for i, diff in enumerate(diff_list):
            desired_abs = int(round(self.initial_positions[i] + diff))
            if self.do_unwrap:
                goal = unwrap_to_reference(desired_abs, self.prev_goal_positions[i])
            else:
                goal = desired_abs

            delta = goal - self.prev_goal_positions[i]
            if delta > self.max_goal_step:
                goal = self.prev_goal_positions[i] + self.max_goal_step
            elif delta < -self.max_goal_step:
                goal = self.prev_goal_positions[i] - self.max_goal_step

            target_positions.append(int(goal))

        if DEBUG_MOTOR_COMMANDS and self.command_count < DEBUG_MOTOR_COMMAND_LIMIT:
            print(f"Motor command #{self.command_count + 1}")
            print("  ID1-6 goals :", target_positions[:6])
            print("  ID7-12 goals:", target_positions[6:])
            print(
                "  ID1-6 delta :",
                [goal - base for goal, base in zip(target_positions[:6], self.initial_positions[:6])],
            )
            print(
                "  ID7-12 delta:",
                [goal - base for goal, base in zip(target_positions[6:], self.initial_positions[6:])],
            )

        self.command_absolute_positions(target_positions)
        self.command_count += 1
        self.prev_goal_positions = target_positions.copy()
        return target_positions

    def command_correction_pulses(
        self,
        correction_list: list[float],
        gain: float,
        max_step_ticks: int,
    ) -> list[int]:
        if len(correction_list) != len(DXL_IDS):
            raise ValueError(
                f"Expected {len(DXL_IDS)} correction values, got {len(correction_list)}"
            )

        target_positions: list[int] = []
        for i, correction in enumerate(correction_list):
            step = clamp(
                gain * float(correction),
                -max_step_ticks,
                max_step_ticks,
            )
            goal = int(round(self.prev_goal_positions[i] + step))
            target_positions.append(goal)

        self.command_absolute_positions(target_positions)
        self.command_count += 1
        self.prev_goal_positions = target_positions.copy()
        return target_positions

    def wait_until_reached(
        self,
        goal_positions: list[int],
        tolerance: int = MOTOR_SETTLE_TOLERANCE_TICKS,
        timeout_s: float = MOTOR_SETTLE_TIMEOUT_S,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                present = self.read_all_present_positions()
            except RuntimeError as exc:
                warning_abort(f"present-position read failed while waiting: {exc}")

            errors = [abs(p - g) for p, g in zip(present, goal_positions)]
            if max(errors) <= tolerance:
                return True
            time.sleep(MOTOR_SETTLE_POLL_S)
        return False

    def reset_initial(self) -> None:
        goals = self.initial_positions if self.initial_positions else PERFECT_POSITIONS
        goals = [int(goal) for goal in goals]
        self.command_absolute_positions(goals)
        self.prev_goal_positions = goals.copy()

        if USE_MOTOR_PRESENT_POSITION_WAIT:
            self.wait_until_reached(goals, timeout_s=8.0)
        else:
            sleep_with_emergency(1.0)

    def reset_perfect(self, settle_wait_s: float = 1.0) -> None:
        self.command_absolute_positions(PERFECT_POSITIONS)
        self.prev_goal_positions = PERFECT_POSITIONS.copy()
        if USE_MOTOR_PRESENT_POSITION_WAIT:
            self.wait_until_reached(PERFECT_POSITIONS, timeout_s=8.0)
        time.sleep(settle_wait_s)

        if USE_DYNAMIXEL_PRESENT_POSITION_READ:
            try:
                self.initial_positions = self.read_all_present_positions()
            except RuntimeError as exc:
                warning_abort(
                    "could not read baseline present positions after reset_perfect. "
                    f"Detail: {exc}"
                )
        else:
            self.initial_positions = PERFECT_POSITIONS.copy()
        self.prev_goal_positions = self.initial_positions.copy()

    def initialize(self, reset_to_perfect: bool) -> None:
        self.open()
        self.configure_extended_position_mode()
        self.init_profiles_all()
        self.enable_torque(True)

        if USE_DYNAMIXEL_PRESENT_POSITION_READ:
            self.initial_positions = self.read_all_present_positions()
        elif reset_to_perfect:
            self.initial_positions = PERFECT_POSITIONS.copy()
        else:
            warning_abort(
                "--no-reset-perfect requires Dynamixel present-position reads, "
                "but USE_DYNAMIXEL_PRESENT_POSITION_READ is False"
            )
        self.prev_goal_positions = self.initial_positions.copy()
        if reset_to_perfect:
            self.reset_perfect()

    def close(self) -> None:
        if self.group_sync_read is not None:
            self.group_sync_read.clearParam()
        self.port_handler.closePort()


class SerialCamera:
    def __init__(
        self,
        port: str,
        baud: int = DEFAULT_CAMERA_BAUD,
        startup_timeout_s: float = CAMERA_STARTUP_TIMEOUT_S,
        capture_timeout_s: float = CAMERA_CAPTURE_TIMEOUT_S,
    ) -> None:
        self.port = port
        self.baud = baud
        self.startup_timeout_s = startup_timeout_s
        self.capture_timeout_s = capture_timeout_s
        self.roll_camera_zero_offset_deg = imu_roll_camera_zero_offset_deg(port)
        self.ser = None
        self.zero_pose: EyePose | None = None
        self.last_pose: EyePose | None = None

    def open(self) -> None:
        import serial

        self.ser = serial.Serial(self.port, self.baud, timeout=1, write_timeout=3)
        self.ser.dtr = False
        self.ser.rts = False
        time.sleep(2.0)
        self.wait_for_ready_or_ping()

    def read_line(self) -> str:
        assert self.ser is not None
        return self.ser.readline().decode("utf-8", errors="replace").strip()

    def parse_imu_line(self, line: str) -> EyePose | None:
        if not line or line == "READ_ERROR":
            return None

        lowered = line.lower()
        ignored_prefixes = (
            "ready",
            "mode",
            "output",
            "send",
            "busy",
            "info",
            "warn",
            "esp32",
            "imu i2c",
            "heap",
            "psram",
            "sensor",
            "bno055",
        )
        if lowered.startswith(ignored_prefixes):
            return None

        parts = [part.strip() for part in line.split(",")]
        try:
            if len(parts) == 3:
                if PLAIN_IMU_ORDER == "yaw,pitch,roll":
                    yaw = float(parts[0])
                    pitch = float(parts[1])
                    roll = float(parts[2])
                else:
                    yaw = float(parts[0])
                    roll = float(parts[1])
                    pitch = float(parts[2])
            elif len(parts) >= 4:
                # Labeled sketches such as: RIGHT,yaw,roll,pitch,sys,gyro,accel,mag
                yaw = float(parts[1])
                roll = float(parts[2])
                pitch = float(parts[3])
            else:
                return None
        except ValueError:
            return None

        roll = normalize_angle_deg(roll - self.roll_camera_zero_offset_deg)
        return EyePose(
            yaw=normalize_angle_deg(yaw),
            pitch=float(pitch),
            roll=roll,
        )

    def raw_to_relative_pose(self, raw_pose: EyePose) -> EyePose:
        if self.zero_pose is None:
            pose = EyePose(
                yaw=normalize_angle_deg(raw_pose.yaw),
                pitch=raw_pose.pitch,
                roll=normalize_angle_deg(raw_pose.roll),
            )
        else:
            pose = EyePose(
                yaw=angle_error_deg(raw_pose.yaw, self.zero_pose.yaw),
                pitch=raw_pose.pitch - self.zero_pose.pitch,
                roll=angle_error_deg(raw_pose.roll, self.zero_pose.roll),
            )
        self.last_pose = pose
        return pose

    def read_raw_imu_pose(self, timeout_s: float = IMU_READ_TIMEOUT_S) -> EyePose:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self.read_line()
            pose = self.parse_imu_line(line)
            if pose is not None:
                return pose
        raise TimeoutError(f"{self.port}: timed out waiting for IMU yaw,pitch,roll")

    def clear_pending_input(self) -> None:
        assert self.ser is not None
        self.ser.reset_input_buffer()

    def read_imu_pose(
        self,
        timeout_s: float = IMU_READ_TIMEOUT_S,
        fresh: bool = False,
    ) -> EyePose:
        if fresh:
            self.clear_pending_input()
        return self.raw_to_relative_pose(self.read_raw_imu_pose(timeout_s))

    def calibrate_zero(
        self,
        samples: int = IMU_ZERO_SAMPLES,
        verbose: bool = True,
    ) -> EyePose:
        raw_samples = [self.read_raw_imu_pose() for _ in range(samples)]
        self.zero_pose = EyePose(
            yaw=circular_mean_deg([pose.yaw for pose in raw_samples]),
            pitch=float(np.mean([pose.pitch for pose in raw_samples])),
            roll=circular_mean_deg([pose.roll for pose in raw_samples]),
        )
        self.last_pose = EyePose(0.0, 0.0, 0.0)
        if verbose:
            print(
                f"{self.port}: IMU zero yaw={self.zero_pose.yaw:.2f}, "
                f"pitch={self.zero_pose.pitch:.2f}, roll={self.zero_pose.roll:.2f}"
            )
        return self.zero_pose

    def read_exact(self, size: int, timeout_s: float) -> bytes:
        assert self.ser is not None
        data = bytearray()
        deadline = time.monotonic() + timeout_s

        while len(data) < size:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.port}: timed out after {len(data)} / {size} bytes")
            chunk = self.ser.read(min(16384, size - len(data)))
            if chunk:
                data.extend(chunk)

        return bytes(data)

    def wait_for_ready_or_ping(self) -> None:
        assert self.ser is not None
        deadline = time.monotonic() + self.startup_timeout_s

        while time.monotonic() < deadline:
            line = self.read_line()
            if line.startswith("READY "):
                return

        for _ in range(5):
            self.ser.reset_input_buffer()
            self.ser.write(b"ping\n")
            self.ser.flush()

            ping_deadline = time.monotonic() + 1.5
            while time.monotonic() < ping_deadline:
                line = self.read_line()
                if line.startswith("PONG READY"):
                    return
                if line.startswith("PONG NOT_READY"):
                    raise RuntimeError(f"{self.port}: camera is not ready")

        raise RuntimeError(f"{self.port}: could not confirm camera readiness")

    def receive_capture(self) -> bytes:
        assert self.ser is not None
        self.ser.reset_input_buffer()
        self.ser.write(b"c\n")
        self.ser.flush()

        while True:
            line = self.read_line()
            if not line:
                continue

            if line.startswith("IMG "):
                image_size = int(line.split()[1])
                image = self.read_exact(image_size, self.capture_timeout_s)
                marker = self.read_line()
                if marker != "END":
                    warning_abort(f"{self.port}: expected END marker, got {marker!r}")
                if not image.startswith(b"\xff\xd8"):
                    raise RuntimeError(f"{self.port}: JPEG SOI marker missing")
                if not image.endswith(b"\xff\xd9"):
                    warning_abort(f"{self.port}: JPEG EOI marker missing")
                return image

            if line.startswith("ERROR "):
                raise RuntimeError(f"{self.port}: {line}")

            pose = self.parse_imu_line(line)
            if pose is not None:
                self.raw_to_relative_pose(pose)

    def capture_to(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.receive_capture()
        output_path.write_bytes(image)
        return output_path

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None


def force_window_to_front(
    hwnd: int | None,
    always_on_top: bool = False,
) -> None:
    if not hwnd or not hasattr(ctypes, "windll"):
        return

    user32 = ctypes.windll.user32
    hwnd_topmost = -1
    hwnd_notopmost = -2
    sw_show = 5
    swp_nomove = 0x0002
    swp_nosize = 0x0001
    swp_showwindow = 0x0040

    user32.ShowWindow(hwnd, sw_show)
    user32.SetWindowPos(
        hwnd, hwnd_topmost, 0, 0, 0, 0, swp_nomove | swp_nosize | swp_showwindow
    )
    user32.SetForegroundWindow(hwnd)
    user32.SetActiveWindow(hwnd)
    user32.SetFocus(hwnd)
    if not always_on_top:
        user32.SetWindowPos(
            hwnd, hwnd_notopmost, 0, 0, 0, 0, swp_nomove | swp_nosize | swp_showwindow
        )


class FullscreenPresenter:
    def __init__(
        self,
        display_index: int = 0,
        bring_to_front: bool = True,
        always_on_top: bool = True,
        borderless_fullscreen: bool = True,
    ) -> None:
        import pygame

        self.pygame = pygame
        self.always_on_top = always_on_top
        pygame.init()
        pygame.display.set_caption("Healthbaseline replay capture")
        display_count = pygame.display.get_num_displays()
        desktop_sizes = pygame.display.get_desktop_sizes()
        if display_index < 0:
            display_index = 0
        if display_count and display_index >= display_count:
            print(
                f"Warning: display index {display_index} is not available; "
                "using display 0 instead."
            )
            display_index = 0
        self.display_index = display_index
        display_size = (
            desktop_sizes[self.display_index]
            if self.display_index < len(desktop_sizes)
            else (0, 0)
        )
        display_flags = pygame.NOFRAME if borderless_fullscreen else pygame.FULLSCREEN

        try:
            self.screen = pygame.display.set_mode(
                display_size,
                display_flags,
                display=self.display_index,
            )
        except (TypeError, pygame.error):
            if self.display_index != 0:
                print(
                    f"Warning: pygame could not open display {self.display_index}; "
                    "using display 0 instead."
                )
            self.display_index = 0
            fallback_size = (
                desktop_sizes[0]
                if desktop_sizes and borderless_fullscreen
                else (0, 0)
            )
            self.screen = pygame.display.set_mode(fallback_size, display_flags)

        pygame.mouse.set_visible(False)
        self.size = self.screen.get_size()
        self.hwnd = pygame.display.get_wm_info().get("window")
        self.current_image = None
        self.closed = False
        if bring_to_front:
            force_window_to_front(
                int(self.hwnd) if self.hwnd else None,
                always_on_top=self.always_on_top,
            )

    def process_window_events(self) -> None:
        for event in self.pygame.event.get([self.pygame.QUIT, self.pygame.KEYDOWN]):
            if event.type == self.pygame.QUIT:
                self.closed = True
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                self.closed = True

    def show(self, image_path: Path) -> bool:
        self.process_window_events()

        if self.closed:
            return False

        image = self.pygame.image.load(str(image_path)).convert()
        self.current_image = self.pygame.transform.smoothscale(image, self.size)
        return self.draw()

    def draw(self) -> bool:
        self.process_window_events()

        if self.closed:
            return False
        if self.current_image is not None:
            self.screen.blit(self.current_image, (0, 0))
        self.pygame.display.flip()
        return True

    def pump_for(
        self,
        seconds: float,
        emergency: JoystickEmergency | None = None,
    ) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if emergency is not None:
                emergency.check()
            if not self.draw():
                return False
            time.sleep(0.03)
        return True

    def close(self) -> None:
        self.pygame.quit()


def max_yaw_pitch_delta(current: EyePose, target: EyePose) -> float:
    return max(abs(angle_error_deg(target.yaw, current.yaw)), abs(target.pitch - current.pitch))


def max_pose_delta(current: EyePose, target: EyePose) -> float:
    return max(
        abs(angle_error_deg(target.yaw, current.yaw)),
        abs(target.pitch - current.pitch),
        abs(angle_error_deg(target.roll, current.roll)),
    )


def pose_abs_error_sum(current: EyePose, target: EyePose) -> float:
    error = pose_error(current, target)
    return abs(error.yaw) + abs(error.pitch) + abs(error.roll)


def two_eye_abs_error_sum(
    left_current: EyePose,
    right_current: EyePose,
    target: TargetPose,
) -> float:
    return pose_abs_error_sum(left_current, target.left) + pose_abs_error_sum(
        right_current, target.right
    )


def is_eye_reached(current: EyePose, target: EyePose, tolerance_deg: float) -> bool:
    return max_pose_delta(current, target) <= tolerance_deg


def pose_error(current: EyePose, target: EyePose) -> EyePose:
    return EyePose(
        angle_error_deg(target.yaw, current.yaw),
        target.pitch - current.pitch,
        angle_error_deg(target.roll, current.roll),
    )


def apply_min_pd_step(command_ticks: float, error_deg: float) -> float:
    if abs(error_deg) < FINE_PD_MIN_STEP_ERROR_DEG:
        return command_ticks
    if command_ticks == 0.0 or abs(command_ticks) >= FINE_PD_MIN_STEP_TICKS:
        return command_ticks
    return math.copysign(FINE_PD_MIN_STEP_TICKS, command_ticks)


def pd_pair_correction_for_eye(
    current: EyePose,
    target: EyePose,
    last_error: np.ndarray | None,
    signs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    tolerance_deg: float = FINE_ANGLE_TOLERANCE_DEG,
) -> tuple[list[float], np.ndarray]:
    error_pose = pose_error(current, target)
    error = error_pose.as_array()
    derivative = np.zeros(3) if last_error is None else error - last_error
    kp = np.array([FINE_PD_KP_YAW, FINE_PD_KP_PITCH, FINE_PD_KP_ROLL], dtype=float)
    kd = np.array([FINE_PD_KD_YAW, FINE_PD_KD_PITCH, FINE_PD_KD_ROLL], dtype=float)
    axis_commands = kp * error + kd * derivative

    correction = [0.0] * 6
    for axis_index, (motor_a, motor_b) in enumerate(FINE_PD_MOTOR_PAIRS):
        if abs(error[axis_index]) <= tolerance_deg:
            continue

        axis_command = clamp(
            float(axis_commands[axis_index]),
            -FINE_PD_MAX_STEP_TICKS,
            FINE_PD_MAX_STEP_TICKS,
        )
        axis_command = apply_min_pd_step(axis_command, float(error[axis_index]))
        sign_a, sign_b = signs[axis_index]
        correction[motor_a] += sign_a * axis_command
        correction[motor_b] += sign_b * axis_command

    return correction, error


def apply_min_ik_pd_step(command_ticks: float, error_ticks: float) -> float:
    if abs(error_ticks) < FINE_IK_PD_MIN_STEP_ERROR_TICKS:
        return command_ticks
    if command_ticks == 0.0 or abs(command_ticks) >= FINE_IK_PD_MIN_STEP_TICKS:
        return command_ticks
    return math.copysign(FINE_IK_PD_MIN_STEP_TICKS, command_ticks)


def ik_pd_correction_for_target(
    current_left: EyePose,
    current_right: EyePose,
    target_pulses: list[float],
    last_pulse_error: np.ndarray | None,
    left_reached: bool,
    right_reached: bool,
) -> tuple[list[float], np.ndarray]:
    actual_pose = TargetPose(left=current_left, right=current_right)
    actual_pulses = np.array(pose_to_pulses(actual_pose), dtype=float)
    pulse_error = np.array(target_pulses, dtype=float) - actual_pulses
    derivative = (
        np.zeros_like(pulse_error)
        if last_pulse_error is None
        else pulse_error - last_pulse_error
    )
    raw_commands = FINE_IK_PD_KP * pulse_error + FINE_IK_PD_KD * derivative

    correction: list[float] = []
    for motor_index, (command_ticks, error_ticks) in enumerate(
        zip(raw_commands, pulse_error)
    ):
        if (motor_index < 6 and left_reached) or (
            motor_index >= 6 and right_reached
        ):
            correction.append(0.0)
            continue

        command = clamp(
            float(command_ticks),
            -FINE_IK_PD_MAX_STEP_TICKS,
            FINE_IK_PD_MAX_STEP_TICKS,
        )
        correction.append(apply_min_ik_pd_step(command, float(error_ticks)))

    return correction, pulse_error


def format_pose(pose: EyePose) -> str:
    return f"(yaw={pose.yaw:.2f}, pitch={pose.pitch:.2f}, roll={pose.roll:.2f})"


def print_pose_feedback(
    label: str,
    target: TargetPose,
    left_actual: EyePose,
    right_actual: EyePose,
) -> None:
    if not PRINT_POSE_FEEDBACK:
        return

    left_error = pose_error(left_actual, target.left)
    right_error = pose_error(right_actual, target.right)
    print(
        f"{label} L target={format_pose(target.left)} "
        f"current={format_pose(left_actual)} error={format_pose(left_error)}"
    )
    print(
        f"{label} R target={format_pose(target.right)} "
        f"current={format_pose(right_actual)} error={format_pose(right_error)}"
    )


def command_pose_and_wait(
    controller: DynamixelTwoEyesController,
    left_pose: EyePose,
    right_pose: EyePose,
    timeout_s: float = MOTOR_SETTLE_TIMEOUT_S,
    settle_delay_s: float = MOTOR_IK_COMMAND_DELAY_S,
    emergency: JoystickEmergency | None = None,
) -> bool:
    target = TargetPose(left=left_pose, right=right_pose)
    goals = controller.command_pulses(pose_to_pulses(target))
    if not USE_MOTOR_PRESENT_POSITION_WAIT:
        sleep_with_emergency(settle_delay_s, emergency)
        return True
    return controller.wait_until_reached(goals, timeout_s=timeout_s)


def read_both_imu_poses(
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    timeout_s: float = IMU_READ_TIMEOUT_S,
) -> tuple[EyePose, EyePose]:
    left_camera.clear_pending_input()
    right_camera.clear_pending_input()
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(left_camera.read_imu_pose, timeout_s)
        right_future = pool.submit(right_camera.read_imu_pose, timeout_s)
        return left_future.result(), right_future.result()


def run_ik_feedback_stage(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    target_pulses: list[float],
    stage_name: str,
    tolerance_deg: float,
    max_iterations: int,
    gain: float,
    max_step_ticks: int,
    emergency: JoystickEmergency | None = None,
    print_feedback: bool = True,
) -> bool:
    for step_index in range(max_iterations):
        if emergency is not None:
            emergency.check()

        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)

        left_reached = is_eye_reached(state.left_actual, target.left, tolerance_deg)
        right_reached = is_eye_reached(state.right_actual, target.right, tolerance_deg)
        if left_reached and right_reached:
            return True

        if print_feedback:
            print_pose_feedback(
                f"{stage_name} IK feedback step {step_index + 1:02d}",
                target,
                state.left_actual,
                state.right_actual,
            )

        # Re-run IK from the IMU pose: calc_angle_left/right rotate the current
        # insertion points and convert their cable-length differences to ticks.
        actual_pose = TargetPose(left=state.left_actual, right=state.right_actual)
        actual_pulses = pose_to_pulses(actual_pose)
        correction_pulses = [
            target_value - actual_value
            for target_value, actual_value in zip(target_pulses, actual_pulses)
        ]

        if left_reached:
            correction_pulses[:6] = [0.0] * 6
        if right_reached:
            correction_pulses[6:] = [0.0] * 6

        controller.command_correction_pulses(
            correction_pulses,
            gain=gain,
            max_step_ticks=max_step_ticks,
        )
        sleep_with_emergency(COARSE_IK_FEEDBACK_DELAY_S, emergency)

        state.left_command = target.left
        state.right_command = target.right

    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
    return is_eye_reached(
        state.left_actual, target.left, tolerance_deg
    ) and is_eye_reached(
        state.right_actual, target.right, tolerance_deg
    )


def run_pd_feedback_stage(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    tolerance_deg: float,
    max_iterations: int,
    emergency: JoystickEmergency | None = None,
) -> bool:
    target_pulses = pose_to_pulses(target)
    last_pulse_error: np.ndarray | None = None

    for step_index in range(max_iterations):
        if emergency is not None:
            emergency.check()

        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)

        left_reached = is_eye_reached(state.left_actual, target.left, tolerance_deg)
        right_reached = is_eye_reached(state.right_actual, target.right, tolerance_deg)
        if left_reached and right_reached:
            return True

        print_pose_feedback(
            f"Fine IK-PD feedback step {step_index + 1:02d}",
            target,
            state.left_actual,
            state.right_actual,
        )

        correction, pulse_error = ik_pd_correction_for_target(
            state.left_actual,
            state.right_actual,
            target_pulses,
            last_pulse_error,
            left_reached,
            right_reached,
        )

        if any(value != 0.0 for value in correction):
            controller.command_correction_pulses(
                correction,
                gain=1.0,
                max_step_ticks=FINE_IK_PD_MAX_STEP_TICKS,
            )
        sleep_with_emergency(FINE_PD_LOOP_DELAY_S, emergency)

        last_pulse_error = pulse_error
        state.left_last_error = pose_error(state.left_actual, target.left).as_array()
        state.right_last_error = pose_error(state.right_actual, target.right).as_array()
        state.left_command = target.left
        state.right_command = target.right

    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
    return is_eye_reached(
        state.left_actual, target.left, tolerance_deg
    ) and is_eye_reached(
        state.right_actual, target.right, tolerance_deg
    )


def run_timed_pd_feedback_stage(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    timeout_s: float,
    tolerance_deg: float,
    emergency: JoystickEmergency | None = None,
    print_every_steps: int = 10,
) -> tuple[bool, float]:
    target_pulses = pose_to_pulses(target)
    last_pulse_error: np.ndarray | None = None
    start = time.monotonic()
    deadline = start + timeout_s
    step_index = 0

    while time.monotonic() < deadline and step_index < FINE_MAX_ITERATIONS:
        if emergency is not None:
            emergency.check()

        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)

        left_reached = is_eye_reached(state.left_actual, target.left, tolerance_deg)
        right_reached = is_eye_reached(state.right_actual, target.right, tolerance_deg)
        if left_reached and right_reached:
            return True, time.monotonic() - start

        if print_every_steps and (
            print_every_steps <= 1 or step_index == 0 or step_index % print_every_steps == 0
        ):
            print_pose_feedback(
                f"Timed Fine IK-PD step {step_index + 1:03d}",
                target,
                state.left_actual,
                state.right_actual,
            )

        correction, pulse_error = ik_pd_correction_for_target(
            state.left_actual,
            state.right_actual,
            target_pulses,
            last_pulse_error,
            left_reached,
            right_reached,
        )

        if any(value != 0.0 for value in correction):
            controller.command_correction_pulses(
                correction,
                gain=1.0,
                max_step_ticks=FINE_IK_PD_MAX_STEP_TICKS,
            )
        sleep_with_emergency(FINE_PD_LOOP_DELAY_S, emergency)

        last_pulse_error = pulse_error
        state.left_last_error = pose_error(state.left_actual, target.left).as_array()
        state.right_last_error = pose_error(state.right_actual, target.right).as_array()
        state.left_command = target.left
        state.right_command = target.right
        step_index += 1

    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
    return (
        is_eye_reached(state.left_actual, target.left, tolerance_deg)
        and is_eye_reached(state.right_actual, target.right, tolerance_deg),
        time.monotonic() - start,
    )


def run_initial_big_step_and_coarse(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    emergency: JoystickEmergency | None = None,
    print_feedback: bool = True,
) -> bool:
    command_pose_and_wait(
        controller,
        target.left,
        target.right,
        timeout_s=MOTOR_SETTLE_TIMEOUT_S,
        settle_delay_s=MOTOR_IK_COMMAND_DELAY_S,
        emergency=emergency,
    )
    state.left_command = target.left
    state.right_command = target.right
    state.left_last_error = np.zeros(3)
    state.right_last_error = np.zeros(3)
    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
    if print_feedback:
        print_pose_feedback("After initial big IK", target, state.left_actual, state.right_actual)

    target_pulses = pose_to_pulses(target)
    return run_ik_feedback_stage(
        controller,
        left_camera,
        right_camera,
        state,
        target,
        target_pulses,
        stage_name="Initial coarse",
        tolerance_deg=COARSE_ANGLE_TOLERANCE_DEG,
        max_iterations=COARSE_MAX_ITERATIONS,
        gain=COARSE_IK_FEEDBACK_GAIN,
        max_step_ticks=COARSE_IK_FEEDBACK_MAX_STEP_TICKS,
        emergency=emergency,
        print_feedback=print_feedback,
    )


def drive_to_target(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    emergency: JoystickEmergency | None = None,
) -> bool:
    if emergency is not None:
        emergency.check()

    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)

    if not ENABLE_IMU_FINE_TUNING:
        command_pose_and_wait(
            controller,
            target.left,
            target.right,
            timeout_s=MOTOR_SETTLE_TIMEOUT_S,
            settle_delay_s=MOTOR_IK_COMMAND_DELAY_S,
            emergency=emergency,
        )
        state.left_command = target.left
        state.right_command = target.right
        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
        print_pose_feedback("After IK", target, state.left_actual, state.right_actual)
        return True

    target_pulses = pose_to_pulses(target)
    left_large = max_yaw_pitch_delta(state.left_actual, target.left) > IK_THRESHOLD_DEG
    right_large = max_yaw_pitch_delta(state.right_actual, target.right) > IK_THRESHOLD_DEG

    if left_large or right_large:
        motors_settled = command_pose_and_wait(
            controller,
            target.left,
            target.right,
            timeout_s=MOTOR_SETTLE_TIMEOUT_S,
            settle_delay_s=MOTOR_IK_COMMAND_DELAY_S,
            emergency=emergency,
        )
        if not motors_settled:
            warning_abort(
                "motor present positions did not fully settle after IK"
            )
        state.left_command = target.left
        state.right_command = target.right
        state.left_last_error = np.zeros(3)
        state.right_last_error = np.zeros(3)
        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
        print_pose_feedback("After large IK", target, state.left_actual, state.right_actual)

    coarse_reached = run_ik_feedback_stage(
        controller,
        left_camera,
        right_camera,
        state,
        target,
        target_pulses,
        stage_name="Coarse",
        tolerance_deg=COARSE_ANGLE_TOLERANCE_DEG,
        max_iterations=COARSE_MAX_ITERATIONS,
        gain=COARSE_IK_FEEDBACK_GAIN,
        max_step_ticks=COARSE_IK_FEEDBACK_MAX_STEP_TICKS,
        emergency=emergency,
    )
    if not coarse_reached:
        left_error = max_pose_delta(state.left_actual, target.left)
        right_error = max_pose_delta(state.right_actual, target.right)
        max_error = max(left_error, right_error)
        if max_error > COARSE_ACCEPT_MAX_ERROR_DEG:
            return False

        print(
            "Coarse IK did not fully reach "
            f"{COARSE_ANGLE_TOLERANCE_DEG:.2f} deg, but max error "
            f"{max_error:.2f} deg is within the fine-entry limit "
            f"{COARSE_ACCEPT_MAX_ERROR_DEG:.2f} deg. Continuing to fine."
        )

    return run_pd_feedback_stage(
        controller,
        left_camera,
        right_camera,
        state,
        target,
        tolerance_deg=FINE_ANGLE_TOLERANCE_DEG,
        max_iterations=FINE_MAX_ITERATIONS,
        emergency=emergency,
    )


def frame_path_for_index(frame_dir: Path, index: int) -> Path:
    return frame_dir / f"{index}.jpg"


def validate_frame_files(frame_dir: Path, start: int, end: int, step: int = 1) -> None:
    missing = [
        idx for idx in range(start, end, step)
        if not frame_path_for_index(frame_dir, idx).exists()
    ]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} frame images are missing in {frame_dir}. "
            f"First missing indices: {preview}"
        )


def capture_both_cameras(
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    left_path: Path,
    right_path: Path,
) -> tuple[Path, Path]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(left_camera.capture_to, left_path)
        right_future = pool.submit(right_camera.capture_to, right_path)
        return left_future.result(), right_future.result()


def capture_both_cameras_while_presenting(
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    left_path: Path,
    right_path: Path,
    presenter: FullscreenPresenter,
    emergency: JoystickEmergency | None = None,
) -> tuple[Path, Path]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(left_camera.capture_to, left_path)
        right_future = pool.submit(right_camera.capture_to, right_path)

        while not (left_future.done() and right_future.done()):
            if emergency is not None:
                emergency.check()
            if not presenter.draw():
                raise RuntimeError("Fullscreen display was closed during capture.")
            time.sleep(0.03)

        return left_future.result(), right_future.result()


def make_log_writer(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "frame_index",
            "left_target_yaw",
            "left_target_pitch",
            "left_target_roll",
            "right_target_yaw",
            "right_target_pitch",
            "right_target_roll",
            "left_actual_yaw",
            "left_actual_pitch",
            "left_actual_roll",
            "right_actual_yaw",
            "right_actual_pitch",
            "right_actual_roll",
            "left_image",
            "right_image",
        ]
    )
    return file, writer


def make_capture_log_writer(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "frame_index",
            "fine_reached",
            "fine_elapsed_s",
            "left_max_error_deg",
            "right_max_error_deg",
            "left_target_yaw",
            "left_target_pitch",
            "left_target_roll",
            "right_target_yaw",
            "right_target_pitch",
            "right_target_roll",
            "left_actual_yaw",
            "left_actual_pitch",
            "left_actual_roll",
            "right_actual_yaw",
            "right_actual_pitch",
            "right_actual_roll",
            "left_image",
            "right_image",
        ]
    )
    return file, writer


def format_pose_summary(pose: EyePose) -> str:
    return f"yaw={pose.yaw:.2f}, pitch={pose.pitch:.2f}, roll={pose.roll:.2f}"


def make_capture_journal_entry(
    frame_index: int,
    target: TargetPose,
    left_actual: EyePose,
    right_actual: EyePose,
    fine_reached: bool,
    fine_elapsed_s: float,
    left_error: float,
    right_error: float,
) -> str:
    return (
        f"第{frame_index:04d}张照片拍好了 | "
        f"target L({format_pose_summary(target.left)}) "
        f"R({format_pose_summary(target.right)}) | "
        f"拍摄角度 L({format_pose_summary(left_actual)}) "
        f"R({format_pose_summary(right_actual)}) | "
        f"fine_reached={fine_reached} "
        f"elapsed={fine_elapsed_s:.2f}s "
        f"L_err={left_error:.2f}deg R_err={right_error:.2f}deg"
    )


def make_capture_skipped_journal_entry(
    frame_index: int,
    target: TargetPose,
    left_actual: EyePose,
    right_actual: EyePose,
    elapsed_s: float,
    left_error: float,
    right_error: float,
) -> str:
    return (
        f"第{frame_index:04d}张照片未保存 | "
        f"target L({format_pose_summary(target.left)}) "
        f"R({format_pose_summary(target.right)}) | "
        f"最终角度 L({format_pose_summary(left_actual)}) "
        f"R({format_pose_summary(right_actual)}) | "
        f"elapsed={elapsed_s:.2f}s "
        f"L_err={left_error:.2f}deg R_err={right_error:.2f}deg | "
        "原因：限定时间内没有进入误差容忍范围"
    )


def show_frame_and_capture_pair(
    presenter: FullscreenPresenter,
    frame_path: Path,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    left_path: Path,
    right_path: Path,
    display_settle_s: float,
    emergency: JoystickEmergency | None = None,
) -> tuple[Path, Path]:
    if not presenter.show(frame_path):
        raise RuntimeError("Fullscreen display was closed before capture.")
    if not presenter.pump_for(display_settle_s, emergency):
        raise RuntimeError("Fullscreen display was closed before capture.")

    return capture_both_cameras_while_presenting(
        left_camera,
        right_camera,
        left_path,
        right_path,
        presenter,
        emergency,
    )


def remove_capture_outputs(left_path: Path, right_path: Path) -> None:
    for path in (left_path, right_path):
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Could not remove old capture file {path}: {exc}") from exc


def capture_candidate_with_retries(
    frame_index: int,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    left_path: Path,
    right_path: Path,
    presenter: FullscreenPresenter,
    emergency: JoystickEmergency | None,
    capture_retries: int,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(capture_retries + 1):
        try:
            capture_both_cameras_while_presenting(
                left_camera,
                right_camera,
                left_path,
                right_path,
                presenter,
                emergency,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= capture_retries:
                raise
            print(
                f"[{frame_index:04d}] Candidate capture failed ({exc}); "
                f"retrying {attempt + 1}/{capture_retries}..."
            )

    if last_exc is not None:
        raise last_exc


def run_timed_pd_best_capture_stage(
    controller: DynamixelTwoEyesController,
    left_camera: SerialCamera,
    right_camera: SerialCamera,
    state: ReplayState,
    target: TargetPose,
    presenter: FullscreenPresenter,
    frame_path: Path,
    left_path: Path,
    right_path: Path,
    frame_index: int,
    timeout_s: float,
    tolerance_deg: float,
    display_settle_s: float,
    capture_retries: int,
    emergency: JoystickEmergency | None = None,
    print_every_steps: int = 0,
) -> BestCaptureResult:
    if not presenter.show(frame_path):
        raise RuntimeError("Fullscreen display was closed before capture.")
    if not presenter.pump_for(display_settle_s, emergency):
        raise RuntimeError("Fullscreen display was closed before capture.")

    remove_capture_outputs(left_path, right_path)

    target_pulses = pose_to_pulses(target)
    last_pulse_error: np.ndarray | None = None
    start = time.monotonic()
    deadline = start + timeout_s
    step_index = 0
    latest_score = math.inf

    while time.monotonic() < deadline and step_index < FINE_MAX_ITERATIONS:
        if emergency is not None:
            emergency.check()

        state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
        left_reached = is_eye_reached(state.left_actual, target.left, tolerance_deg)
        right_reached = is_eye_reached(state.right_actual, target.right, tolerance_deg)
        latest_score = two_eye_abs_error_sum(state.left_actual, state.right_actual, target)

        if left_reached and right_reached:
            capture_candidate_with_retries(
                frame_index,
                left_camera,
                right_camera,
                left_path,
                right_path,
                presenter,
                emergency,
                capture_retries,
            )
            return BestCaptureResult(
                captured=True,
                elapsed_s=time.monotonic() - start,
                capture_count=1,
                total_error_deg=latest_score,
                left_pose=state.left_actual,
                right_pose=state.right_actual,
            )

        if print_every_steps and (
            print_every_steps <= 1 or step_index == 0 or step_index % print_every_steps == 0
        ):
            print_pose_feedback(
                f"Timed Fine IK-PD/capture step {step_index + 1:03d}",
                target,
                state.left_actual,
                state.right_actual,
            )

        correction, pulse_error = ik_pd_correction_for_target(
            state.left_actual,
            state.right_actual,
            target_pulses,
            last_pulse_error,
            left_reached,
            right_reached,
        )

        if any(value != 0.0 for value in correction):
            controller.command_correction_pulses(
                correction,
                gain=1.0,
                max_step_ticks=FINE_IK_PD_MAX_STEP_TICKS,
            )

        last_pulse_error = pulse_error
        state.left_last_error = pose_error(state.left_actual, target.left).as_array()
        state.right_last_error = pose_error(state.right_actual, target.right).as_array()
        state.left_command = target.left
        state.right_command = target.right
        step_index += 1

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not presenter.pump_for(min(FINE_PD_LOOP_DELAY_S, remaining), emergency):
            raise RuntimeError("Fullscreen display was closed during fine/capture window.")

    state.left_actual, state.right_actual = read_both_imu_poses(left_camera, right_camera)
    latest_score = two_eye_abs_error_sum(state.left_actual, state.right_actual, target)

    remove_capture_outputs(left_path, right_path)
    return BestCaptureResult(
        captured=False,
        elapsed_s=time.monotonic() - start,
        capture_count=0,
        total_error_deg=latest_score,
        left_pose=state.left_actual,
        right_pose=state.right_actual,
    )


def run_ik_step_test(args: argparse.Namespace) -> None:
    targets = load_targets(args.mat_path, args.mat_key)
    if args.ik_test_index < 0 or args.ik_test_index >= len(targets):
        raise ValueError(
            f"--ik-test-index must be in [0, {len(targets) - 1}], got {args.ik_test_index}"
        )

    target = targets[args.ik_test_index]
    pulses = pose_to_pulses(target)
    if args.ik_test_side == "left":
        pulses = [*pulses[:6], *([0.0] * 6)]
    elif args.ik_test_side == "right":
        pulses = [*([0.0] * 6), *pulses[6:]]

    print(f"Loaded {len(targets)} target samples from {args.mat_path}")
    print(f"IK step test index: {args.ik_test_index}")
    print(f"IK step test side: {args.ik_test_side}")
    print(f"Target gaze scale = {TARGET_GAZE_SCALE:.2f}")
    print(
        f"Target L=({target.left.yaw:.2f}, {target.left.pitch:.2f}, {target.left.roll:.2f}) "
        f"R=({target.right.yaw:.2f}, {target.right.pitch:.2f}, {target.right.roll:.2f})"
    )
    print(f"KINEMATICS_RATE = {KINEMATICS_RATE}")
    print("Pulse offsets:", [round(float(value), 1) for value in pulses])
    print(
        "Approx goals from PERFECT_POSITIONS:",
        [round(float(base + pulse), 1) for base, pulse in zip(PERFECT_POSITIONS, pulses)],
    )

    if args.dry_run:
        print("Dry run OK. IK step test did not open Dynamixel port.")
        return

    controller = DynamixelTwoEyesController(args.dxl_port, args.dxl_baud)
    emergency = JoystickEmergency(enabled=not args.no_joystick_emergency)
    reset_done = False

    try:
        emergency.open()
        controller.initialize(reset_to_perfect=not args.no_reset_perfect)
        goals = controller.command_pulses(pulses)
        print("Absolute goal positions:", goals)
        print(f"Holding target for {args.ik_hold_s:.1f} seconds...")
        sleep_with_emergency(args.ik_hold_s, emergency)
        print("Returning to initial position...")
        controller.reset_initial()
        reset_done = True
        print("IK step test finished.")

    except EmergencyStop as exc:
        print(f"Emergency stop requested: {exc}")
    except WarningAbort as exc:
        print(str(exc))

    finally:
        if not reset_done:
            try:
                print("Returning to initial position after interrupted IK test...")
                controller.reset_initial()
            except Exception as exc:
                print(f"Error: reset_initial failed: {exc}")
        try:
            controller.enable_torque(False)
        except Exception as exc:
            print(f"Error: disabling torque failed: {exc}")
        emergency.close()
        controller.close()


def run_capture_replay(args: argparse.Namespace) -> None:
    targets = load_targets(args.mat_path, args.mat_key)
    end_index = args.end_index if args.end_index is not None else len(targets)
    end_index = min(end_index, len(targets))
    sample_step = int(getattr(args, "sample_step", 1))
    if sample_step <= 0:
        raise ValueError("sample_step must be a positive integer")
    fine_tolerance_deg = float(getattr(args, "fine_tolerance_deg", FINE_ANGLE_TOLERANCE_DEG))
    if fine_tolerance_deg <= 0:
        raise ValueError("fine_tolerance_deg must be positive")

    if args.start_index < 0 or args.start_index >= end_index:
        raise ValueError("Invalid start/end index range")

    frame_indices = list(range(args.start_index, end_index, sample_step))
    if not frame_indices:
        raise ValueError("No capture frames selected")

    validate_frame_files(args.frame_dir, args.start_index, end_index, step=sample_step)
    args.left_save_dir.mkdir(parents=True, exist_ok=True)
    args.right_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(targets)} target samples from {args.mat_path}")
    print(f"Capture replay samples [{args.start_index}, {end_index}) step={sample_step}")
    print(
        f"Selected {len(frame_indices)} / {end_index - args.start_index} samples: "
        f"{frame_indices[0]} ... {frame_indices[-1]}"
    )
    print(f"Expected captures: {len(frame_indices)} left + {len(frame_indices)} right")
    print(f"Target gaze scale = {TARGET_GAZE_SCALE:.2f}")
    print(f"Fine tolerance = {fine_tolerance_deg:.2f} deg")
    print(
        "Timing: first fine "
        f"{args.first_fine_timeout_s:.1f}s; later fine {args.fine_timeout_s:.1f}s; "
        f"display settle {args.display_settle_s:.2f}s; "
        f"first frame extra settle {getattr(args, 'first_frame_extra_settle_s', 0.0):.2f}s"
    )
    print(
        f"Fullscreen display index = {getattr(args, 'display_index', 0)} "
        f"(topmost={bool(getattr(args, 'bring_display_to_front', True))})"
    )
    print(
        "Initial stage: first target uses big IK + coarse IK. "
        "Later targets use timed Fine IK-PD. If a target times out, it is "
        "skipped with no recovery attempt. The run stops after "
        f"{CAPTURE_MAX_CONSECUTIVE_MISSES} consecutive missed captures."
    )

    if args.dry_run:
        max_abs_pulse = 0.0
        for frame_index in frame_indices:
            max_abs_pulse = max(
                max_abs_pulse,
                max(abs(v) for v in pose_to_pulses(targets[frame_index])),
            )
        print(f"Dry run OK. Max abs IK pulse: {max_abs_pulse:.1f}")
        return

    controller = DynamixelTwoEyesController(args.dxl_port, args.dxl_baud)
    left_camera = SerialCamera(args.left_camera_port, args.camera_baud)
    right_camera = SerialCamera(args.right_camera_port, args.camera_baud)
    emergency = JoystickEmergency(enabled=not args.no_joystick_emergency)
    presenter: FullscreenPresenter | None = None
    log_file = None
    journal_entries: list[str] = []
    missing_capture_indices: list[int] = []
    consecutive_misses = 0
    journal_dir = Path(getattr(args, "journal_dir", DEFAULT_JOURNAL_DIR))
    journal_path = journal_dir / (
        f"capture_journal_step{sample_step}_{datetime.now():%Y%m%d_%H%M%S}_"
        f"{frame_indices[0]:04d}_{frame_indices[-1]:04d}.txt"
    )
    print_tuning = bool(getattr(args, "print_tuning", False))

    try:
        emergency.open()
        controller.initialize(reset_to_perfect=not args.no_reset_perfect)
        left_camera.open()
        right_camera.open()
        if not args.no_imu_zero:
            left_camera.calibrate_zero(
                args.imu_zero_samples,
                verbose=bool(getattr(args, "print_calibration", False)),
            )
            right_camera.calibrate_zero(
                args.imu_zero_samples,
                verbose=bool(getattr(args, "print_calibration", False)),
            )

        log_file, log_writer = make_capture_log_writer(args.log_path)

        left_actual, right_actual = read_both_imu_poses(left_camera, right_camera)
        state = ReplayState(
            left_command=EyePose.from_yaw_pitch(0.0, 0.0),
            right_command=EyePose.from_yaw_pitch(0.0, 0.0),
            left_actual=left_actual,
            right_actual=right_actual,
            left_last_error=np.zeros(3),
            right_last_error=np.zeros(3),
        )

        first_frame_index = frame_indices[0]
        for selected_count, frame_index in enumerate(frame_indices, start=1):
            emergency.check()

            target = targets[frame_index]
            if print_tuning:
                print(
                    f"[{frame_index:04d}] selected {selected_count}/{len(frame_indices)} "
                    f"L=({target.left.yaw:.2f}, {target.left.pitch:.2f}, {target.left.roll:.2f}) "
                    f"R=({target.right.yaw:.2f}, {target.right.pitch:.2f}, {target.right.roll:.2f})"
                )

            if frame_index == first_frame_index:
                coarse_reached = run_initial_big_step_and_coarse(
                    controller,
                    left_camera,
                    right_camera,
                    state,
                    target,
                    emergency=emergency,
                    print_feedback=print_tuning,
                )
                if print_tuning and not coarse_reached:
                    print(
                        "Initial coarse did not reach tolerance; continuing to first timed fine."
                    )
                fine_timeout_s = args.first_fine_timeout_s
            else:
                fine_timeout_s = args.fine_timeout_s

            frame_path = frame_path_for_index(args.frame_dir, frame_index)
            left_path = args.left_save_dir / f"frame_{frame_index:04d}.jpg"
            right_path = args.right_save_dir / f"frame_{frame_index:04d}.jpg"
            if presenter is None:
                presenter = FullscreenPresenter(
                    display_index=int(getattr(args, "display_index", 0)),
                    bring_to_front=bool(getattr(args, "bring_display_to_front", True)),
                )

            frame_settle_s = args.display_settle_s
            if frame_index == first_frame_index:
                frame_settle_s += float(
                    getattr(args, "first_frame_extra_settle_s", 0.0)
                )

            capture_result = run_timed_pd_best_capture_stage(
                controller,
                left_camera,
                right_camera,
                state,
                target,
                presenter,
                frame_path,
                left_path,
                right_path,
                frame_index,
                timeout_s=fine_timeout_s,
                tolerance_deg=fine_tolerance_deg,
                display_settle_s=frame_settle_s,
                capture_retries=args.capture_retries,
                emergency=emergency,
                print_every_steps=args.feedback_print_every_steps if print_tuning else 0,
            )

            fine_reached = capture_result.captured
            fine_elapsed_s = capture_result.elapsed_s
            capture_left_pose = capture_result.left_pose
            capture_right_pose = capture_result.right_pose
            left_error = max_pose_delta(capture_left_pose, target.left)
            right_error = max_pose_delta(capture_right_pose, target.right)
            left_output = str(left_path) if capture_result.captured else ""
            right_output = str(right_path) if capture_result.captured else ""

            log_writer.writerow(
                [
                    frame_index,
                    fine_reached,
                    fine_elapsed_s,
                    left_error,
                    right_error,
                    target.left.yaw,
                    target.left.pitch,
                    target.left.roll,
                    target.right.yaw,
                    target.right.pitch,
                    target.right.roll,
                    capture_left_pose.yaw,
                    capture_left_pose.pitch,
                    capture_left_pose.roll,
                    capture_right_pose.yaw,
                    capture_right_pose.pitch,
                    capture_right_pose.roll,
                    left_output,
                    right_output,
                ]
            )
            log_file.flush()

            if capture_result.captured:
                consecutive_misses = 0
                journal_entry = make_capture_journal_entry(
                    frame_index,
                    target,
                    capture_left_pose,
                    capture_right_pose,
                    fine_reached,
                    fine_elapsed_s,
                    left_error,
                    right_error,
                )
                journal_entry += (
                    f" | capture_count={capture_result.capture_count} "
                    f"total_err={capture_result.total_error_deg:.2f}deg"
                )
            else:
                consecutive_misses += 1
                missing_capture_indices.append(frame_index)
                journal_entry = make_capture_skipped_journal_entry(
                    frame_index,
                    target,
                    capture_left_pose,
                    capture_right_pose,
                    fine_elapsed_s,
                    left_error,
                    right_error,
                )
                journal_entry += (
                    f" | consecutive_misses={consecutive_misses}/"
                    f"{CAPTURE_MAX_CONSECUTIVE_MISSES}"
                )
            print(journal_entry)
            journal_entries.append(journal_entry)

            if consecutive_misses >= CAPTURE_MAX_CONSECUTIVE_MISSES:
                stop_message = (
                    "Stopping capture: "
                    f"{CAPTURE_MAX_CONSECUTIVE_MISSES} consecutive targets were "
                    f"not captured. Recent missing frames: "
                    f"{missing_capture_indices[-CAPTURE_MAX_CONSECUTIVE_MISSES:]}"
                )
                print(stop_message)
                journal_entries.append(stop_message)
                break

    except EmergencyStop as exc:
        print(f"Emergency stop requested: {exc}")
    except WarningAbort as exc:
        print(str(exc))
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")
        raise

    finally:
        try:
            print("Returning to initial position...")
            controller.reset_initial()
        except Exception as exc:
            print(f"Error: reset_initial failed: {exc}")
        missing_summary = f"Frames not captured: {missing_capture_indices}"
        print(missing_summary)
        if journal_entries:
            journal_entries.append(missing_summary)
        if journal_entries:
            try:
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                journal_path.write_text(
                    "\n".join(journal_entries) + "\n",
                    encoding="utf-8",
                )
                print(f"Journal saved: {journal_path}")
            except Exception as exc:
                print(f"Error: failed to save journal: {exc}")
        if log_file is not None:
            log_file.close()
        if presenter is not None:
            presenter.close()
        left_camera.close()
        right_camera.close()
        try:
            controller.enable_torque(False)
        except Exception as exc:
            print(f"Error: disabling torque failed: {exc}")
        emergency.close()
        controller.close()


def run_replay(args: argparse.Namespace) -> None:
    targets = load_targets(args.mat_path, args.mat_key)
    end_index = args.end_index if args.end_index is not None else len(targets)
    end_index = min(end_index, len(targets))
    display_capture_enabled = ENABLE_FRAME_DISPLAY_AND_CAPTURE or args.enable_display_capture

    if args.start_index < 0 or args.start_index >= end_index:
        raise ValueError("Invalid start/end index range")

    if display_capture_enabled:
        validate_frame_files(args.frame_dir, args.start_index, end_index)
        args.left_save_dir.mkdir(parents=True, exist_ok=True)
        args.right_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(targets)} target samples from {args.mat_path}")
    print(f"Replaying samples [{args.start_index}, {end_index})")
    print(f"Target gaze scale = {TARGET_GAZE_SCALE:.2f}")
    print(
        "IK feedback enabled: "
        f"coarse(tol={COARSE_ANGLE_TOLERANCE_DEG:.2f} deg, "
        f"gain={COARSE_IK_FEEDBACK_GAIN:.2f}, "
        f"max_step={COARSE_IK_FEEDBACK_MAX_STEP_TICKS} ticks, "
        f"fine_entry<={COARSE_ACCEPT_MAX_ERROR_DEG:.2f} deg, "
        f"max_iter={COARSE_MAX_ITERATIONS}); "
        f"fine_ik_pd(tol={FINE_ANGLE_TOLERANCE_DEG:.2f} deg, "
        f"kp={FINE_IK_PD_KP:.2f}, "
        f"kd={FINE_IK_PD_KD:.2f}, "
        f"max_step={FINE_IK_PD_MAX_STEP_TICKS} ticks, "
        f"min_step={FINE_IK_PD_MIN_STEP_TICKS} ticks, "
        f"min_error={FINE_IK_PD_MIN_STEP_ERROR_TICKS:.1f} ticks, "
        f"loop_dt={FINE_PD_LOOP_DELAY_S:.2f}s, "
        f"max_iter={FINE_MAX_ITERATIONS})"
    )
    if not display_capture_enabled:
        print("Frame display and camera capture are disabled for this run.")

    if args.dry_run:
        max_abs_pulse = 0.0
        for target in targets[args.start_index:end_index]:
            max_abs_pulse = max(max_abs_pulse, max(abs(v) for v in pose_to_pulses(target)))
        print(f"Dry run OK. Max abs IK pulse: {max_abs_pulse:.1f}")
        return

    controller = DynamixelTwoEyesController(args.dxl_port, args.dxl_baud)
    left_camera = SerialCamera(args.left_camera_port, args.camera_baud)
    right_camera = SerialCamera(args.right_camera_port, args.camera_baud)
    emergency = JoystickEmergency(enabled=not args.no_joystick_emergency)
    presenter = None
    log_file = None

    try:
        emergency.open()
        controller.initialize(reset_to_perfect=not args.no_reset_perfect)
        left_camera.open()
        right_camera.open()
        if not args.no_imu_zero:
            left_camera.calibrate_zero(args.imu_zero_samples)
            right_camera.calibrate_zero(args.imu_zero_samples)
        log_file, log_writer = make_log_writer(args.log_path)

        left_actual, right_actual = read_both_imu_poses(left_camera, right_camera)

        state = ReplayState(
            left_command=EyePose.from_yaw_pitch(0.0, 0.0),
            right_command=EyePose.from_yaw_pitch(0.0, 0.0),
            left_actual=left_actual,
            right_actual=right_actual,
            left_last_error=np.zeros(3),
            right_last_error=np.zeros(3),
        )

        for frame_index in range(args.start_index, end_index):
            emergency.check()

            target = targets[frame_index]
            print(
                f"[{frame_index:04d}] "
                f"L=({target.left.yaw:.2f}, {target.left.pitch:.2f}, {target.left.roll:.2f}) "
                f"R=({target.right.yaw:.2f}, {target.right.pitch:.2f}, {target.right.roll:.2f})"
            )

            reached = drive_to_target(
                controller, left_camera, right_camera, state, target, emergency=emergency
            )
            if not reached:
                raise TimeoutError(
                    f"IMU pose did not reach target for frame {frame_index}: "
                    f"L actual=({state.left_actual.yaw:.2f}, {state.left_actual.pitch:.2f}, "
                    f"{state.left_actual.roll:.2f}), "
                    f"R actual=({state.right_actual.yaw:.2f}, {state.right_actual.pitch:.2f}, "
                    f"{state.right_actual.roll:.2f})"
                )

            left_output = ""
            right_output = ""

            if display_capture_enabled:
                if presenter is None:
                    presenter = FullscreenPresenter()

                frame_path = frame_path_for_index(args.frame_dir, frame_index)
                if not presenter.show(frame_path):
                    print("ESC/quit received; stopping replay.")
                    break
                time.sleep(DISPLAY_SETTLE_DELAY_S)

                left_path = args.left_save_dir / f"frame_{frame_index:04d}.jpg"
                right_path = args.right_save_dir / f"frame_{frame_index:04d}.jpg"
                capture_both_cameras(left_camera, right_camera, left_path, right_path)
                left_output = str(left_path)
                right_output = str(right_path)

            log_writer.writerow(
                [
                    frame_index,
                    target.left.yaw,
                    target.left.pitch,
                    target.left.roll,
                    target.right.yaw,
                    target.right.pitch,
                    target.right.roll,
                    state.left_actual.yaw,
                    state.left_actual.pitch,
                    state.left_actual.roll,
                    state.right_actual.yaw,
                    state.right_actual.pitch,
                    state.right_actual.roll,
                    left_output,
                    right_output,
                ]
            )
            log_file.flush()

    except EmergencyStop as exc:
        print(f"Emergency stop requested: {exc}")
    except WarningAbort as exc:
        print(str(exc))

    finally:
        if log_file is not None:
            log_file.close()
        if presenter is not None:
            presenter.close()
        left_camera.close()
        right_camera.close()
        try:
            controller.reset_initial()
        except Exception as exc:
            print(f"Error: reset_initial failed: {exc}")
        try:
            controller.enable_torque(False)
        except Exception as exc:
            print(f"Error: disabling torque failed: {exc}")
        emergency.close()
        controller.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay healthbaseline gaze trajectory with two robotic eyes."
    )
    parser.add_argument("--mat-path", type=Path, default=DEFAULT_MAT_PATH)
    parser.add_argument("--mat-key", default="SGazeAngleSmoothed_cell")
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--left-save-dir", type=Path, default=DEFAULT_LEFT_SAVE_DIR)
    parser.add_argument("--right-save-dir", type=Path, default=DEFAULT_RIGHT_SAVE_DIR)
    parser.add_argument("--left-camera-port", default=DEFAULT_LEFT_CAMERA_PORT)
    parser.add_argument("--right-camera-port", default=DEFAULT_RIGHT_CAMERA_PORT)
    parser.add_argument("--camera-baud", type=int, default=DEFAULT_CAMERA_BAUD)
    parser.add_argument("--dxl-port", default=DEFAULT_DXL_PORT)
    parser.add_argument("--dxl-baud", type=int, default=DEFAULT_DXL_BAUD)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--ik-step-test",
        action="store_true",
        help="Only run one IK move, hold, then return to initial position.",
    )
    parser.add_argument("--ik-test-index", type=int, default=0)
    parser.add_argument(
        "--ik-test-side",
        choices=("both", "left", "right"),
        default="both",
        help="For --ik-step-test only, command both eyes, only ID1-6, or only ID7-12.",
    )
    parser.add_argument("--ik-hold-s", type=float, default=IK_STEP_TEST_HOLD_S)
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
    parser.add_argument("--imu-zero-samples", type=int, default=IMU_ZERO_SAMPLES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and IK pulses without opening motors, cameras, or fullscreen display.",
    )
    parser.add_argument(
        "--enable-display-capture",
        action="store_true",
        help="After IMU target is reached, show the matching frame and capture both cameras.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("D:/visualization2026/replay_log.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ik_step_test:
        run_ik_step_test(args)
    else:
        run_replay(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
