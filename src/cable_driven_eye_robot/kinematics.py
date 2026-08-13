from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import LEFT, RIGHT, MOTOR_CHANNELS, MotorChannel
from .math_utils import clamp, compute_listing_roll


MUSCLES = ("LR", "MR", "SR", "IR", "SO", "IO")


@dataclass(frozen=True)
class EyePose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0

    @classmethod
    def from_yaw_pitch(cls, yaw_deg: float, pitch_deg: float) -> "EyePose":
        return cls(yaw_deg, pitch_deg, compute_listing_roll(yaw_deg, pitch_deg))


@dataclass(frozen=True)
class EyeGeometry:
    radius_mm: float
    rate: float
    insertion_points: dict[str, tuple[float, float, float]]
    pulley_points: dict[str, tuple[float, float, float]]
    pulse_signs: dict[str, float]
    pulse_gains: dict[str, float]
    pulse_offsets: dict[str, float]


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ],
        dtype=float,
    )
    return rz @ (ry @ rx)


def _sphere_tangent_arc_cable_length(
    insertion: np.ndarray,
    pulley: np.ndarray,
    radius_mm: float,
) -> float:
    insertion_norm = float(np.linalg.norm(insertion))
    pulley_norm = float(np.linalg.norm(pulley))

    if insertion_norm < 1e-9:
        raise ValueError("Insertion point cannot be at the eyeball center.")
    if pulley_norm <= radius_mm:
        raise ValueError(
            f"Pulley must be outside the eyeball sphere: |pulley|={pulley_norm:.3f}, "
            f"radius={radius_mm:.3f}."
        )

    insertion_hat = insertion / insertion_norm
    pulley_hat = pulley / pulley_norm

    tangent_cos = clamp(radius_mm / pulley_norm, -1.0, 1.0)
    tangent_sin = math.sqrt(max(0.0, 1.0 - tangent_cos * tangent_cos))
    insertion_along_pulley = clamp(float(np.dot(insertion_hat, pulley_hat)), -1.0, 1.0)
    insertion_perp = math.sqrt(max(0.0, 1.0 - insertion_along_pulley**2))

    best_tangent_dot = clamp(
        tangent_cos * insertion_along_pulley + tangent_sin * insertion_perp,
        -1.0,
        1.0,
    )
    arc_length = radius_mm * math.acos(best_tangent_dot)
    tangent_length = math.sqrt(max(0.0, pulley_norm * pulley_norm - radius_mm * radius_mm))
    return tangent_length + arc_length


def _cable_length_difference(
    base_insertion: np.ndarray,
    rotated_insertion: np.ndarray,
    pulley: np.ndarray,
    radius_mm: float,
    use_sphere_tangent: bool,
) -> float:
    if not use_sphere_tangent:
        return float(
            np.linalg.norm(base_insertion - pulley)
            - np.linalg.norm(rotated_insertion - pulley)
        )

    return _sphere_tangent_arc_cable_length(
        base_insertion,
        pulley,
        radius_mm,
    ) - _sphere_tangent_arc_cable_length(
        rotated_insertion,
        pulley,
        radius_mm,
    )


def build_left_eye_geometry() -> EyeGeometry:
    radius = 15.0
    return EyeGeometry(
        radius_mm=radius,
        rate=10.0,
        insertion_points={
            "SR": (-6, 0, 12.5),
            "IR": (-6, 0, -12.5),
            "SO": (-6, 0, 12.5),
            "IO": (-6, 0, -12.5),
            "MR": (0, radius, 0),
            "LR": (0, -radius, 0),
        },
        pulley_points={
            "SR": (-21, 0, 5),
            "IR": (-21, 0, -5),
            "SO": (12, 24, 7),
            "IO": (12, 24, -7),
            "MR": (-21, 4.5, 0),
            "LR": (-21, -4.5, 0),
        },
        pulse_signs={"LR": 1.0, "MR": 1.0, "SR": 1.0, "IR": 1.0, "SO": 1.0, "IO": -1.0},
        pulse_gains={"LR": 2.2, "MR": 2.2, "SR": 2.2, "IR": 2.2, "SO": 2.0, "IO": 2.0},
        pulse_offsets={"SO": -50.0, "IO": -50.0},
    )


def build_right_eye_geometry() -> EyeGeometry:
    radius = 15.0
    return EyeGeometry(
        radius_mm=radius,
        rate=10.0,
        insertion_points={
            "SR": (-6, 0, 12.5),
            "IR": (-6, 0, -12.5),
            "SO": (-6, 0, 12.5),
            "IO": (-6, 0, -12.5),
            "MR": (0, radius, 0),
            "LR": (0, -radius, 0),
        },
        pulley_points={
            "SR": (-21, 0, 5),
            "IR": (-21, 0, -5),
            "SO": (12, -24, 7),
            "IO": (12, -24, -7),
            "MR": (-21, -4.5, 0),
            "LR": (-21, 4.5, 0),
        },
        pulse_signs={"LR": -1.0, "MR": -1.0, "SR": -1.0, "IR": -1.0, "SO": -1.0, "IO": 1.0},
        pulse_gains={"LR": 2.2, "MR": 2.2, "SR": 2.2, "IR": 2.2, "SO": 2.0, "IO": 2.0},
        pulse_offsets={"SO": -50.0, "IO": -50.0},
    )


def compute_muscle_pulses(
    pose: EyePose,
    geometry: EyeGeometry,
    use_sphere_tangent: bool = True,
) -> dict[str, float]:
    rotation = _rotation_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    pulses: dict[str, float] = {}

    for muscle in MUSCLES:
        insertion = np.array(geometry.insertion_points[muscle], dtype=float)
        pulley = np.array(geometry.pulley_points[muscle], dtype=float)
        rotated = rotation @ insertion
        length_delta = _cable_length_difference(
            insertion,
            rotated,
            pulley,
            geometry.radius_mm,
            use_sphere_tangent,
        )
        gain = geometry.pulse_gains[muscle]
        sign = geometry.pulse_signs[muscle]
        offset = geometry.pulse_offsets.get(muscle, 0.0)
        pulses[muscle] = sign * length_delta / geometry.rate * 4096.0 * gain + offset

    return pulses


def motor_pulses_from_pose(
    left_pose: EyePose,
    right_pose: EyePose | None = None,
    channels: tuple[MotorChannel, ...] = MOTOR_CHANNELS,
    use_sphere_tangent: bool = True,
) -> list[float]:
    if right_pose is None:
        right_pose = left_pose

    left = compute_muscle_pulses(left_pose, build_left_eye_geometry(), use_sphere_tangent)
    right = compute_muscle_pulses(right_pose, build_right_eye_geometry(), use_sphere_tangent)

    values: list[float] = []
    for channel in channels:
        source = left if channel.eye == LEFT else right
        values.append(float(source[channel.muscle]))
    return values


def motor_pulses_from_gaze(
    yaw_deg: float,
    pitch_deg: float,
    use_sphere_tangent: bool = True,
) -> tuple[list[float], EyePose]:
    pose = EyePose.from_yaw_pitch(yaw_deg, pitch_deg)
    return motor_pulses_from_pose(pose, use_sphere_tangent=use_sphere_tangent), pose
