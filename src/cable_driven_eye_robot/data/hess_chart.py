from __future__ import annotations

from dataclasses import dataclass

from ..kinematics import EyePose


@dataclass(frozen=True)
class HessChartTarget:
    index: int
    label: str
    left_pose: EyePose
    right_pose: EyePose


def _paired_pose(yaw_deg: float, pitch_deg: float) -> tuple[EyePose, EyePose]:
    pose = EyePose.from_yaw_pitch(yaw_deg, pitch_deg)
    return pose, pose


def generate_hess9_targets(amplitude_deg: float = 8.0) -> list[HessChartTarget]:
    """Return the center plus eight cardinal/diagonal gaze targets."""

    points = [
        ("center", 0.0, 0.0),
        ("right", amplitude_deg, 0.0),
        ("left", -amplitude_deg, 0.0),
        ("up", 0.0, amplitude_deg),
        ("down", 0.0, -amplitude_deg),
        ("up_right", amplitude_deg, amplitude_deg),
        ("up_left", -amplitude_deg, amplitude_deg),
        ("down_right", amplitude_deg, -amplitude_deg),
        ("down_left", -amplitude_deg, -amplitude_deg),
    ]
    return _targets_from_points(points)


def generate_grid_targets(
    yaw_points_deg: list[float],
    pitch_points_deg: list[float],
    center_first: bool = True,
) -> list[HessChartTarget]:
    points: list[tuple[str, float, float]] = []
    for pitch in pitch_points_deg:
        for yaw in yaw_points_deg:
            points.append((_label_from_pose(yaw, pitch), yaw, pitch))

    if center_first:
        center = [point for point in points if point[1] == 0.0 and point[2] == 0.0]
        non_center = [point for point in points if point not in center]
        points = center + non_center

    return _targets_from_points(points)


def parse_degree_list(value: str) -> list[float]:
    points: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        points.append(float(stripped))
    if not points:
        raise ValueError("Degree list cannot be empty.")
    return points


def _targets_from_points(points: list[tuple[str, float, float]]) -> list[HessChartTarget]:
    targets: list[HessChartTarget] = []
    for index, (label, yaw, pitch) in enumerate(points):
        left_pose, right_pose = _paired_pose(yaw, pitch)
        targets.append(
            HessChartTarget(
                index=index,
                label=label,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        )
    return targets


def _label_from_pose(yaw_deg: float, pitch_deg: float) -> str:
    yaw_token = f"yaw_{yaw_deg:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
    pitch_token = f"pitch_{pitch_deg:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"{yaw_token}_{pitch_token}"
