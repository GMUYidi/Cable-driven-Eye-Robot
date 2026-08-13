from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..kinematics import EyePose


@dataclass(frozen=True)
class EyeTrackerTarget:
    index: int
    left_pose: EyePose
    right_pose: EyePose


def _as_numeric_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype == object:
        matrix = np.asarray(matrix.tolist(), dtype=float)
    matrix = np.squeeze(matrix).astype(float)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError(
            "Eye-tracker target matrix must have at least four columns: "
            "right yaw, right pitch, left yaw, left pitch."
        )
    return matrix


def load_eye_tracker_targets(
    mat_path: Path,
    key: str = "GazeAngleSmoothed_cell",
    start_index: int = 0,
    max_targets: int | None = None,
    sign: float = -1.0,
) -> list[EyeTrackerTarget]:
    import scipy.io as scio

    data = scio.loadmat(mat_path)
    if key not in data:
        available = ", ".join(sorted(name for name in data if not name.startswith("__")))
        raise KeyError(f"MAT key '{key}' not found. Available keys: {available}")

    matrix = _as_numeric_matrix(data[key])
    start = max(0, int(start_index))
    stop = len(matrix) if max_targets is None else min(len(matrix), start + int(max_targets))

    targets: list[EyeTrackerTarget] = []
    for index in range(start, stop):
        right_yaw = sign * float(matrix[index, 0])
        right_pitch = sign * float(matrix[index, 1])
        left_yaw = sign * float(matrix[index, 2])
        left_pitch = sign * float(matrix[index, 3])
        targets.append(
            EyeTrackerTarget(
                index=index,
                left_pose=EyePose.from_yaw_pitch(left_yaw, left_pitch),
                right_pose=EyePose.from_yaw_pitch(right_yaw, right_pitch),
            )
        )
    return targets
