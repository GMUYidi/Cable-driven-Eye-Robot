from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from ..config import IMUConfig
from ..kinematics import EyePose
from ..math_utils import angle_error_deg, normalize_angle_deg


@dataclass(frozen=True)
class IMUSample:
    pose: EyePose
    angular_velocity: EyePose = EyePose(math.nan, math.nan, math.nan)
    angular_acceleration: EyePose = EyePose(math.nan, math.nan, math.nan)
    linacc_xyz: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    gyro_xyz: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    timestamp_s: float = 0.0


def circular_mean_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    sin_sum = sum(math.sin(math.radians(value)) for value in values)
    cos_sum = sum(math.cos(math.radians(value)) for value in values)
    return normalize_angle_deg(math.degrees(math.atan2(sin_sum, cos_sum)))


def parse_imu_line(line: str, roll_zero_offset_deg: float = 0.0) -> IMUSample | None:
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
        "error",
        "setup failed",
        "commands",
        "pong",
        "stream:",
        "camera init",
        "wifi",
        "softap",
        "http",
    )
    if lowered.startswith(ignored_prefixes):
        return None

    parts = [part.strip() for part in line.split(",")]
    try:
        if len(parts) == 3:
            yaw = float(parts[0])
            pitch = float(parts[1])
            roll = float(parts[2])
            angular_velocity = EyePose(math.nan, math.nan, math.nan)
            angular_acceleration = EyePose(math.nan, math.nan, math.nan)
            linacc = (math.nan, math.nan, math.nan)
            gyro = (math.nan, math.nan, math.nan)
        elif len(parts) >= 4 and parts[0].upper() == "IMU":
            yaw = float(parts[1])
            roll = float(parts[2])
            pitch = float(parts[3])
            angular_velocity = EyePose(
                float(parts[4]) if len(parts) >= 7 else math.nan,
                float(parts[5]) if len(parts) >= 7 else math.nan,
                float(parts[6]) if len(parts) >= 7 else math.nan,
            )
            angular_acceleration = EyePose(
                float(parts[7]) if len(parts) >= 10 else math.nan,
                float(parts[8]) if len(parts) >= 10 else math.nan,
                float(parts[9]) if len(parts) >= 10 else math.nan,
            )
            linacc = (
                float(parts[10]) if len(parts) >= 13 else math.nan,
                float(parts[11]) if len(parts) >= 13 else math.nan,
                float(parts[12]) if len(parts) >= 13 else math.nan,
            )
            gyro = (
                float(parts[13]) if len(parts) >= 16 else math.nan,
                float(parts[14]) if len(parts) >= 16 else math.nan,
                float(parts[15]) if len(parts) >= 16 else math.nan,
            )
        elif len(parts) >= 7:
            yaw = float(parts[0])
            pitch = float(parts[1])
            roll = float(parts[2])
            angular_velocity = EyePose(math.nan, math.nan, math.nan)
            angular_acceleration = EyePose(math.nan, math.nan, math.nan)
            linacc = (math.nan, math.nan, math.nan)
            gyro = (math.nan, math.nan, math.nan)
        else:
            return None
    except ValueError:
        return None

    pose = EyePose(
        normalize_angle_deg(yaw),
        float(pitch),
        normalize_angle_deg(roll - roll_zero_offset_deg),
    )
    return IMUSample(
        pose=pose,
        angular_velocity=angular_velocity,
        angular_acceleration=angular_acceleration,
        linacc_xyz=linacc,
        gyro_xyz=gyro,
        timestamp_s=time.monotonic(),
    )


class IMUReader(threading.Thread):
    def __init__(
        self,
        name: str,
        port: str,
        config: IMUConfig,
        callback: Callable[[str, IMUSample], None] | None = None,
        zero_on_open: bool = True,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.port = port
        self.config = config
        self.callback = callback
        self.zero_on_open = zero_on_open
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ser = None
        self.zero_pose: EyePose | None = None
        self.last_sample: IMUSample | None = None
        self.last_error = ""

    def open(self) -> None:
        import serial

        self.ser = serial.Serial(
            self.port,
            self.config.baudrate,
            timeout=self.config.serial_timeout_s,
            write_timeout=1,
        )
        self.ser.dtr = False
        self.ser.rts = False
        time.sleep(self.config.startup_delay_s)
        self.ser.reset_input_buffer()
        if self.zero_on_open:
            self.calibrate_zero(self.config.zero_samples)

    def close(self) -> None:
        self.stop_event.set()
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        if self.is_alive():
            self.join(timeout=1.0)

    def read_line(self) -> str:
        if self.ser is None:
            return ""
        return self.ser.readline().decode("utf-8", errors="replace").strip()

    def read_raw_sample(self, timeout_s: float = 2.0) -> IMUSample:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            sample = parse_imu_line(
                self.read_line(),
                self.config.roll_zero_offsets_deg.get(self.port.upper(), 0.0),
            )
            if sample is not None:
                return sample
        raise TimeoutError(f"{self.name} IMU timed out on {self.port}")

    def calibrate_zero(self, samples: int) -> EyePose:
        raw_samples = [self.read_raw_sample(timeout_s=2.0) for _ in range(samples)]
        self.zero_pose = EyePose(
            circular_mean_deg([sample.pose.yaw_deg for sample in raw_samples]),
            mean(sample.pose.pitch_deg for sample in raw_samples),
            circular_mean_deg([sample.pose.roll_deg for sample in raw_samples]),
        )
        zeroed = IMUSample(EyePose(0.0, 0.0, 0.0), timestamp_s=time.monotonic())
        with self.lock:
            self.last_sample = zeroed
        return self.zero_pose

    def to_relative_sample(self, sample: IMUSample) -> IMUSample:
        if self.zero_pose is None:
            relative_pose = sample.pose
        else:
            relative_pose = EyePose(
                angle_error_deg(self.zero_pose.yaw_deg, sample.pose.yaw_deg),
                sample.pose.pitch_deg - self.zero_pose.pitch_deg,
                angle_error_deg(self.zero_pose.roll_deg, sample.pose.roll_deg),
            )
        return IMUSample(
            pose=relative_pose,
            angular_velocity=sample.angular_velocity,
            angular_acceleration=sample.angular_acceleration,
            linacc_xyz=sample.linacc_xyz,
            gyro_xyz=sample.gyro_xyz,
            timestamp_s=sample.timestamp_s,
        )

    def latest(self, max_age_s: float | None = None) -> tuple[IMUSample | None, float, bool]:
        if max_age_s is None:
            max_age_s = self.config.max_age_s
        with self.lock:
            sample = self.last_sample
        if sample is None:
            return None, math.nan, False
        age = time.monotonic() - sample.timestamp_s
        return sample, age, age <= max_age_s

    def run(self) -> None:
        try:
            if self.ser is None:
                self.open()
            while not self.stop_event.is_set():
                raw = parse_imu_line(
                    self.read_line(),
                    self.config.roll_zero_offsets_deg.get(self.port.upper(), 0.0),
                )
                if raw is None:
                    continue
                sample = self.to_relative_sample(raw)
                with self.lock:
                    self.last_sample = sample
                if self.callback is not None:
                    self.callback(self.name, sample)
        except Exception as exc:
            self.last_error = str(exc)
