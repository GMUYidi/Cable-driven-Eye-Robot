from __future__ import annotations

from dataclasses import dataclass, field


LEFT = "left"
RIGHT = "right"


@dataclass(frozen=True)
class MotorChannel:
    motor_id: int
    eye: str
    muscle: str


# Physical mapping reported for the current robot.
MOTOR_CHANNELS: tuple[MotorChannel, ...] = (
    MotorChannel(1, LEFT, "LR"),
    MotorChannel(2, LEFT, "MR"),
    MotorChannel(3, LEFT, "SR"),
    MotorChannel(4, LEFT, "IR"),
    MotorChannel(5, LEFT, "SO"),
    MotorChannel(6, LEFT, "IO"),
    MotorChannel(7, RIGHT, "MR"),
    MotorChannel(8, RIGHT, "LR"),
    MotorChannel(9, RIGHT, "SR"),
    MotorChannel(10, RIGHT, "IR"),
    MotorChannel(11, RIGHT, "SO"),
    MotorChannel(12, RIGHT, "IO"),
)


@dataclass(frozen=True)
class ControlTable:
    goal_current: int = 102
    goal_velocity: int = 104
    profile_accel: int = 108
    profile_velocity: int = 112
    torque_enable: int = 64
    operating_mode: int = 11
    present_current: int = 126
    present_position: int = 132
    goal_position: int = 116


@dataclass(frozen=True)
class DynamixelConfig:
    device_name: str = "COM12"
    baudrate: int = 1_000_000
    protocol_version: float = 2.0
    motor_ids: tuple[int, ...] = tuple(channel.motor_id for channel in MOTOR_CHANNELS)
    ticks_per_rev: int = 4096
    torque_enable: int = 1
    torque_disable: int = 0
    extended_position_mode: int = 4
    max_goal_step_ticks: int = 12_000
    do_unwrap: bool = True
    table: ControlTable = field(default_factory=ControlTable)


@dataclass(frozen=True)
class ProfileConfig:
    accel_left: int = 30_000
    velocity_left: int = 20_000
    accel_right: int = 30_000
    velocity_right: int = 20_000


@dataclass(frozen=True)
class JoystickConfig:
    joystick_index: int = 0
    yaw_axis: int = 2
    pitch_axis: int = 3
    triangle_button: int = 3
    deadband: float = 0.08
    max_yaw_deg: float = 10.5
    max_pitch_deg: float = 8.4
    max_yaw_step_deg: float = 0.08
    max_pitch_step_deg: float = 0.06
    yaw_sign: float = 1.0
    pitch_sign: float = 1.0


@dataclass(frozen=True)
class IMUConfig:
    left_port: str = "COM11"
    right_port: str = "COM6"
    baudrate: int = 921_600
    serial_timeout_s: float = 0.5
    startup_delay_s: float = 1.5
    max_age_s: float = 0.5
    zero_samples: int = 8
    roll_zero_offsets_deg: dict[str, float] = field(
        default_factory=lambda: {
            "COM11": 0.1,
            "COM6": -8.25,
        }
    )


@dataclass(frozen=True)
class CameraConfig:
    left_url: str = "http://192.168.4.1/stream"
    right_url: str = "http://192.168.4.200/stream"


@dataclass(frozen=True)
class SafetyConfig:
    max_abs_pulse: float = 9192.0
    control_period_s: float = 0.002
    reset_settle_wait_s: float = 1.0


@dataclass(frozen=True)
class FeedbackConfig:
    loop_period_s: float = 0.05
    tolerance_deg: float = 0.5
    target_update_threshold_deg: float = 1.0
    kp_yaw_ticks_per_deg: float = 24.0
    kp_pitch_ticks_per_deg: float = 24.0
    kp_roll_ticks_per_deg: float = 20.0
    kd_yaw_ticks_per_dps: float = 5.0
    kd_pitch_ticks_per_dps: float = 5.0
    kd_roll_ticks_per_dps: float = 3.0
    min_step_ticks: float = 5.0
    max_step_ticks: float = 80.0
    min_error_deg: float = 0.5


@dataclass(frozen=True)
class RobotConfig:
    dynamixel: DynamixelConfig = field(default_factory=DynamixelConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    joystick: JoystickConfig = field(default_factory=JoystickConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


DEFAULT_CONFIG = RobotConfig()
