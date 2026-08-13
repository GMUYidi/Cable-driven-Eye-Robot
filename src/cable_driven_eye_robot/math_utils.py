from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_deadband(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def limit_step(current_value: float, target_value: float, max_step: float) -> float:
    delta = target_value - current_value
    if delta > max_step:
        return current_value + max_step
    if delta < -max_step:
        return current_value - max_step
    return target_value


def normalize_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def angle_error_deg(actual: float, target: float) -> float:
    return normalize_angle_deg(target - actual)


def compute_listing_roll(yaw_deg: float, pitch_deg: float) -> float:
    numerator = math.sin(math.radians(yaw_deg)) * math.sin(math.radians(pitch_deg))
    denominator = 1.0 + math.cos(math.radians(yaw_deg)) * math.cos(math.radians(pitch_deg))
    if abs(denominator) < 1e-9:
        return 0.0

    ratio = clamp(numerator / denominator, -1.0, 1.0)
    roll = math.degrees(math.asin(ratio))
    return 0.0 if math.isnan(roll) else roll


def fix_signed_32bit(value: int) -> int:
    value = int(value)
    if value > 2**31:
        value -= 2**32
    return value


def fix_signed_16bit(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


def unwrap_to_reference(desired_abs: int, ref_abs: int, ticks_per_rev: int) -> int:
    k = round((ref_abs - desired_abs) / ticks_per_rev)
    return int(desired_abs + k * ticks_per_rev)
