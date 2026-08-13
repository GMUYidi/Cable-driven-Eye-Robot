from __future__ import annotations

import argparse
import time
from dataclasses import replace

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

from ..config import DynamixelConfig, ProfileConfig, RobotConfig
from ..hardware.dynamixel import DynamixelController
from ..input_devices.joystick import JoystickReader
from ..math_utils import apply_deadband, clamp


CROSS_BTN = 0
CIRCLE_BTN = 1
SQUARE_BTN = 2
TRIANGLE_BTN = 3
L1_BTN = 9
R1_BTN = 10
LEFT_STICK_CLICK_BTN = 7
DPAD_UP_BTN = 11
DPAD_DOWN_BTN = 12
DPAD_LEFT_BTN = 13
DPAD_RIGHT_BTN = 14
L2_AXIS = 4
R2_AXIS = 5
RIGHT_Y_AXIS = 3
DPAD_HAT_INDEX = 0
TRIGGER_PRESS_THRESHOLD = 0.5


BUTTON_MOTOR_MAP = {
    CIRCLE_BTN: 1,
    SQUARE_BTN: 2,
    TRIANGLE_BTN: 3,
    CROSS_BTN: 4,
    R1_BTN: 5,
    L1_BTN: 11,
    DPAD_RIGHT_BTN: 7,
    DPAD_LEFT_BTN: 8,
    DPAD_UP_BTN: 9,
    DPAD_DOWN_BTN: 10,
}

TRIGGER_MOTOR_MAP = {
    R2_AXIS: 6,
    L2_AXIS: 12,
}

HAT_MOTOR_MAP = {
    (1, 0): 7,
    (-1, 0): 8,
    (0, 1): 9,
    (0, -1): 10,
}

MOTOR_DIRECTIONS = {
    1: 1,
    2: 1,
    3: -1,
    4: -1,
    5: -1,
    6: 1,
    7: -1,
    8: -1,
    9: 1,
    10: 1,
    11: 1,
    12: -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual neutral-position tuner for all 12 Dynamixel motors.")
    parser.add_argument("--dxl-port", default="COM12")
    parser.add_argument("--dxl-baud", type=int, default=1_000_000)
    parser.add_argument("--max-jog-ticks-per-sec", type=float, default=700.0)
    parser.add_argument("--write-period-s", type=float, default=0.02)
    parser.add_argument("--deadband", type=float, default=0.08)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RobotConfig:
    base = RobotConfig()
    return replace(
        base,
        dynamixel=replace(base.dynamixel, device_name=args.dxl_port, baudrate=args.dxl_baud),
    )


def print_positions(label: str, positions: list[float]) -> None:
    print(f"{label}: [{', '.join(str(int(value)) for value in positions)}]")


def keyboard_stop_requested() -> bool:
    if msvcrt is None:
        return False
    if msvcrt.kbhit():
        msvcrt.getch()
        return True
    return False


def run() -> int:
    args = parse_args()
    config = build_config(args)
    joystick = JoystickReader(config.joystick)
    controller = DynamixelController(config.dynamixel, config.profile)

    try:
        joystick.open()
        print(f"Detected joystick: {joystick.name}")
        controller.open()
        goal_positions = [float(value) for value in controller.initial_positions]
        last_written_positions = [int(round(value)) for value in goal_positions]
        print_positions("Initial present positions", goal_positions)

        print("Select motor, then push right stick up/down to jog.")
        print("Circle=1, Square=2, Triangle=3, Cross=4, R1=5, R2=6")
        print("D-pad Right=7, Left=8, Up=9, Down=10, L1=11, L2=12")
        print("Click left stick (L3) to finish and print positions.")

        selected_motor = 1
        previous_buttons = {button: joystick.safe_get_button(button) for button in BUTTON_MOTOR_MAP}
        previous_triggers = {
            axis: joystick.safe_get_axis(axis) > TRIGGER_PRESS_THRESHOLD
            for axis in TRIGGER_MOTOR_MAP
        }
        previous_hat = joystick.safe_get_hat(DPAD_HAT_INDEX)
        previous_finish = joystick.safe_get_button(LEFT_STICK_CLICK_BTN)
        last_loop = time.monotonic()
        last_write = 0.0
        last_status = 0.0

        while True:
            joystick.pump()
            now = time.monotonic()
            dt = clamp(now - last_loop, 0.0, 0.05)
            last_loop = now

            finish_now = joystick.safe_get_button(LEFT_STICK_CLICK_BTN)
            if finish_now and not previous_finish:
                print("Calibration finished by L3.")
                break
            previous_finish = finish_now

            if keyboard_stop_requested():
                print("Keyboard stop requested.")
                break

            for button, motor_id in BUTTON_MOTOR_MAP.items():
                pressed = joystick.safe_get_button(button)
                if pressed and not previous_buttons.get(button, False):
                    selected_motor = motor_id
                    goal_positions[motor_id - 1] = float(controller.read_single_present_position(motor_id))
                    print(f"Selected motor: ID {selected_motor}")
                previous_buttons[button] = pressed

            for axis, motor_id in TRIGGER_MOTOR_MAP.items():
                pressed = joystick.safe_get_axis(axis) > TRIGGER_PRESS_THRESHOLD
                if pressed and not previous_triggers.get(axis, False):
                    selected_motor = motor_id
                    goal_positions[motor_id - 1] = float(controller.read_single_present_position(motor_id))
                    print(f"Selected motor: ID {selected_motor}")
                previous_triggers[axis] = pressed

            hat = joystick.safe_get_hat(DPAD_HAT_INDEX)
            if hat != previous_hat:
                if hat in HAT_MOTOR_MAP:
                    selected_motor = HAT_MOTOR_MAP[hat]
                    goal_positions[selected_motor - 1] = float(controller.read_single_present_position(selected_motor))
                    print(f"Selected motor: ID {selected_motor}")
                previous_hat = hat

            stick_up = -joystick.safe_get_axis(RIGHT_Y_AXIS)
            stick_up = apply_deadband(stick_up, args.deadband)
            if stick_up != 0.0:
                index = selected_motor - 1
                direction = MOTOR_DIRECTIONS[selected_motor]
                goal_positions[index] += stick_up * direction * args.max_jog_ticks_per_sec * dt
                target_position = int(round(goal_positions[index]))
                if target_position != last_written_positions[index] and now - last_write >= args.write_period_s:
                    controller.set_goal_position(selected_motor, target_position)
                    last_written_positions[index] = target_position
                    last_write = now
                if now - last_status > 0.25:
                    print(f"ID {selected_motor} goal position: {target_position}")
                    last_status = now

            time.sleep(0.005)

        final_present = controller.read_all_present_positions()
        print_positions("Final goal positions", goal_positions)
        print_positions("Final present positions", final_present)
        return 0
    finally:
        try:
            if controller.is_open:
                controller.enable_torque(False)
        finally:
            controller.close()
            joystick.close()


if __name__ == "__main__":
    raise SystemExit(run())
