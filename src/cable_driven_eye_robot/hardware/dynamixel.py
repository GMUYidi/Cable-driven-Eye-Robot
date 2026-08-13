from __future__ import annotations

from dataclasses import dataclass

from ..config import DynamixelConfig, ProfileConfig
from ..math_utils import fix_signed_16bit, fix_signed_32bit, unwrap_to_reference


@dataclass
class MotorTelemetry:
    present_positions: list[float]
    present_currents_raw: list[float]
    present_currents_ma: list[float]


class DynamixelController:
    def __init__(self, config: DynamixelConfig, profile: ProfileConfig):
        self.config = config
        self.profile = profile
        self.dxl = None
        self.port_handler = None
        self.packet_handler = None
        self._sync_read_position = None
        self._sync_read_current = None
        self.initial_positions: list[int] = []
        self.previous_goal_positions: list[int] = []
        self.is_open = False

    def open(self, configure: bool = True, torque: bool = True) -> None:
        import dynamixel_sdk as dxl

        self.dxl = dxl
        self.port_handler = dxl.PortHandler(self.config.device_name)
        self.packet_handler = dxl.PacketHandler(self.config.protocol_version)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port: {self.config.device_name}")
        self.is_open = True

        if not self.port_handler.setBaudRate(self.config.baudrate):
            self.close()
            raise RuntimeError(f"Failed to set Dynamixel baudrate: {self.config.baudrate}")

        if configure:
            self.set_extended_position_mode_all()
            self.init_profiles_all()

        self.init_sync_readers()
        if torque:
            self.enable_torque(True)
        self.capture_current_as_baseline()

    def close(self) -> None:
        if self.port_handler is not None and self.is_open:
            self.port_handler.closePort()
        self.is_open = False

    def require_open(self) -> None:
        if not self.is_open or self.dxl is None or self.port_handler is None or self.packet_handler is None:
            raise RuntimeError("DynamixelController is not open.")

    def enable_torque(self, enable: bool = True) -> None:
        self.require_open()
        value = self.config.torque_enable if enable else self.config.torque_disable
        for motor_id in self.config.motor_ids:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.config.table.torque_enable,
                value,
            )
            if result != self.dxl.COMM_SUCCESS:
                raise RuntimeError(
                    f"Failed torque write on ID {motor_id}: {self.packet_handler.getTxRxResult(result)}"
                )
            if error:
                raise RuntimeError(
                    f"Torque write error on ID {motor_id}: {self.packet_handler.getRxPacketError(error)}"
                )

    def set_extended_position_mode_all(self) -> None:
        self.require_open()
        for motor_id in self.config.motor_ids:
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.config.table.torque_enable,
                self.config.torque_disable,
            )
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.config.table.operating_mode,
                self.config.extended_position_mode,
            )
            if result != self.dxl.COMM_SUCCESS:
                raise RuntimeError(
                    f"Failed to set Extended Position Mode on ID {motor_id}: "
                    f"{self.packet_handler.getTxRxResult(result)}"
                )
            if error:
                raise RuntimeError(
                    f"Mode set error on ID {motor_id}: {self.packet_handler.getRxPacketError(error)}"
                )

    def init_profiles_all(self) -> None:
        self.require_open()
        for motor_id in self.config.motor_ids:
            if motor_id <= 6:
                accel = self.profile.accel_left
                velocity = self.profile.velocity_left
            else:
                accel = self.profile.accel_right
                velocity = self.profile.velocity_right
            self.set_profile(motor_id, accel, velocity)

    def set_profile(self, motor_id: int, accel: int, velocity: int) -> None:
        self.require_open()
        self.packet_handler.write4ByteTxRx(
            self.port_handler,
            motor_id,
            self.config.table.profile_accel,
            int(accel),
        )
        self.packet_handler.write4ByteTxRx(
            self.port_handler,
            motor_id,
            self.config.table.profile_velocity,
            int(velocity),
        )

    def init_sync_readers(self) -> None:
        self.require_open()
        self._sync_read_position = self.dxl.GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            self.config.table.present_position,
            4,
        )
        self._sync_read_current = self.dxl.GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            self.config.table.present_current,
            2,
        )
        for motor_id in self.config.motor_ids:
            self._sync_read_position.addParam(motor_id)
            self._sync_read_current.addParam(motor_id)

    def read_all_present_positions(self) -> list[int]:
        self.require_open()
        if self._sync_read_position is None:
            self.init_sync_readers()

        positions = [0] * len(self.config.motor_ids)
        result = self._sync_read_position.txRxPacket()
        if result != self.dxl.COMM_SUCCESS:
            return [self.read_single_present_position(motor_id) for motor_id in self.config.motor_ids]

        for index, motor_id in enumerate(self.config.motor_ids):
            if self._sync_read_position.isAvailable(motor_id, self.config.table.present_position, 4):
                raw = self._sync_read_position.getData(motor_id, self.config.table.present_position, 4)
                positions[index] = fix_signed_32bit(raw)
            else:
                positions[index] = self.read_single_present_position(motor_id)
        return positions

    def read_single_present_position(self, motor_id: int) -> int:
        self.require_open()
        raw, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler,
            motor_id,
            self.config.table.present_position,
        )
        if result != self.dxl.COMM_SUCCESS or error:
            return 0
        return fix_signed_32bit(raw)

    def read_all_present_currents_raw(self) -> list[float]:
        self.require_open()
        if self._sync_read_current is None:
            self.init_sync_readers()

        currents: list[float] = [float("nan")] * len(self.config.motor_ids)
        result = self._sync_read_current.txRxPacket()
        if result == self.dxl.COMM_SUCCESS:
            for index, motor_id in enumerate(self.config.motor_ids):
                if self._sync_read_current.isAvailable(motor_id, self.config.table.present_current, 2):
                    raw = self._sync_read_current.getData(motor_id, self.config.table.present_current, 2)
                    currents[index] = fix_signed_16bit(raw)
            return currents

        for index, motor_id in enumerate(self.config.motor_ids):
            raw, read_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler,
                motor_id,
                self.config.table.present_current,
            )
            if read_result == self.dxl.COMM_SUCCESS and not error:
                currents[index] = fix_signed_16bit(raw)
        return currents

    def read_telemetry(self) -> MotorTelemetry:
        positions = [float(value) for value in self.read_all_present_positions()]
        current_raw = self.read_all_present_currents_raw()
        current_ma = [
            value * 2.69 if isinstance(value, (int, float)) else float("nan")
            for value in current_raw
        ]
        return MotorTelemetry(positions, current_raw, current_ma)

    def capture_current_as_baseline(self) -> list[int]:
        self.initial_positions = self.read_all_present_positions()
        self.previous_goal_positions = self.initial_positions.copy()
        return self.initial_positions.copy()

    def set_goal_position(self, motor_id: int, position: int) -> None:
        self.require_open()
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            motor_id,
            self.config.table.goal_position,
            int(position),
        )
        if result != self.dxl.COMM_SUCCESS:
            raise RuntimeError(
                f"Failed goal write on ID {motor_id}: {self.packet_handler.getTxRxResult(result)}"
            )
        if error:
            raise RuntimeError(
                f"Goal write error on ID {motor_id}: {self.packet_handler.getRxPacketError(error)}"
            )

    def command_position_diffs(self, diff_list: list[float], do_unwrap: bool | None = None) -> list[int]:
        self.require_open()
        if do_unwrap is None:
            do_unwrap = self.config.do_unwrap

        count = min(
            len(self.config.motor_ids),
            len(diff_list),
            len(self.initial_positions),
            len(self.previous_goal_positions),
        )
        target_positions: list[int] = []
        for index in range(count):
            desired_abs = int(self.initial_positions[index] + diff_list[index])
            if do_unwrap:
                goal = unwrap_to_reference(
                    desired_abs,
                    self.previous_goal_positions[index],
                    self.config.ticks_per_rev,
                )
            else:
                goal = desired_abs

            delta = goal - self.previous_goal_positions[index]
            if delta > self.config.max_goal_step_ticks:
                goal = self.previous_goal_positions[index] + self.config.max_goal_step_ticks
            elif delta < -self.config.max_goal_step_ticks:
                goal = self.previous_goal_positions[index] - self.config.max_goal_step_ticks
            target_positions.append(int(goal))

        sync_write = self.dxl.GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            self.config.table.goal_position,
            4,
        )
        for motor_id, goal_position in zip(self.config.motor_ids, target_positions):
            if not sync_write.addParam(motor_id, self._pack_goal_position(goal_position)):
                raise RuntimeError(f"Failed to add SyncWrite goal for ID {motor_id}")

        result = sync_write.txPacket()
        sync_write.clearParam()
        if result != self.dxl.COMM_SUCCESS:
            raise RuntimeError(f"SyncWrite failed: {self.packet_handler.getTxRxResult(result)}")

        self.previous_goal_positions = target_positions.copy()
        return target_positions

    def reset_to_baseline(self) -> None:
        self.command_position_diffs([0.0] * len(self.config.motor_ids), do_unwrap=False)

    @staticmethod
    def _pack_goal_position(goal_position: int) -> list[int]:
        import dynamixel_sdk as dxl

        return [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(goal_position)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(goal_position)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(goal_position)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(goal_position)),
        ]
