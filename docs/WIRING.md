# Wiring and Power

This document describes the current wiring architecture for the cable-driven
robotic-eye platform.

## Dynamixel Power and Communication

The Dynamixel bus is connected through a ROBOTIS U2D2 and U2D2 Power Hub Board.

```text
Computer USB
   |
   | USB: communication + power for U2D2 electronics
   v
U2D2
   |
   | TTL / RS485 Dynamixel communication
   v
U2D2 Power Hub Board <---- external supply
                             reported: 12 V / 5 A adapter
                             required: verify/regulate for XL330 bus voltage
   |
   | communication + servo power
   v
Dynamixel XL330-M288-T servos
```

## Important Safety Checks

Before powering the robot:

- Confirm the actual voltage measured at the Dynamixel bus.
- Confirm that the bus voltage is safe for the XL330-M288-T servos.
- Confirm that the adapter polarity matches the Power Hub requirement.
- Confirm that the servo bus connectors are fully seated.
- Confirm that no cable is wrapped around a moving joint or spool.
- Start with the robot near its neutral mechanical pose.
- Keep access to the power switch or adapter plug during first motion tests.

## Dynamixel Bus

The current robot uses 12 Dynamixel XL330-M288-T servos on one coordinated bus.

| Motor ID | Eye | Muscle / Cable |
| --- | --- | --- |
| 1 | Left | LR |
| 2 | Left | MR |
| 3 | Left | SR |
| 4 | Left | IR |
| 5 | Left | SO |
| 6 | Left | IO |
| 7 | Right | MR |
| 8 | Right | LR |
| 9 | Right | SR |
| 10 | Right | IR |
| 11 | Right | SO |
| 12 | Right | IO |

Recommended physical practice:

- Label each motor with its ID.
- Label each cable with its muscle name.
- Photograph the final wiring and cable routing.
- Keep a saved Dynamixel Wizard screenshot showing all detected IDs.
- Save a measured bus-voltage note before running the motors.

## IMU Connections

The IMU model is:

```text
Adafruit 9-DOF Absolute Orientation IMU Fusion Breakout - BNO055
```

The current host-side demo expects two IMU streams:

```text
Left IMU:  COM11
Right IMU: COM6
Baudrate:  921600
```

These port names are machine-specific. If another computer is used, update the
port names in the demo script or pass command-line arguments when supported.

## Camera Connections

The camera model is:

```text
ESP32-S3-CAM Development Board with OV3660 Camera + Antenna
ESP32-S3 N16R8 module, 16 MB flash, 8 MB PSRAM
```

The current integrated dashboard uses Wi-Fi camera streams. Current default
stream URLs are:

```text
Left camera:  http://192.168.4.1/stream
Right camera: http://192.168.4.200/stream
```

These IP addresses depend on the ESP32 Wi-Fi mode and network configuration.

## Firmware

ESP32 camera/IMU firmware:

```text
firmware/esp32_imu_camera/IMUcamera.ino
```

Local Wi-Fi credentials should be stored in `wifi_secrets.h`. Do not commit that
file to git.

## Still Needed

To complete this page, add:

- A photo of the real U2D2 + Power Hub wiring.
- A measured voltage value at the Dynamixel bus.
- A photo of the full Dynamixel bus.
- A photo of left/right IMU placement.
- A photo of left/right ESP32-S3-CAM mounting.
- A drawn or exported wiring diagram.
