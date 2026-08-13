# Hardware Architecture

This document describes the current hardware architecture of the cable-driven
binocular robotic-eye platform. The goal is to make the physical system
understandable and reproducible from the repository.

## Physical System

The platform is a binocular cable-driven robotic-eye system. Each eye is driven
by six actuator channels inspired by the six human extraocular muscles:

- LR: lateral rectus
- MR: medial rectus
- SR: superior rectus
- IR: inferior rectus
- SO: superior oblique
- IO: inferior oblique

The full platform uses 12 Dynamixel servos total, with 6 servos per eye.

## Actuation

The current actuator configuration is:

```text
Servo model:       ROBOTIS Dynamixel XL330-M288-T
Servo count:       12
Servos per eye:    6
Protocol:          Dynamixel Protocol 2.0
Host interface:    ROBOTIS U2D2
Power interface:   ROBOTIS U2D2 Power Hub Board
External supply:   reported as 12 V / 5 A DC adapter; verify/regulate bus voltage
Control mode:      Extended Position Mode for the current demos
```

The current Python demos command the motors through synchronized Dynamixel
position writes. The 12 servos are treated as one coordinated actuator group so
that all cable commands are updated together.

## Motor ID and Muscle Map

This is the current motor-to-muscle mapping used by the physical robot.

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

Note that the left and right horizontal rectus ordering is mirrored:

```text
Left eye IDs 1-2:   LR, MR
Right eye IDs 7-8:  MR, LR
```

This matters for both cable routing and sign conventions in the kinematic model.

## Power and Communication Chain

The Dynamixel bus uses the U2D2 for communication and the U2D2 Power Hub Board
for servo power injection.

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

Important electrical notes:

- The computer USB powers the U2D2 communication interface.
- The external supply powers the Dynamixel servos through the Power Hub Board.
- The Power Hub Board carries both Dynamixel communication and servo power to
  the bus.
- Always confirm the supply polarity and voltage before connecting the motors.
- The XL330-M288-T operating voltage must be checked against the official
  ROBOTIS specification before applying power. Do not assume the Power Hub Board
  regulates the input voltage unless that has been verified on the physical
  setup.

## Sensing

The current sensing stack includes:

- Left eye IMU.
- Right eye IMU.
- Left eye camera.
- Right eye camera.
- Dynamixel present-position feedback.
- Optional Dynamixel present-current feedback in analysis scripts.

## IMU

The IMU module is:

```text
Adafruit 9-DOF Absolute Orientation IMU Fusion Breakout - BNO055
```

The BNO055 is used to measure eye orientation, including yaw, pitch, and roll.
The current demo software reads IMU data over serial and uses it for live
visualization and feedback-control experiments.

## Camera

The eye camera module is:

```text
ESP32-S3-CAM Development Board with OV3660 Camera + Antenna
ESP32-S3 N16R8 module, 16 MB flash, 8 MB PSRAM
```

The camera provides the robot-eye viewpoint for visualization. In the current
dashboard, the host computer reads left and right camera streams over Wi-Fi.

The ESP32-S3 camera firmware is located at:

```text
firmware/esp32_imu_camera/IMUcamera.ino
```

The firmware includes:

- ESP32 camera streaming.
- BNO055 IMU communication.
- Serial IMU output.
- Wi-Fi access point or client configuration.

Do not commit local Wi-Fi credentials. Put credentials in `wifi_secrets.h`,
which is ignored by git.

## Mechanical Design Files

Existing mechanical files are stored in:

```text
hardware/cad/fusion360_stl/
```

This folder includes STL exports for the eyeball, supports, base parts, spools,
and related mechanical pieces.

## Documentation Still Needed

To make the hardware fully reproducible, add the following:

- Front, side, rear, and cable-routing photos of the robot.
- Motor ID labels visible on the physical robot.
- Cable routing diagram for all 12 actuator channels.
- Wiring diagram photo or schematic.
- Assembly screenshots from Fusion 360.
- Final bill of materials with purchase links.
