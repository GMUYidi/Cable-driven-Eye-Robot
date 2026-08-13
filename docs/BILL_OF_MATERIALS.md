# Bill of Materials

This document lists the main hardware required to reproduce the current
cable-driven binocular robotic-eye platform.

## Core Actuation

| Item | Quantity | Current Part / Specification | Purpose |
| --- | ---: | --- | --- |
| Dynamixel servo | 12 | ROBOTIS Dynamixel XL330-M288-T | Cable actuation, 6 motors per eye |
| U2D2 | 1 | ROBOTIS U2D2 | USB-to-Dynamixel communication |
| U2D2 Power Hub Board | 1 | ROBOTIS U2D2 Power Hub Board | Injects external power into Dynamixel bus |
| DC power adapter | 1 | Reported current adapter: 12 V / 5 A | External supply for Power Hub; verify/regulate actual servo bus voltage before use |

## Sensors

| Item | Quantity | Current Part / Specification | Purpose |
| --- | ---: | --- | --- |
| IMU | 2 | Adafruit 9-DOF Absolute Orientation IMU Fusion Breakout - BNO055 | Measures left/right eye pose |
| Camera board | 2 | ESP32-S3-CAM Development Board with OV3660 Camera + Antenna, ESP32-S3 N16R8, 16 MB flash, 8 MB PSRAM | Eye-view camera stream |

## Mechanical Parts

The current 3D-printable parts are stored in:

```text
hardware/cad/fusion360_stl/
```

Existing STL exports include:

- Front half of the eyeball adapted for the BNO055 IMU.
- Rear half of the eyeball adapted for the BNO055 IMU.
- Main base of left eye.
- Main base of right eye.
- Front eye support of left eye.
- Front eye support of right eye.
- Servo spool 1.
- Servo spool 2.
- Bolt.

## Consumables and Mechanical Hardware

The exact fastener and cable list still needs to be finalized. The repository
should eventually document:

- Cable or string type and diameter.
- Screws, nuts, bearings, inserts, and shafts.
- Spool dimensions.
- Cable anchor method.
- 3D printing material and print settings.

## Power Notes

The reported servo power path is:

```text
external adapter -> U2D2 Power Hub Board -> Dynamixel servo bus
```

The computer USB connection powers the U2D2 electronics, but the servos should
be powered by the external adapter through the Power Hub Board.

Important: the reported adapter is 12 V / 5 A, but the XL330-M288-T servo bus
voltage must be verified or regulated before use. Do not treat the adapter
voltage as automatically safe for the servo unless the Power Hub/regulator
configuration has been measured.

## Information Still Needed

To make this BOM complete, add:

- Purchase links for each part.
- Confirmed servo bus voltage measured at the Dynamixel connector.
- Exact number and size of screws/fasteners.
- Cable material and length.
- 3D print settings.
- Any custom PCB, adapter, or mounting hardware.
