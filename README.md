# Cable-Driven Eye Robot

Robotic-eye platform for human vision visualization, binocular gaze control,
strabismus simulation, and saccade/foveation experiments.

This repository contains hardware design files, control software, calibration
data, experiment scripts, videos, and analysis tools for a two-eye cable-driven
robotic platform. The current engineering direction is to organize the project
as a full robotics system: hardware, sensing, actuation, kinematics, calibration,
control, telemetry, and visual perception.

## Project Goal

The platform is designed to visualize what a moving eye sees during human-like
eye motion. In particular, the project focuses on:

- Cable-driven robotic eye actuation with 12 Dynamixel motors.
- Binocular yaw, pitch, and torsional eye motion.
- Open-loop inverse-kinematic control from desired gaze angle to motor command.
- IMU-based feedback and pose measurement.
- Dual camera streaming from the eye viewpoint.
- Camera and eye-pose calibration.
- Saccade, fixation, and foveated-perception visualization.
- Data collection for human vision visualization and strabismus diagnosis.

The long-term system objective is:

> From a high-level gaze or saccade command, generate coordinated two-eye motor
> motion, measure the actual eye pose, capture the visual scene from the eye's
> viewpoint, and render what the eye sees during fixation and saccadic motion.

## System Architecture

The project is organized around a layered robotics architecture:

```text
Experiment protocol
  -> gaze target or saccade command
  -> binocular motion planner
  -> eye kinematics and calibration
  -> Dynamixel motor command generation
  -> physical cable-driven eye motion
  -> IMU and camera feedback
  -> synchronized telemetry
  -> foveated visualization and analysis
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document.

## Demo Programs

The runnable demos are thin entry points in [`demos`](demos). They call reusable
modules from [`src/cable_driven_eye_robot`](src/cable_driven_eye_robot), so the
hardware drivers, kinematics, control loops, sensors, and UI are not duplicated
inside large one-file scripts.

```bash
python demos/dashboard_demo.py
python demos/initial_position_tuner.py
python demos/feedback_control.py
python demos/hess_chart_tracking.py
python demos/eye_tracker_replay.py --data-path experiments/eye_tracking/processed_data/<file>.mat
```

- `dashboard_demo.py`: joystick open-loop control, dual IMU pose display, and
  dual ESP32 camera streaming.
- `initial_position_tuner.py`: manual neutral-position calibration for all 12
  Dynamixel motors.
- `feedback_control.py`: joystick target plus IMU feedback correction and CSV
  telemetry logging.
- `hess_chart_tracking.py`: generate Hess-chart gaze targets, track each point
  with IMU feedback, then capture synchronized eye-camera frames when the pose
  reaches tolerance.
- `eye_tracker_replay.py`: replay recorded eye-tracker gaze targets with
  open-loop inverse kinematics and capture synchronized eye-camera frames.

## Repository Structure

```text
.
|-- README.md
|-- ARCHITECTURE.md
|-- requirements.txt
|-- demos/
|   |-- dashboard_demo.py
|   |-- initial_position_tuner.py
|   |-- feedback_control.py
|   |-- hess_chart_tracking.py
|   `-- eye_tracker_replay.py
|-- src/
|   `-- cable_driven_eye_robot/
|       |-- apps/
|       |-- control/
|       |-- data/
|       |-- hardware/
|       |-- input_devices/
|       |-- sensors/
|       |-- ui/
|       |-- config.py
|       `-- kinematics.py
|-- docs/
|   |-- DEMO_RUNBOOK.md
|   |-- HARDWARE.md
|   |-- BILL_OF_MATERIALS.md
|   |-- WIRING.md
|   |-- SOFTWARE_ARCHITECTURE.md
|   |-- CONTROL_AND_CALIBRATION.md
|   |-- REPOSITORY_STRUCTURE.md
|   `-- FILES_NEEDED.md
|-- firmware/
|   `-- esp32_imu_camera/
|-- calibration/
|   `-- camera_intrinsics/
|-- hardware/
|   `-- cad/
|-- experiments/
|   |-- open_loop_accuracy/
|   |-- eye_tracking/
|   `-- binocular_visualization/
|-- media/
|   `-- videos/
`-- tools/
    `-- fusion360/
```

The top-level folders are grouped by responsibility. `src` and `demos` are the
runtime software, `hardware` and `firmware` describe the physical system,
`calibration` stores reusable calibration artifacts, `experiments` stores
research datasets and analysis outputs, `media` stores videos, and `tools`
stores project maintenance scripts.

Representative binocular visualization outputs are included under
[`experiments/binocular_visualization/hess_matched_results`](experiments/binocular_visualization/hess_matched_results).
This result set contains Hess-matched strabismus simulation videos, a healthy
binocular reference video, and the reports/metadata/matrix files needed to
trace each generated video.

## Hardware Summary

- 2 cable-driven robotic eyes.
- 12 ROBOTIS Dynamixel XL330-M288-T actuators, 6 per eye.
- Dynamixel Protocol 2.0 over USB serial.
- ROBOTIS U2D2 and U2D2 Power Hub Board.
- Servo power is injected through the U2D2 Power Hub Board; the currently
  reported adapter is 12 V / 5 A, but the XL330-M288-T bus voltage must be
  verified or regulated before operation.
- Adafruit BNO055 IMU sensing for eye orientation.
- ESP32-S3-CAM OV3660 modules for eye-view video streaming.
- Joystick input for manual gaze control.
- Fusion 360 mechanical design files and exported STL parts.

See [docs/HARDWARE.md](docs/HARDWARE.md),
[docs/BILL_OF_MATERIALS.md](docs/BILL_OF_MATERIALS.md), and
[docs/WIRING.md](docs/WIRING.md) for more detail.

## Software Summary

The control software currently supports:

- Dynamixel sync-write of goal positions.
- Dynamixel sync-read of present positions.
- Extended Position Mode.
- Joystick input through `pygame`.
- Inverse-kinematic cable/muscle pulse computation.
- Listing-law-inspired torsional roll computation.
- IMU serial parsing.
- ESP32 camera stream display through OpenCV.
- CSV logging and offline visualization scripts.

See [docs/CONTROL_AND_CALIBRATION.md](docs/CONTROL_AND_CALIBRATION.md) for the
control and calibration model, and
[docs/SOFTWARE_ARCHITECTURE.md](docs/SOFTWARE_ARCHITECTURE.md) for the modular
software layout.

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the integrated joystick, IMU, and camera dashboard:

```bash
python demos/dashboard_demo.py
```

Tune the neutral motor positions:

```bash
python demos/initial_position_tuner.py
```

Run the IMU feedback-control demo:

```bash
python demos/feedback_control.py
```

For detailed hardware setup, port configuration, and safety checks, see
[docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md).

## Engineering Status

This repository currently contains both research artifacts used to produce
experimental results and a modular runtime for the current robotic-eye demos.

The current refactor direction is to separate:

- Experiment protocols.
- Gaze target generation.
- Saccade and fixation planning.
- Binocular coordination.
- Eye kinematics.
- Dynamixel transport.
- IMU and camera sensing.
- Calibration loading.
- Telemetry logging.
- Foveated visualization.

## Citation / Research Context

This project supports the research work:

**A Novel Cable-driven Eye Robot for Human Vision Visualization and Strabismus
Diagnosis**

The repository includes control code, hardware design files, eye-tracking data,
strabismus simulation resources, and videos related to that work.

## Contact

Yidi Huang
George Mason University
Email: yhuang35@gmu.edu
