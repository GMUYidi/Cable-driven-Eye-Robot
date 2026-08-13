# Repository Structure

This repository contains the robotic-eye demo software, documentation, firmware,
calibration files, and research artifacts.

## Top-Level Layout

```text
README.md
ARCHITECTURE.md
requirements.txt
demos/
src/cable_driven_eye_robot/
docs/
firmware/
hardware/
calibration/
experiments/
media/
tools/
```

These folders are grouped by responsibility so a reviewer can quickly separate
robot runtime code from hardware files, calibration assets, experiment results,
and project maintenance tools.

## Folder Responsibilities

```text
demos/                         runnable demo entry points
src/cable_driven_eye_robot/     reusable Python robotics package
docs/                           architecture, hardware, wiring, runbook
firmware/                       ESP32-S3 camera and IMU firmware
hardware/cad/                   mechanical design and STL files
calibration/camera_intrinsics/  camera matrix and distortion files
experiments/open_loop_accuracy/ open-loop IK accuracy dataset
experiments/eye_tracking/       participant eye-tracking data
experiments/binocular_visualization/
                                homography, strabismus, and binocular visualization outputs
media/videos/                   experiment and synthesized vision videos
tools/fusion360/                Fusion 360 CAD export helpers
```

## Recommended Future Additions

Long term, the project should move toward:

```text
src/cable_driven_eye_robot/
  hardware/
  kinematics/
  control/
  sensors/
  calibration/
  telemetry/
  visualization/
  experiments/

demos/
  dashboard_demo.py
  initial_position_tuner.py
  feedback_control.py
  hess_chart_tracking.py
  eye_tracker_replay.py
  scripted_saccade/
  foveated_view/

docs/
  architecture
  hardware
  bill of materials
  wiring
  software architecture
  control
  calibration
  experiments
  troubleshooting
```

The current structure gives reviewers a clean entry point while leaving room for
future experiments such as scripted saccades and foveated-view rendering.
