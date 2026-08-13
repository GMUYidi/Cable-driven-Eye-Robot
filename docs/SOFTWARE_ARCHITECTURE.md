# Software Architecture

The repository separates runnable demo entry points from reusable robotics
modules.

1. `demos/` contains the small scripts a user runs from the command line.
2. `src/cable_driven_eye_robot/` contains the reusable implementation.

This keeps the project readable: demo files describe what is being run, while
the package owns the hardware drivers, sensing, kinematics, control, UI, and
calibration-related logic.

## Modular Package Layout

```text
src/cable_driven_eye_robot/
  config.py
  math_utils.py
  kinematics.py
  hardware/
    dynamixel.py
  input_devices/
    joystick.py
  data/
    eye_tracker.py
    hess_chart.py
  sensors/
    imu.py
    camera.py
    frame_capture.py
  control/
    open_loop.py
    feedback.py
  ui/
    state.py
    dashboard.py
  apps/
    dashboard_demo.py
    initial_position_tuner.py
    feedback_control.py
    hess_chart_tracking.py
    eye_tracker_replay.py
```

## Responsibility Boundaries

### Configuration

`config.py` centralizes motor IDs, motor-to-muscle mapping, Dynamixel settings,
joystick axes, IMU ports, camera URLs, safety limits, and feedback gains.

### Hardware

`hardware/dynamixel.py` owns all Dynamixel communication: port setup, Extended
Position Mode, profile velocity/acceleration, torque enable/disable, sync-read,
sync-write, startup baseline capture, and reset.

Application code should not directly call `PacketHandler` or `GroupSyncWrite`.

### Input

`input_devices/joystick.py` owns joystick setup and target reading. It converts
right-stick input into bounded yaw/pitch targets.

### Data

`data/eye_tracker.py` loads recorded eye-tracker `.mat` files and converts them
into left/right `EyePose` targets for replay experiments.

`data/hess_chart.py` generates scripted Hess-style gaze targets. It supports a
9-point center/cardinal/diagonal pattern and configurable yaw/pitch grids.

### Kinematics

`kinematics.py` owns the eye geometry and inverse kinematics. It computes
Listing-style roll, cable/muscle length changes, and the final 12-motor pulse
vector.

### Sensors

`sensors/imu.py` parses BNO055 serial data, applies zero calibration, and
publishes the latest pose sample.

`sensors/camera.py` opens ESP32-S3-CAM streams, reconnects after dropped frames,
and publishes the latest camera frame.

`sensors/frame_capture.py` captures still frames from ESP32-S3 camera streams
for scripted experiments.

### Control

`control/open_loop.py` implements the current joystick demo behavior:

```text
joystick right stick
  -> yaw/pitch target
  -> smoothed command
  -> Listing-style roll
  -> inverse kinematics
  -> 12 motor pulse commands
  -> Dynamixel sync-write
```

`control/feedback.py` implements a bounded PD-style correction around open-loop
inverse kinematics:

```text
open-loop IK motor command
  + IMU pose error correction
  -> corrected motor command
```

The feedback layer should be treated as a tunable baseline. The open-loop model
remains the main command, and feedback adds a bounded correction.

`apps/eye_tracker_replay.py` uses the same inverse-kinematic model to replay
recorded eye-tracker targets and capture camera frames after each commanded
pose settles. It is intentionally open-loop and should not be confused with the
IMU feedback controller.

`apps/hess_chart_tracking.py` is the scripted closed-loop experiment. It
generates Hess-chart targets, commands IK feedforward plus IMU feedback, waits
for consecutive in-tolerance pose samples, captures left/right camera frames,
and logs summary plus loop-level telemetry.

### UI

`ui/state.py` stores the latest command, IMU, camera, and log state.

`ui/dashboard.py` renders the Tkinter dashboard. The UI does not directly
command motors.

## Demo Entrypoints

Run from the repository root:

```bash
python demos/dashboard_demo.py
python demos/initial_position_tuner.py
python demos/feedback_control.py
python demos/hess_chart_tracking.py
python demos/eye_tracker_replay.py --data-path experiments/eye_tracking/processed_data/<file>.mat
```

The dashboard supports debug modes:

```bash
python demos/dashboard_demo.py --no-motors
python demos/dashboard_demo.py --no-serial
python demos/dashboard_demo.py --no-cameras
```

## Migration Strategy

The project should migrate in stages:

1. Keep demo entry points small and readable.
2. Put shared behavior into package modules.
3. Test each demo entry point on hardware.
4. Move new behavior into `src/cable_driven_eye_robot/` instead of copying code
   between scripts.
5. Keep research data, videos, and CAD files in documented folders.

This makes the repository easier to understand and easier to extend without
turning each new experiment into another large standalone script.
