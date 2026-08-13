# Control and Calibration Architecture

## Control Objective

The control objective is to convert a desired gaze target into coordinated
two-eye motion, then measure how accurately the physical eyes follow that target.

The current control path is:

```text
joystick or scripted target
  -> desired yaw and pitch
  -> optional torsional roll from Listing-style geometry
  -> left/right eye kinematics
  -> six actuator pulse changes per eye
  -> Dynamixel goal positions
  -> physical cable-driven motion
  -> IMU and camera feedback
```

## Open-Loop Control

The main current demo is open-loop:

- The joystick defines target yaw and pitch.
- The model computes cable/muscle length changes.
- Motor goal positions are sent directly.
- There is no real-time correction from measured IMU error in the basic demo.

Open-loop control is useful because it tests the mechanical model, motor mapping,
and cable routing without hiding errors behind feedback.

## Feedback Control

Feedback-control experiment scripts use IMU pose as the actual eye orientation.
The controller compares target pose and actual pose, then updates the command.

The feedback-control demo uses this structure:

```text
joystick target
  -> target eye pose
  -> open-loop inverse kinematics
  -> feedforward motor pulses
  -> IMU pose error
  -> bounded PD-style correction on antagonistic motor pairs
  -> corrected motor command
```

This is intentionally conservative. The open-loop model remains the main motor
command, and feedback adds a bounded correction rather than replacing the model.

Feedback control should eventually be organized into explicit modes:

- Open-loop inverse kinematics.
- IMU feedback correction.
- Feedforward plus feedback.
- Calibration-grid collection.
- Offline replay.

The current scripted closed-loop experiment is:

```text
Hess-chart target
  -> left/right target eye pose
  -> open-loop inverse kinematics
  -> IMU feedback correction
  -> consecutive in-tolerance samples
  -> synchronized camera capture
  -> summary and sample telemetry
```

It is implemented in:

```text
demos/hess_chart_tracking.py
src/cable_driven_eye_robot/apps/hess_chart_tracking.py
src/cable_driven_eye_robot/data/hess_chart.py
```

## Kinematic Model

The modular control code represents each eye with:

- Eyeball radius.
- Muscle insertion points.
- Pulley points.
- Per-muscle pulse scales.
- Per-muscle offsets.

For each desired eye pose, the model rotates the muscle insertion points and
computes the change in cable path length relative to the pulley geometry.

The current model is implemented in:

```text
src/cable_driven_eye_robot/kinematics.py
```

## Torsion / Listing-Style Roll

The project includes a helper that estimates roll from horizontal and vertical
gaze angle. This gives a more eye-like torsional behavior than commanding yaw
and pitch alone.

The current helper is implemented in:

```text
src/cable_driven_eye_robot/math_utils.py
```

## Calibration Layers

The system needs several independent calibrations.

### Motor Neutral Calibration

Defines the "straight ahead" or neutral pose of every motor.

Outputs:

- Perfect motor positions.
- Baseline present positions.
- Per-eye neutral pose.

### Kinematic Calibration

Fits the relationship between target eye pose and actuator motion.

Outputs:

- Pulse scale per muscle.
- Pulse offset per muscle.
- Valid motion range.
- Expected error over a yaw/pitch grid.

### IMU Calibration

Maps raw IMU yaw, pitch, and roll into robot-eye pose.

Outputs:

- Zero pose.
- Axis sign convention.
- Roll offset.
- Valid sample age threshold.

### Camera Intrinsic Calibration

Corrects lens distortion and defines camera matrix.

Current calibration files are stored in:

```text
calibration/camera_intrinsics/
```

Outputs:

- Camera matrix.
- Distortion coefficients.
- Resolution.
- Camera identity.

### Pixel-to-Gaze Calibration

Maps eye pose to pixel location in the eye camera frame.

This is required for foveated visualization because the system must know where
the high-resolution foveal center should be placed in the image.

Recommended method:

1. Place a known visual target in front of the eyes.
2. Command a grid of yaw/pitch targets.
3. Capture camera frames and IMU pose at each grid point.
4. Detect the target in image coordinates.
5. Fit a mapping from `(yaw_deg, pitch_deg)` to `(u_px, v_px)`.
6. Validate the mapping on held-out points.

## Telemetry

Each experiment should log:

- Timestamp.
- Target yaw, pitch, and roll.
- Commanded left/right pose.
- Measured left/right IMU pose.
- Pose error.
- Motor goal positions.
- Motor present positions.
- Motor present current if available.
- Camera frame index or video timestamp.
- Experiment stage.
- Safety events.

This makes the project more credible because a demo can be backed by measured
tracking performance.

## Success Criteria

A mature demo should be able to show:

- Repeatable saccade or gaze-step commands.
- Target vs actual pose plots.
- Final fixation error.
- Overshoot and settling time.
- Camera view synchronized with eye pose.
- Gaze-center overlay in the image.
- Foveated rendering around the gaze center.
