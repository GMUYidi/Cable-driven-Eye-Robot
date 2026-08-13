# Demo Runbook

This document describes how to run the current robotic-eye demonstrations in a
repeatable and safe way.

## Demo Goals

The demo should show that the system is more than a motor script:

- A joystick command becomes a desired binocular gaze target.
- Gaze target is converted into cable/muscle actuator commands.
- Commands are sent to 12 Dynamixel motors.
- IMU feedback reports actual eye orientation.
- Eye-mounted cameras show the scene from the robot's viewpoint.
- The dashboard visualizes motion, sensing, and camera data together.

## Hardware Checklist

Before running the demo:

- The U2D2 is connected to the computer over USB.
- The U2D2 Power Hub Board is connected to the Dynamixel bus.
- The external adapter is connected to the Power Hub Board.
- The actual Dynamixel bus voltage has been measured and confirmed safe for the
  XL330-M288-T servos.
- All 12 Dynamixel IDs are available on the bus.
- Left and right IMUs are connected.
- Left and right ESP32 camera streams are powered.
- Joystick is connected.
- Cables do not bind at the expected motion range.
- The eyes are near the calibrated neutral pose.

## Default Port and Stream Configuration

Current default values used by the newer demo code:

```text
Servo model:    Dynamixel XL330-M288-T
Motor IDs:      1-12
Power path:     external adapter -> U2D2 Power Hub -> Dynamixel bus
Power note:     reported adapter is 12 V / 5 A; verify/regulate bus voltage
Dynamixel bus: COM12
Left IMU:      COM11
Right IMU:     COM6
IMU baud:      921600
Left camera:   http://192.168.4.1/stream
Right camera:  http://192.168.4.200/stream
```

If hardware ports differ on another computer, override them from the command
line or update `src/cable_driven_eye_robot/config.py`.

## Python Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Important packages:

- `dynamixel-sdk`
- `pygame`
- `pyserial`
- `opencv-python`
- `pillow`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Demo 1: Integrated Joystick, IMU, and Camera Dashboard

Run:

```bash
python demos/dashboard_demo.py
```

What to show:

- Right joystick movement drives horizontal and vertical gaze.
- Both eyes move together through open-loop inverse kinematics.
- Joystick command and physical eye motion.
- Left and right IMU eye-pose visualization.
- Left and right camera streams.
- Connection status and live dashboard behavior.
- The mechanism returns to neutral when the program exits.
- Safety clamps prevent very large commands.

Debug modes:

```bash
python demos/dashboard_demo.py --no-motors
python demos/dashboard_demo.py --no-serial
python demos/dashboard_demo.py --no-cameras
```

Use `--no-motors` when testing the UI without moving hardware.

## Demo 2: Manual Initial Position Tuning

Run:

```bash
python demos/initial_position_tuner.py
```

Purpose:

- Select one motor at a time with the controller buttons.
- Jog that motor with the right stick.
- Print the final 12 present positions.
- Use those positions as candidate neutral or `perfect_positions` values.

## Demo 3: Joystick IMU Feedback Control

Run:

```bash
python demos/feedback_control.py
```

Control concept:

```text
joystick target
  -> open-loop inverse-kinematic feedforward
  -> IMU yaw/pitch/roll feedback
  -> bounded PD-style motor correction
  -> corrected Dynamixel command
```

This feedback controller is a tunable baseline. Gains and signs should be
validated on hardware before treating it as final.

## Demo 4: Eye-Tracker Replay and Camera Capture

Run:

```bash
python demos/eye_tracker_replay.py --data-path experiments/eye_tracking/processed_data/<file>.mat
```

Purpose:

- Load recorded eye-tracker gaze targets from a MATLAB `.mat` file.
- Convert each left/right target angle into motor commands using the inverse
  kinematic model.
- Wait briefly for the commanded pose to settle.
- Capture one left and one right camera frame for each target.
- Save an experiment log under `outputs/eye_tracker_replay/`.

This demo is open-loop. It replays recorded gaze targets through the kinematic
model and does not apply PD feedback correction from IMU pose error.

## Demo 5: Hess-Chart IMU Feedback Tracking

Run the default 9-point target sequence:

```bash
python demos/hess_chart_tracking.py
```

Run a custom yaw/pitch grid:

```bash
python demos/hess_chart_tracking.py --pattern grid --yaw-points=-10,-5,0,5,10 --pitch-points=-8,0,8
```

Purpose:

- Generate a scripted Hess-style gaze sequence.
- Convert each target into inverse-kinematic feedforward motor commands.
- Use left/right IMU pose error to add bounded PD correction.
- Require several consecutive in-tolerance IMU samples before accepting a
  target.
- Capture synchronized left/right eye-camera frames after the target is reached.
- Save target-level and loop-level telemetry under `outputs/hess_chart_tracking/`.

Dry run without moving hardware or opening cameras:

```bash
python demos/hess_chart_tracking.py --no-motors --no-cameras --no-imu
```

This is the cleaner replacement for the older monolithic "open/close loop"
experiment script: the target sequence, feedback controller, hardware driver,
IMU reader, camera capture, and telemetry logging are now separate modules.

## Safety Procedure

Before motion:

- Keep one hand near the motor power switch.
- Confirm the Dynamixel bus is powered through the U2D2 Power Hub Board.
- Confirm the measured bus voltage is safe for XL330-M288-T before enabling
  torque.
- Confirm joystick axes are centered.
- Confirm the robot starts near neutral.
- Start with small motions.

During motion:

- Watch for cable slack, over-tension, or mechanical contact.
- Stop if any motor moves in the wrong direction.
- Stop if the camera or IMU cable interferes with the eye.

After motion:

- Return to neutral.
- Disable motor torque.
- Save any logs or videos from the run.

## Demo Narrative for Interviews

A concise explanation:

> The demo starts from a high-level gaze command, generated by a joystick. The
> controller maps that command into binocular yaw, pitch, and roll, computes the
> corresponding cable/muscle actuator changes, and sends synchronized commands to
> 12 Dynamixel motors. IMUs measure the actual eye pose, while eye-mounted
> cameras stream the visual scene. This lets the same platform support hardware
> motion demos, closed-loop control experiments, and future saccade/foveation
> visualization.
