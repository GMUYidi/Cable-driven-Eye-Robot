# Open and Closed Loop Control for Robotic Eye Platform

This repository contains the control code for the robotic eye platform used in the Scientific Reports manuscript titled:

**"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"**

The code implements both open-loop and closed-loop control strategies to drive the robotic eyeballs and simulate human eye movements. It uses Dynamixel servos, IMU feedback, and Pi Cameras to track desired gaze targets and record binocular vision videos.

## Overview

The control program includes:

- **Open-loop control**: Calculate servo commands from desired gaze angles using geometric models.
- **Closed-loop control**: Refine servo commands using real-time IMU feedback to achieve accurate eye orientation.
- **IMU data streaming**: Real-time yaw, pitch, roll feedback from Raspberry Pi via socket communication.
- **Camera capture**: Automatically capture images from left and right eyes when target positions are reached.
- **Dynamixel servo control**: Using Extended Position Control mode and current control.

## Hardware Requirements

- **12x Dynamixel XL330-M288-T servos**
- **Raspberry Pi Compute Module 4B+ with PiCamera2 modules (Left and Right)**
- **Bosch BNO055 IMU sensors (Left and Right eyes)**
- **PC with Python 3.x and Dynamixel SDK installed**
- **Network connection between PC and Raspberry Pi for IMU data streaming**

## Software Requirements

- Python 3.x
- Dynamixel SDK
- Numpy
- Pandas
- Scipy
- Picamera2 (for Raspberry Pi Camera)
- Socket communication (standard Python module)

## Running the Code

1. Connect Raspberry Pi (IMU and Cameras) and PC in the same local network.
2. Make sure the Raspberry Pi IMU streaming server is running.
3. Modify the following parameters in `open_close_loop_control.py` if needed:
   - `RASPBERRY_PI_IP`: IP address of Raspberry Pi
   - `data` path: Path to `.mat` file storing target gaze angles
4. Connect Dynamixel servos to PC and check `DEVICENAME` (e.g., COM3 or Linux equivalent).
5. Run the control script on PC:

```bash
python open_close_loop_control.py
