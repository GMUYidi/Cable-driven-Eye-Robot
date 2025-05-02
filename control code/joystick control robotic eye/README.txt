Joystick-based Open-Loop Control for Robotic Eye Platform
This repository contains the control script for manually operating the robotic eye platform via a joystick (or mouse) in open-loop mode. The code allows users to intuitively control the robotic eye movements and test muscle coordination using pure forward geometric calculations without feedback.

This program was used as part of the experiments described in the Scientific Reports submission:

"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"

Overview
Manual control of robotic eye movements using joystick or mouse position

Open-loop geometric calculations of six simulated muscle lengths (cable drive model)

Direct servo actuation through calculated muscle extensions

Supports dual-eye simultaneous control (12 Dynamixel servos total)

No IMU feedback or closed-loop correction (purely kinematics-driven)

Hardware Requirements
12x Dynamixel XL330-M288-T servos

PC running Python

Joystick (optional) or Mouse (default mode)

Dynamixel USB2Dynamixel / U2D2 or USB-Serial converter

Software Requirements
Python 3.x

Dynamixel SDK (Python)

Pygame

Numpy

PyAutoGUI (optional, for mouse-based control)

How to Run
1. Make sure Dynamixel servos are connected and powered, and the USB port name in the script (DEVICENAME) matches your system (e.g., COM14 on Windows).

2. Install required Python libraries if not yet installed:
pip install numpy pygame pyautogui dynamixel_sdk

3. Run the script:
python write_position_velocity_pointer_twoeyes_joystick_bothcontrol.py

4. Move your mouse (default) or connect a joystick to control the robotic eyes.

Horizontal movement → Yaw (horizontal rotation)

Vertical movement → Pitch (vertical rotation)

Torsion (roll) is automatically calculated based on Listing's law.

5. To stop the program, press CTRL + C or close the terminal.

Notes
This script implements pure open-loop control, meaning there is no feedback correction. Eye positions are determined only by joystick/mouse input and geometric calculations.

Exceedingly large movements are automatically clamped to avoid overdriving servos.

When the program ends (normally or interrupted), it will automatically reset all servos to the initial (straight-ahead) position and disable torque.

License
This repository is provided under the MIT License for academic use.
