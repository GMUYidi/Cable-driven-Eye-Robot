# Files and Information Needed

This checklist tracks what is already documented and what still needs to be
added so that another person can understand and reproduce the project.

## Already Documented

The following information has been added to the repository:

- Main actuator model: ROBOTIS Dynamixel XL330-M288-T.
- Actuator count: 12 total, 6 per eye.
- Power architecture: U2D2 + U2D2 Power Hub Board.
- Reported servo power adapter: external 12 V / 5 A DC adapter.
- IMU model: Adafruit 9-DOF Absolute Orientation IMU Fusion Breakout - BNO055.
- Camera model: ESP32-S3-CAM Development Board with OV3660 Camera + Antenna,
  ESP32-S3 N16R8 module, 16 MB flash, 8 MB PSRAM.
- Current motor-to-muscle map:
  - Left eye IDs 1-6: LR, MR, SR, IR, SO, IO.
  - Right eye IDs 7-12: MR, LR, SR, IR, SO, IO.

## Highest Priority Still Needed

1. Clear photos of the physical robot.
   - Front view.
   - Side view.
   - Back/wiring view.
   - Close-up of one eye and cable routing.

2. A short demo video.
   - Joystick moving both eyes.
   - Dashboard showing IMU and camera streams.
   - Optional split-screen of physical robot and software dashboard.

3. Wiring evidence.
   - Photo of computer -> U2D2 -> Power Hub -> Dynamixel bus.
   - Photo of the 12 V / 5 A adapter connected to the Power Hub.
   - Measured voltage at the Dynamixel connector after the Power Hub/regulator.
   - Photo of left/right IMU mounting.
   - Photo of left/right ESP32-S3-CAM mounting.

4. Current known-good software settings.
   - Dynamixel port used by the main demo.
   - Dynamixel baudrate used by the main demo.
   - Left and right IMU serial ports.
   - Left and right camera stream URLs.

5. Neutral calibration values.
   - Current final `perfect_positions`.
   - How those values were measured.
   - Date and hardware configuration.

## Important for Engineering Credibility

6. Mechanical assembly documentation.
   - Fusion 360 screenshot.
   - Exploded view if available.
   - Cable routing path.
   - Spool radius or cable attachment geometry.

7. Calibration experiment data.
   - Target yaw/pitch grid.
   - Actual IMU yaw/pitch/roll.
   - Motor positions.
   - Camera frames if available.

8. Representative plots.
   - Target vs actual yaw.
   - Target vs actual pitch.
   - Error over time.
   - Open-loop vs feedback comparison.

9. Foveation demo material.
   - Final Hess-matched strabismus videos have been added under
     `experiments/binocular_visualization/hess_matched_results/`.
   - Healthy-reference binocular video has been added for comparison.
   - Remaining future need: connect these videos back to the exact generation
     scripts and document the full regeneration command sequence.

## Useful but Optional

10. Purchase links for each part in the bill of materials.
11. Exact screw/fastener list.
12. Cable material, diameter, and length.
13. 3D print settings.
14. CAD source files in addition to STL exports.
15. Troubleshooting notes.
16. Known limitations.
17. A one-page system diagram image for the README.

## Please Do Not Commit

- Wi-Fi passwords.
- Private participant information.
- Raw data that should not be public.
- Local machine-specific paths unless they are clearly marked as examples.
- Very large videos unless they are intentionally part of the public repository.
