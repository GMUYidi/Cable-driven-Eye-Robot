# Calibration

This folder stores reusable calibration artifacts used by the robotic-eye
software. Calibration files are separated by type so camera, motor, IMU, and
pixel-to-gaze calibration do not become mixed together.

Current structure:

```text
calibration/
  camera_intrinsics/
    camera_parameter_left_or_default.npz
    camera_parameter_right.npz
```

Recommended future files:

```text
calibration/
  motor_neutral/
  eye_geometry/
  imu_zero_offsets/
  pixel_to_gaze/
```

Every calibration file should include:

- Date.
- Hardware version.
- Operator.
- Sensor or motor IDs.
- Method used to collect the calibration.
- Valid motion range.
