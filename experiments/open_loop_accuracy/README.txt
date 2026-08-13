
Accuracy Test of Pure Open-loop Control Under Inverse Kinematics

This folder contains experimental data and analysis code used to evaluate the accuracy of a pure open-loop control strategy based on inverse kinematics for robotic eye movement.

Overview

The test aims to verify how precisely the robotic eye can follow desired gaze angles (yaw and pitch) when driven purely by inverse kinematics calculations, without real-time feedback correction.

Files included

📄 data_test_open_loop.txt
- Recorded dataset of robotic eye movement tests
- Contains measured vs. target yaw, pitch, and roll angles during open-loop operation
- Columns:
  - Column 1: Actual yaw angle (degrees)
  - Column 2: Actual pitch angle (degrees)
  - Column 3: Actual roll angle (degrees)
  - Column 4: Target yaw angle (degrees)
  - Column 5: Target pitch angle (degrees)
  - Column 6: Target roll angle (degrees)

📄 bilinear_interpolation_eachangle.m
- MATLAB script to process the above dataset
- Generates 2D contour heatmaps visualizing:
  - Yaw angle errors
  - Pitch angle errors
  - Roll angle errors
- The heatmaps provide intuitive representations of open-loop control performance across different gaze angles.

How to use

1. Open bilinear_interpolation_eachangle.m in MATLAB.
2. Ensure data_test_open_loop.txt is in the same directory or adjust the path.
3. Run the script.
4. The following plots will be generated and displayed:
    - Yaw error heatmap
    - Pitch error heatmap
    - Roll error heatmap

Purpose

These files were used in the study to:
- Evaluate the accuracy and limitations of pure inverse kinematics-based open-loop control.
- Provide quantitative and visualized results for robotic eye design and control algorithm validation.

License

The dataset and code are released for academic research use under the MIT License.

Citation

If you use these files for your research or publications, please cite:

Yidi Huang, Qi Wei, Joseph L. Demer, and Ningshi Yao. A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis. Scientific Reports, 2025.
