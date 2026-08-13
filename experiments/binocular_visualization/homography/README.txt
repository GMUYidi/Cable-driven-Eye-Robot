
Robotic Eyes Homography Matrices Calculation and Processing

This Jupyter notebook performs homography estimation for binocular vision processing using data captured from the robotic eye platform. It is used in the study:

"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"

Overview

The notebook calculates the homographic matrices for the left and right eyes to enable image alignment and binocular vision fusion.

Key functions of this notebook include:
- Loading left and right eye gaze and video data
- Estimating homography matrices for each time frame or sample
- Visualizing homography matrix components and trends
- Saving homography matrices for use in later vision simulation steps

Files

📒 Robotic eyes Homographic matrics.ipynb
- Jupyter Notebook with all processing steps
- Supports reading experimental eye gaze or image data
- Computes and saves homography matrices for left and right eyes
- Plots homography matrix entries across time for validation and analysis

Usage

1. Install required Python packages:
   - OpenCV
   - NumPy
   - Matplotlib
   - (Others as specified in the notebook)

2. Place the relevant gaze data or left/right eye images in the designated input path (modify paths in the notebook accordingly).

3. Run the notebook step-by-step to:
   - Calculate homography matrices
   - Visualize matrix entries
   - Save homography matrices for future use (e.g., video stitching and binocular vision simulation)

Purpose

- To enable binocular image registration and alignment based on eye gaze data
- To provide accurate homographic transformations for left and right eye fusion
- To support visual simulation of normal and strabismic binocular vision

License

The notebook is released for academic and non-commercial research purposes only.

Citation

If you use this notebook or generated homography matrices for your research or publications, please cite:

Yidi Huang, Qi Wei, Joseph L. Demer, and Ningshi Yao. A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis. Scientific Reports, 2025.
