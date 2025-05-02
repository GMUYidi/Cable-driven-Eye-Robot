
Robotic Eyes Video Stitching and Processing (Strabismus Case)

This Jupyter notebook performs video processing and fusion tasks for simulating binocular vision in a strabismus patient using the robotic eye platform recordings. It was used in the study:

"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"

Overview

The notebook processes left and right eye videos captured from the robotic eye platform and generates a fused binocular vision video representing how a strabismus patient would perceive the visual scene.

Key functions of this notebook include:
- Loading synchronized left and right eye videos
- Applying alignment and stitching techniques to create a single fused video
- Simulating binocular visual perception in the presence of strabismus (image doubling, misalignment)
- Saving the processed video for analysis and publication

Files

📒 Robotic eyes Video stitching and process-mid-STR.ipynb
- Jupyter Notebook with all processing steps
- Includes code for reading video files, synchronizing, stitching, and saving the result
- Parameters can be adjusted for different patients or scenarios

Usage

1. Install required Python packages:
   - OpenCV
   - NumPy
   - Other dependencies as specified in the notebook

2. Place the left and right eye video files in the designated directory (modify paths in the notebook accordingly).

3. Run the notebook step-by-step to:
   - Load and process left/right eye videos
   - Perform video stitching
   - Export the fused binocular vision video

Purpose

- To create realistic simulations of binocular vision impairment due to strabismus
- To visualize differences between healthy and strabismic binocular fusion
- To generate supplementary materials for publication and presentations

License

The notebook is released for academic and non-commercial research purposes only.

Citation

If you use this notebook or generated videos for your research or publications, please cite:

Yidi Huang, Qi Wei, Joseph L. Demer, and Ningshi Yao. A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis. Scientific Reports, 2025.
