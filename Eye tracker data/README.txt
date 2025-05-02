# Human Eye Gaze Trajectory Dataset (Processed)

This repository contains processed human eye gaze trajectory data collected using an eye tracker during standardized video watching tasks. The data has been processed to remove noise and interference artifacts, and is used to drive a robotic eye platform to replicate human eye movements.

This dataset is associated with the Scientific Reports manuscript:

**"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"**

## Dataset Overview

- **Format**: MATLAB `.mat` file
- **Content**:
  - Smoothed and cleaned eye gaze angles (Yaw and Pitch for left and right eyes)
  - Total number of frames: 500 frames and 150 frames(two types) 
  - Sampling frequency: 30 Hz

## Data Description

The file contains the following variables:

- **GazeAngleSmoothed_cell**: A matrix where each row corresponds to a sampled timestamp and columns represent the following:
    - Column 1: Right Eye Yaw (degrees)
    - Column 2: Right Eye Pitch (degrees)
    - Column 3: Left Eye Yaw (degrees)
    - Column 4: Left Eye Pitch (degrees)

All angles are in degrees and have been smoothed and processed to remove interference.

## Data Collection Method

- **Device**: Viewpoint Eye Tracker®
- **Task**: Participants were instructed to watch a standardized video and track a moving green dot target on the screen.
- **Environment**: Monitors were positioned at a fixed distance with the eye tracker mounted below.
- **Processing**: Raw gaze data was smoothed and filtered to remove sudden jumps, blinks, and other noise artifacts.

## Usage

The data can be used for:

- Driving robotic eye platforms to mimic human eye movements
- Validating gaze control algorithms
- Studying human eye movement characteristics during visual tasks

## License

This dataset is released under the MIT License for academic research purposes. Please contact the authors for commercial use or if used in a publication.

## Citation

If you use this dataset, please cite:

> Yidi Huang, Qi Wei, Joseph L. Demer, and Ningshi Yao. A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis. Scientific Reports, 2025.

