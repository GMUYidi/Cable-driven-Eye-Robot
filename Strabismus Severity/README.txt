
Severity Values and Visualization for Healthy Participants and Strabismus Patient

This folder contains data and MATLAB code for visualizing the severity values of healthy participants along with a strabismus patient's severity score for comparative analysis.

Overview

The purpose of this dataset and visualization is to illustrate the severity distribution among healthy participants and to highlight the difference compared to a strabismus patient. This is part of the study described in:

"A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis"

Files included

📄 Participants Severity values.txt
- Contains severity values for 25 healthy participants
- Each line corresponds to the severity value of one participant
- The data is used for statistical analysis and visualization in the box plot

📄 Box_plot_severity.m
- MATLAB script to generate a box plot for healthy participants' severity values
- Highlights:
  - Strabismus patient severity score (hardcoded in the script) shown as a dashed red line
  - Maximum and minimum values marked with green and blue stars
  - Variance of the healthy participants' severity values displayed
  - Legend, annotations, and formatted axis labels for publication-quality plots

Usage

1. Open Box_plot_severity.m in MATLAB.
2. Ensure Participants Severity values.txt is loaded into severity_values (or use the hardcoded version).
3. Run the script.
4. The following will be generated:
    - Box plot showing healthy participants' severity distribution
    - Highlighted strabismus patient severity score
    - Statistical annotations including maximum, minimum, and variance

Purpose

- To quantitatively compare the severity between healthy participants and a strabismus patient
- To visualize the distribution and variability of severity values in healthy individuals
- To be used as part of the Results or Supplementary Materials in the corresponding scientific paper

License

The dataset and code are released for academic research use only under the MIT License.

Citation

If you use these files for your research or publications, please cite:

Yidi Huang, Qi Wei, Joseph L. Demer, and Ningshi Yao. A Novel Cable-driven Eye Robot For Human Vision Visualization and Strabismus Diagnosis. Scientific Reports, 2025.
