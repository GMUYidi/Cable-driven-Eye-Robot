# Hess-Matched Binocular Visualization Results

This folder contains representative binocular visualization results for
Hess-chart-matched strabismus simulations and a healthy binocular reference.

The goal of this experiment is to visualize how the scene seen by the robotic
eyes changes when the binocular view is warped according to Hess-chart-derived
strabismus motion, then compare it against the corresponding healthy binocular
view.

## Included Results

```text
hess_matched_results/
  RESULTS_MANIFEST.csv
  FILE_MANIFEST.csv
  healthy_reference/
    stitched_frames_flipped_cropped_60fps.mp4
    flip_crop_metadata.json
    flip_crop_report.csv
    ffmpeg_frames_health_second.txt
    matrices/
  strabismus/
    Q3/
    Q11/
    Q13/
    ...
    Q48/
```

Each strabismus subject folder includes:

- `stitched_frames_flipped_cropped_60fps_with_gaze_<subject>_lead0p5s.mp4`
- `hess_match_metadata.json`
- `hess_match_report.csv`
- `flip_crop_metadata.json`
- `flip_crop_report.csv`
- `matrices/*.npz`

The healthy reference folder includes:

- `stitched_frames_flipped_cropped_60fps.mp4`
- `flip_crop_metadata.json`
- `flip_crop_report.csv`
- `ffmpeg_frames_health_second.txt`
- `matrices/*.npz`

## Subject Set

The strabismus simulation set currently includes:

```text
Q3, Q11, Q13, Q14, Q16, Q21, Q22, Q23, Q27, Q29,
Q33, Q37, Q40, Q44, Q45, Q47, Q48
```

The `Q*` identifiers are anonymized Hess-chart subject IDs used to keep the
results traceable without exposing participant names.

## Manifests

`RESULTS_MANIFEST.csv` summarizes each result folder, source folder, final
video filename, video size, and data-file count.

`FILE_MANIFEST.csv` lists every copied result/data file with its relative path,
size, role, and SHA-256 checksum. This makes the copied results auditable after
uploading to GitHub.

## Generation Pipeline

The scripts used to generate these results are archived in:

```text
experiments/binocular_visualization/pipeline/
```

That pipeline contains the healthy-baseline capture/replay scripts, Hess-chart
interpolation tools, and video stitching/homography scripts used to produce the
current result videos.

## Excluded Intermediate Files

The original local result folders also contain thousands of intermediate JPEG
frames, debug frames, and temporary visual checks. Those files are intentionally
excluded from this repository because they are large and can be regenerated from
the video-generation pipeline.

The committed result package keeps the final videos and the small data files
needed to interpret or verify each result.
