# Compression Noise Effect on Vision Algorithms

This repo contains a reproducible experiment pipeline for studying how video
compression affects computer vision output quality.

The pipeline:

1. Takes high-quality input videos.
2. Generates compressed versions with FFmpeg.
3. Runs one or more vision algorithms on the original and compressed videos.
4. Compares compressed outputs against the original high-quality baseline.
5. Writes metrics and plots for analysis.

## Setup

Install system dependencies:

```bash
brew install ffmpeg
```

Create a Python environment and install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Add Videos

Place original videos under `data/raw/`, grouped by category:

```text
data/raw/
  single_object/
    video1.mp4
  multi_object/
    video2.mp4
```

Then copy `project_config.example.json` to `project_config.json` and edit the
video list if needed.

## Run the Full Experiment

```bash
python -m compression_noise.pipeline all --config project_config.example.json
```

Outputs are written to:

- `data/compressed/`: FFmpeg-compressed videos
- `outputs/detections/`: per-frame algorithm detections
- `outputs/metrics/summary.csv`: evaluation metrics
- `outputs/plots/`: performance graphs

## Algorithms

Two modular algorithms are included:

- `motion`: moving-object detection using background subtraction and contour boxes
- `edges`: edge/contour detection using Canny edges

The `motion` algorithm is the recommended primary algorithm for videos with
moving objects. The `edges` algorithm is useful as a second comparison method,
but it can be noisier on complex scenes.

## Main Metric

Each compressed video is scored against the original baseline on a 0 to 1 scale.
The score combines:

- Correct detections
- Missed detections
- Extra detections
- Bounding-box localization similarity using IoU

A score of `1.0` means the compressed output perfectly matches the original
algorithm output. A score near `0.0` means the compressed output differs strongly
or fails to detect comparable objects.
