# XIS CV Assessment — Calibrated Object Measurement

An end-to-end computer vision pipeline that segments a custom object and measures its real-world width and height in millimetres from a single image.

**Pipeline:** camera calibration → undistortion → instance segmentation → pixel-to-mm conversion via a reference marker.

> Work in progress. Sections are filled in as each stage is completed.

## Repository structure

```
calibration/   # Calibration images, scripts and camera parameters
dataset/       # Raw images and labels (train/val/test splits)
models/        # Training configs and saved weights
inference/     # Inference scripts and demo outputs
measurement/   # Pixel-to-mm pipeline and accuracy report
docs/          # All documentation files
```

## Quick start

See [docs/SETUP.md](docs/SETUP.md) (coming soon).
