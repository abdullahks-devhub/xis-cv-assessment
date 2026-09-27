# Setup and Run Guide

## 1. Requirements

- Python **3.11** (tested; 3.10–3.12 should work)
- About 1.5 GB of disk space for the repository (images included) plus 184 MB for the model weights
- For **training**: an NVIDIA GPU. Google Colab's free T4 is enough, and a notebook is provided.
- For **inference and measurement**: CPU is fine, at about 5–10 s per image. Apple MPS is not used, because torchvision's detection ops (`nms`) are not implemented for it.

No Docker image is provided. The pipeline has only pip dependencies, listed in `requirements.txt`.

## 2. Installation

```bash
git clone https://github.com/abdullahks-devhub/xis-cv-assessment.git
cd xis-cv-assessment

conda create -n xis python=3.11 -y
conda activate xis
pip install -r requirements.txt
```

**Intel Macs:** PyTorch stopped publishing x86-64 macOS wheels after 2.2.2, and that build needs NumPy 1.x:

```bash
pip install "numpy<2" "opencv-contrib-python<4.11"
```

## 3. Model weights

`best.pth` (184 MB) is too large for git. Download it from the repository's **Releases** page and place it at `models/output/best.pth`:

```bash
mkdir -p models/output
curl -L -o models/output/best.pth \
  https://github.com/abdullahks-devhub/xis-cv-assessment/releases/download/v1.0/best.pth
```

Alternatively, retrain it (Section 5.3).

## 4. Capturing new images

Measurements are only valid for images from the **calibrated camera with the same settings**:

| Setting | Value |
|---|---|
| Device | Huawei Y7 Prime 2019, main rear camera (Open Camera app) |
| Resolution | 4160 × 3120 |
| Orientation | Landscape (locked) |
| Mode | Standard; HDR, auto-level and flash off |
| Distance | About 25–40 cm, camera pointing straight down |
| Focus | Tap on the book |

Place the card (63 × 88 mm) **flat beside the book**, 2–5 cm away, fully visible. Ideally, raise the card to the height of the book's top cover. Otherwise, pass both heights to the demo (Section 5.5).

Transfer the photos without compression (USB, LocalSend or Google Drive originals, **not** WhatsApp).

A different camera needs recalibration (Section 5.1).

## 5. Running each stage

All commands are run from the repository root. Each script prints its full usage with `--help`.

### 5.1 Camera calibration

```bash
python -m calibration.make_board                                  # generate the checkerboard (screen PNG / A4 PDF)
python -m calibration.calibrate --square-mm 23.7 --scale 0.5      # images in calibration/images/
```

This writes `calibration/camera_params.yaml` and visualisations to `calibration/output/`. See [CALIBRATION_REPORT.md](CALIBRATION_REPORT.md).

### 5.2 Dataset preparation

```bash
python -m dataset.prepare          # dataset/raw -> dataset/undistorted (undistort + resize)
python -m dataset.split            # dataset/annotations -> dataset/splits (70/20/10, seed 42)
```

### 5.3 Training (GPU)

**Colab:** open [`models/train_colab.ipynb`](../models/train_colab.ipynb) in Google Colab, choose **Runtime → Change runtime type → T4 GPU**, then **Run all**. It downloads `training_results.zip`. Unzip it at the repository root.

**Local GPU:**

```bash
python -m models.train --config models/config.yaml
python -m models.evaluate --weights models/output/best.pth --split test
```

Smoke test (CPU, a few minutes): `python -m models.train --epochs 1 --limit 4`

### 5.4 Inference (segmentation only)

```bash
python -m inference.predict --input path/to/photo.jpg --out inference/output
python -m inference.predict --input measurement/images            # a whole folder
```

### 5.5 Measurement demo (single image → mm)

```bash
python -m measurement.demo --image measurement/images/book01_2.jpg \
    --book-thickness-mm 12 --card-height-mm 16.3
```

```
Width:      176.8 mm
Height:     238.7 mm
Confidence: 0.996
Overlay:    measurement/output/demo/book01_2_measured.jpg
```

If the card is raised exactly to the cover height, omit both height arguments.

### 5.6 Accuracy validation

```bash
python -m measurement.evaluate_accuracy --card-height-mm 16.3
```

This writes `measurement/output/results.csv`, `summary.json` and annotated images. See [MEASUREMENT_REPORT.md](MEASUREMENT_REPORT.md).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Image is WxH; expected a 4160x3120 capture...` | The photo was resized, for example by a messaging app, or taken at another resolution. Recapture at 4160 × 3120. |
| `need a book and a card` | One of the objects was not detected with confidence ≥ 0.5. Make sure both are fully in frame, with no strong glare on the card. |
| `NotImplementedError ... 'torchvision::nms' ... MPS` | Old code path. The current `pick_device()` uses CUDA or CPU only. |
| `Numpy is not available` (Intel Mac) | Install `numpy<2` (Section 2). |
| Colab run stops with `^C` and no traceback | The process ran out of RAM. Fixed in the current code, which RLE-encodes predicted masks. |
| CVAT polygon tool does nothing | Use Chrome. CVAT does not fully support Safari. |
