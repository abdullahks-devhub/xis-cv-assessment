"""Inference: raw capture -> undistortion -> Mask R-CNN -> annotated result.

Accepts a single image or a folder. Raw 4160x3120 captures are resized to the
working resolution and undistorted with the stored calibration before inference.

Outputs per image (in --out):
  <name>.jpg    annotated image (mask overlay, class, confidence)
  <name>.json   detections: class, score, box, mask polygon (working-resolution pixels)

Usage:
    python -m inference.predict --input path/to/photo.jpg
    python -m inference.predict --input measurement/images --out inference/output
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from calibration.undistort import load_params, undistort
from models.common import load_config, load_trained, pick_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COLORS = {"book": (255, 0, 255), "card": (0, 200, 255)}  # BGR


class Segmenter:
    """Loads the calibration and trained model once; call on raw BGR images."""

    def __init__(self, weights: str | Path = "models/output/best.pth", config: str | Path = "models/config.yaml",
                 calibration: str | Path | None = None, score_threshold: float | None = None):
        self.cfg = load_config(config)
        self.classes = self.cfg["data"]["classes"]
        self.device = pick_device()
        self.model = load_trained(self.cfg, weights, self.device)
        self.params = load_params(calibration) if calibration else load_params()
        self.score_threshold = score_threshold if score_threshold is not None else self.cfg["eval"]["score_threshold"]

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, apply_undistortion: bool = True) -> tuple[np.ndarray, list[dict]]:
        """Returns (working-resolution image the model saw, detections sorted by score)."""
        if apply_undistortion:
            image = undistort(image_bgr, self.params)
        else:  # for the distortion ablation only: same resize, no correction
            h, w = image_bgr.shape[:2]
            image = image_bgr if (w, h) == self.params["image_size"] else cv2.resize(
                image_bgr, self.params["image_size"], interpolation=cv2.INTER_AREA)
        tensor = to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(self.device)
        out = self.model([tensor])[0]
        detections = []
        for box, score, label, mask in zip(out["boxes"], out["scores"], out["labels"], out["masks"]):
            if score < self.score_threshold:
                continue
            detections.append({
                "class": self.classes[int(label) - 1],
                "score": float(score),
                "box": [float(v) for v in box],
                "mask": (mask[0] > 0.5).cpu().numpy(),
            })
        return image, sorted(detections, key=lambda d: -d["score"])


def mask_polygon(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    return max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()


def annotate(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    overlay, canvas = image.copy(), image.copy()
    for d in detections:
        color = COLORS.get(d["class"], (0, 255, 0))
        overlay[d["mask"]] = color
        contours, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 3)
        x0, y0 = int(d["box"][0]), int(d["box"][1])
        cv2.putText(canvas, f"{d['class']} {d['score']:.2f}", (x0, max(y0 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="image file or folder")
    parser.add_argument("--out", type=Path, default=Path("inference/output"))
    parser.add_argument("--weights", default="models/output/best.pth")
    parser.add_argument("--config", default="models/config.yaml")
    args = parser.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    seg = Segmenter(args.weights, args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    for p in paths:
        image, detections = seg(cv2.imread(str(p)))
        cv2.imwrite(str(args.out / f"{p.stem}.jpg"), annotate(image, detections))
        record = [{"class": d["class"], "score": round(d["score"], 4), "box": [round(v, 1) for v in d["box"]],
                   "polygon": mask_polygon(d["mask"])} for d in detections]
        (args.out / f"{p.stem}.json").write_text(json.dumps({"image": p.name, "detections": record}, indent=1))
        summary = ", ".join(f"{d['class']} {d['score']:.2f}" for d in detections) or "nothing detected"
        print(f"{p.name}: {summary}")
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
