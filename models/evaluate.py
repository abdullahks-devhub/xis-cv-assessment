"""Evaluate trained weights on a held-out split and visualise predictions.

Outputs:
  models/output/<split>_metrics.json          mAP (bbox + mask), IoU, precision, recall, F1
  inference/<split>_predictions/*.jpg         prediction overlays (+ grid.jpg)

Usage:
    python -m models.evaluate --weights models/output/best.pth
    python -m models.evaluate --weights models/output/best.pth --split val
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from models.common import CocoSegmentation, load_config, load_trained, pick_device
from models.metrics import coco_map, matching_metrics, predict

COLORS = {1: (255, 0, 255), 2: (0, 200, 255)}  # BGR: book magenta, card orange


def draw_predictions(image_rgb: np.ndarray, pred: dict, class_names: list[str], score_thr: float) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    for box, score, label, mask in zip(pred["boxes"], pred["scores"], pred["labels"], pred["masks"]):
        if score < score_thr:
            continue
        color = COLORS.get(int(label), (0, 255, 0))
        overlay[mask] = color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 4)
        x0, y0 = int(box[0]), int(box[1])
        cv2.putText(img, f"{class_names[int(label) - 1]} {score:.2f}", (x0, max(y0 - 12, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    return cv2.addWeighted(overlay, 0.35, img, 0.65, 0)


def save_grid(paths: list[Path], out: Path, cols: int = 5, width: int = 520) -> None:
    tiles = [cv2.resize(cv2.imread(str(p)), (width, int(width * 0.75))) for p in paths]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="models/config.yaml")
    parser.add_argument("--weights", default="models/output/best.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    classes = cfg["data"]["classes"]
    thr = cfg["eval"]["score_threshold"]
    ann_file = cfg["data"][args.split]

    dataset = CocoSegmentation(ann_file, cfg["data"]["images"])
    model = load_trained(cfg, args.weights, device)
    preds = predict(model, dataset, device)

    results = {
        "split": args.split,
        "images": len(dataset),
        "weights": str(args.weights),
        "coco": coco_map(ann_file, preds),
        "matching": matching_metrics(dataset, preds, classes, thr, cfg["eval"]["iou_threshold"]),
        "score_threshold": thr,
        "iou_threshold": cfg["eval"]["iou_threshold"],
    }
    out_json = Path(cfg["output_dir"]) / f"{args.split}_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    vis_dir = Path("inference") / f"{args.split}_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for info, pred in zip(dataset.images, preds):
        rgb = cv2.cvtColor(cv2.imread(str(dataset.image_dir / info["file_name"])), cv2.COLOR_BGR2RGB)
        path = vis_dir / info["file_name"]
        cv2.imwrite(str(path), cv2.resize(draw_predictions(rgb, pred, classes, thr), None, fx=0.5, fy=0.5))
        saved.append(path)
    save_grid(saved, vis_dir / "grid.jpg")

    c, m = results["coco"], results["matching"]["all"]
    print(f"[{args.split}] mask mAP@0.5 {c['segm']['mAP50']:.3f} | mask mAP@0.5:0.95 {c['segm']['mAP50_95']:.3f} | "
          f"box mAP@0.5 {c['bbox']['mAP50']:.3f} | box mAP@0.5:0.95 {c['bbox']['mAP50_95']:.3f}")
    print(f"[{args.split}] precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['f1']:.3f} | "
          f"mean mask IoU {m['mean_iou']:.3f}")
    print(f"Saved {out_json} and overlays in {vis_dir}")


if __name__ == "__main__":
    main()
