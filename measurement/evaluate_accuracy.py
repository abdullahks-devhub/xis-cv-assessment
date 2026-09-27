"""Accuracy validation: system measurements vs ruler ground truth.

Runs every image in measurement/images (named bookNN_K.jpg) through four variants:
  final           undistorted, mask quadrilaterals, height-corrected   (the delivered pipeline)
  edge_refined    as final, with sides snapped to image edges          (refinement ablation)
  no_height_corr  as final, card and cover assumed coplanar             (height ablation)
  distorted       as final, but on the raw image without undistortion  (calibration ablation)

The height correction uses each book's thickness (ground_truth.csv) and the height
of the support the card rested on (--card-height-mm).

Outputs (measurement/output/):
  results.csv          one row per image x variant
  summary.json         MAE / MPE / MAPE per variant, overall and per dimension
  annotated/*.jpg      final-variant visualisations

Usage:
    python -m measurement.evaluate_accuracy
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.undistort import load_params
from inference.predict import Segmenter
from measurement.measure import draw, measure

ROOT = Path(__file__).parent
VARIANTS = {  # name: (undistort, quad method, height correction)
    "final": (True, "mask", True),
    "edge_refined": (True, "edge", True),
    "no_height_corr": (True, "mask", False),
    "distorted": (False, "mask", True),
}


def load_ground_truth(path: Path) -> dict:
    with open(path) as f:
        return {int(r["book_id"]): r for r in csv.DictReader(f)}


def summarise(rows: list[dict]) -> dict:
    out = {"n_images": len(rows)}
    for dim in ("width", "height", "all"):
        dims = ("width", "height") if dim == "all" else (dim,)
        err = np.array([r[f"{d}_err_mm"] for r in rows for d in dims])
        pct = np.array([r[f"{d}_err_pct"] for r in rows for d in dims])
        out[dim] = {
            "MAE_mm": float(np.mean(np.abs(err))),
            "MPE_pct": float(np.mean(pct)),        # signed: systematic over/under-estimation
            "MAPE_pct": float(np.mean(np.abs(pct))),
            "max_abs_err_mm": float(np.max(np.abs(err))),
            "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "ground_truth.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--weights", default="models/output/best.pth")
    parser.add_argument("--card-height-mm", type=float, default=16.3,
                        help="height of the card's top surface above the table (support + card)")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    seg = Segmenter(args.weights)
    K = load_params()["K"]
    focal_px = (K[0, 0] + K[1, 1]) / 2
    (args.out / "annotated").mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.images.glob("book*_*.jpg")):
        book_id = int(path.stem[4:6])
        truth = gt[book_id]
        raw = cv2.imread(str(path))
        cache = {}
        offset = float(truth["thickness_mm"]) - args.card_height_mm
        for variant, (undistorted, method, corrected) in VARIANTS.items():
            if undistorted not in cache:
                cache[undistorted] = seg(raw, apply_undistortion=undistorted)
            image, detections = cache[undistorted]
            m = measure(image, detections, method=method, focal_px=focal_px if corrected else None,
                        cover_above_card_mm=offset if corrected else 0.0)
            row = {"image": path.name, "book_id": book_id, "description": truth["description"], "variant": variant,
                   "gt_width_mm": float(truth["width_mm"]), "gt_height_mm": float(truth["height_mm"]),
                   "width_mm": round(m.width_mm, 2), "height_mm": round(m.height_mm, 2),
                   "px_per_mm": round(m.px_per_mm, 4), "card_px_per_mm": round(m.card_px_per_mm, 4),
                   "camera_distance_mm": round(m.camera_distance_mm, 1) if m.camera_distance_mm else "",
                   "cover_above_card_mm": round(offset, 1), "card_anisotropy": round(m.scale_anisotropy, 4),
                   "book_score": round(m.book_score, 4), "card_score": round(m.card_score, 4),
                   "notes": "; ".join(m.notes)}
            for d in ("width", "height"):
                err = row[f"{d}_mm"] - row[f"gt_{d}_mm"]
                row[f"{d}_err_mm"] = round(err, 2)
                row[f"{d}_err_pct"] = round(100 * err / row[f"gt_{d}_mm"], 3)
            rows.append(row)
            if variant == "final":
                cv2.imwrite(str(args.out / "annotated" / path.name),
                            cv2.resize(draw(image, detections, m), None, fx=0.5, fy=0.5))
        f = next(r for r in rows if r["image"] == path.name and r["variant"] == "final")
        print(f"{path.name}: W {f['width_mm']:.1f} (gt {f['gt_width_mm']:.0f})  "
              f"H {f['height_mm']:.1f} (gt {f['gt_height_mm']:.0f})", flush=True)

    with open(args.out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {v: summarise([r for r in rows if r["variant"] == v]) for v in VARIANTS}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'variant':15s} {'MAE mm':>8s} {'MPE %':>8s} {'MAPE %':>8s} {'max mm':>8s}")
    for v, s in summary.items():
        a = s["all"]
        print(f"{v:15s} {a['MAE_mm']:8.2f} {a['MPE_pct']:8.2f} {a['MAPE_pct']:8.2f} {a['max_abs_err_mm']:8.2f}")


if __name__ == "__main__":
    main()
