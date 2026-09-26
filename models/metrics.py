"""Evaluation: COCO mAP (bbox + mask), mask IoU, precision, recall, F1."""

import contextlib
import io

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def predict(model, dataset, device: str) -> list[dict]:
    """Run the model over a dataset. Returns per-image dicts with numpy boxes/scores/labels/binary masks."""
    model.eval()
    results = []
    for i in range(len(dataset)):
        image, target = dataset[i]
        out = model([image.to(device)])[0]
        results.append({
            "image_id": target["image_id"],
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
            "labels": out["labels"].cpu().numpy(),
            "masks": (out["masks"][:, 0] > 0.5).cpu().numpy(),
            "mask_probs": out["masks"][:, 0].cpu().numpy(),
        })
    return results


def _coco_results(preds: list[dict]) -> tuple[list, list]:
    bbox, segm = [], []
    for p in preds:
        for box, score, label, m in zip(p["boxes"], p["scores"], p["labels"], p["masks"]):
            x0, y0, x1, y1 = box.tolist()
            common = {"image_id": p["image_id"], "category_id": int(label), "score": float(score)}
            bbox.append({**common, "bbox": [x0, y0, x1 - x0, y1 - y0]})
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            segm.append({**common, "segmentation": rle})
    return bbox, segm


def coco_map(ann_file: str, preds: list[dict]) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 for boxes and masks, overall and per class."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_file)
    bbox, segm = _coco_results(preds)
    out = {}
    for iou_type, dets in (("bbox", bbox), ("segm", segm)):
        if not dets:
            out[iou_type] = {"mAP50_95": 0.0, "mAP50": 0.0, "per_class": {}}
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
            ev = COCOeval(gt, dt, iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        per_class = {}
        precision = ev.eval["precision"]  # [iou, recall, class, area, maxdet]
        for k, cat_id in enumerate(ev.params.catIds):
            name = gt.cats[cat_id]["name"]
            p_all = precision[:, :, k, 0, -1]
            p_50 = precision[0, :, k, 0, -1]
            per_class[name] = {
                "AP50_95": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else 0.0,
                "AP50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else 0.0,
            }
        out[iou_type] = {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "per_class": per_class}
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def matching_metrics(dataset, preds: list[dict], class_names: list[str],
                     score_thr: float = 0.5, iou_thr: float = 0.5) -> dict:
    """Greedy one-to-one matching per class on mask IoU -> TP/FP/FN, precision, recall, F1, mean IoU."""
    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for name in class_names}
    for i, p in enumerate(preds):
        _, target = dataset[i]
        gt_masks = target["masks"].numpy().astype(bool)
        gt_labels = target["labels"].numpy()
        keep = p["scores"] >= score_thr
        for c, name in enumerate(class_names, start=1):
            gts = [m for m, l in zip(gt_masks, gt_labels) if l == c]
            order = np.argsort(-p["scores"][keep])
            dts = [p["masks"][keep][j] for j in order if p["labels"][keep][j] == c]
            used = set()
            for d in dts:
                best, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if j not in used:
                        iou = mask_iou(d, g)
                        if iou > best:
                            best, best_j = iou, j
                if best >= iou_thr:
                    used.add(best_j)
                    stats[name]["tp"] += 1
                    stats[name]["ious"].append(best)
                else:
                    stats[name]["fp"] += 1
            stats[name]["fn"] += len(gts) - len(used)

    def summarise(tp, fp, fn, ious):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                "mean_iou": float(np.mean(ious)) if ious else 0.0}

    out = {name: summarise(s["tp"], s["fp"], s["fn"], s["ious"]) for name, s in stats.items()}
    total = {k: sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")}
    out["all"] = summarise(total["tp"], total["fp"], total["fn"], [x for s in stats.values() for x in s["ious"]])
    return out
