"""Fine-tune Mask R-CNN on the book/card dataset.

Logs train and val loss per epoch, evaluates val mAP each epoch, keeps the best
weights (by val mask mAP@0.5:0.95) and saves loss/metric curves.

Outputs (in cfg.output_dir):
  best.pth, last.pth       model weights
  history.csv              per-epoch losses, lr and val metrics
  curves.png               loss and mAP curves
  config_used.yaml         exact configuration of the run

Usage:
    python -m models.train --config models/config.yaml
    python -m models.train --config models/config.yaml --epochs 1 --limit 4   # smoke test
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models.common import CocoSegmentation, build_model, collate, load_config, pick_device
from models.metrics import coco_map, predict


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, log_every: int = 0) -> dict:
    """One pass over loader. With an optimizer: trains. Without: computes val loss (no grad)."""
    model.train()  # torchvision detection models only return losses in train mode; BN is frozen
    totals, n = {}, 0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        with torch.set_grad_enabled(optimizer is not None):
            losses = model(images, targets)
            loss = sum(losses.values())
        if optimizer is not None:
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite loss: {losses}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        n += 1
        if log_every and n % log_every == 0:
            print(f"    iter {n}/{len(loader)} loss {loss.item():.3f}", flush=True)
    return {k: v / max(n, 1) for k, v in totals.items()}


def plot_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Total loss")
    for key, label in (("loss_classifier", "classifier"), ("loss_box_reg", "box"), ("loss_mask", "mask")):
        axes[1].plot(epochs, [h[f"train_{key}"] for h in history], label=f"train {label}")
        axes[1].plot(epochs, [h[f"val_{key}"] for h in history], "--", label=f"val {label}")
    axes[1].set_title("Loss components")
    for key, label in (("val_segm_mAP50", "mask mAP@0.5"), ("val_segm_mAP50_95", "mask mAP@0.5:0.95"),
                       ("val_bbox_mAP50_95", "box mAP@0.5:0.95")):
        axes[2].plot(epochs, [h[key] for h in history], label=label)
    axes[2].set_title("Validation mAP")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="models/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]
    if args.epochs is not None:
        tc["epochs"] = args.epochs
    seed_everything(tc["seed"])
    device = pick_device()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d = cfg["data"]
    train_ds = CocoSegmentation(d["train"], d["images"], augment=cfg["augmentation"])
    val_ds = CocoSegmentation(d["val"], d["images"])
    val_loss_ds = CocoSegmentation(d["val"], d["images"])
    if args.limit:
        for ds in (train_ds, val_ds, val_loss_ds):
            ds.images = ds.images[:args.limit]

    loader_kw = dict(collate_fn=collate, num_workers=tc["num_workers"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_loss_ds, batch_size=tc["batch_size"], shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    iters_per_epoch = len(train_loader)

    def lr_factor(it: int) -> float:
        # linear warmup over the first iterations, then step decay at the configured epochs
        warm = min(1.0, (it + 1) / tc["warmup_iters"])
        decays = sum(it >= m * iters_per_epoch for m in tc["lr_milestones"])
        return warm * tc["lr_gamma"] ** decays

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    print(f"Device: {device} | train {len(train_ds)} | val {len(val_ds)} | epochs {tc['epochs']}", flush=True)
    history, best = [], -1.0
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, device, optimizer, scheduler,
                                log_every=10 if epoch == 1 else 0)
        val_stats = run_epoch(model, val_loader, device)
        metrics = coco_map(d["val"], predict(model, val_ds, device))

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": train_stats["loss"], "val_loss": val_stats["loss"]}
        for k in ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"):
            row[f"train_{k}"] = train_stats[k]
            row[f"val_{k}"] = val_stats[k]
        for t in ("bbox", "segm"):
            row[f"val_{t}_mAP50"] = metrics[t]["mAP50"]
            row[f"val_{t}_mAP50_95"] = metrics[t]["mAP50_95"]
        history.append(row)

        score = row["val_segm_mAP50_95"]
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "best.pth")
        torch.save(model.state_dict(), out_dir / "last.pth")

        print(f"epoch {epoch:2d} | train {row['train_loss']:.3f} | val {row['val_loss']:.3f} | "
              f"mask mAP50 {row['val_segm_mAP50']:.3f} mAP50:95 {score:.3f} | "
              f"box mAP50:95 {row['val_bbox_mAP50_95']:.3f} | {time.time() - t0:.0f}s", flush=True)

        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        plot_curves(history, out_dir / "curves.png")

    print(f"Best val mask mAP@0.5:0.95 = {best:.3f} -> {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
