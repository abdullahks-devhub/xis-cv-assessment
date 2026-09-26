"""Dataset, augmentation and model construction shared by training, evaluation and inference."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_tensor


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def polygons_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for poly in segmentation:
        pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


class CocoSegmentation(torch.utils.data.Dataset):
    """COCO polygon annotations -> (image tensor, Mask R-CNN target dict)."""

    def __init__(self, ann_file: str | Path, image_dir: str | Path, augment: dict | None = None):
        coco = json.loads(Path(ann_file).read_text())
        self.image_dir = Path(image_dir)
        self.images = sorted(coco["images"], key=lambda i: i["file_name"])
        self.anns = {i["id"]: [] for i in self.images}
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.augment = augment
        cj = (augment or {}).get("color_jitter")
        self.jitter = ColorJitter(**cj) if cj else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(self.image_dir / info["file_name"])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        anns = self.anns[info["id"]]
        masks = np.stack([polygons_to_mask(a["segmentation"], h, w) for a in anns]) if anns \
            else np.zeros((0, h, w), np.uint8)
        labels = np.array([a["category_id"] for a in anns], np.int64)

        if self.augment:
            if random.random() < self.augment.get("hflip", 0):
                img, masks = img[:, ::-1], masks[:, :, ::-1]
            if random.random() < self.augment.get("vflip", 0):
                img, masks = img[::-1], masks[:, ::-1]
            if random.random() < self.augment.get("rot90", 0):
                k = random.choice([1, 3])
                img, masks = np.rot90(img, k), np.rot90(masks, k, axes=(1, 2))

        image = to_tensor(np.ascontiguousarray(img))
        if self.jitter is not None:
            image = self.jitter(image)

        masks = torch.from_numpy(np.ascontiguousarray(masks))
        boxes = torchvision.ops.masks_to_boxes(masks).float() if len(masks) else torch.zeros((0, 4))
        target = {
            "boxes": boxes,
            "labels": torch.from_numpy(labels),
            "masks": masks,
            "image_id": info["id"],
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(anns), dtype=torch.int64),
        }
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(cfg: dict, pretrained: bool = True) -> torch.nn.Module:
    """COCO-pretrained Mask R-CNN with box and mask heads replaced for our classes."""
    num_classes = len(cfg["data"]["classes"]) + 1
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights, min_size=cfg["model"]["min_size"], max_size=cfg["model"]["max_size"],
        box_detections_per_img=cfg["model"].get("detections_per_img", 20),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def load_trained(cfg: dict, weights_path: str | Path, device: str) -> torch.nn.Module:
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def pick_device() -> str:
    # Apple MPS is skipped: torchvision detection ops (e.g. nms) are not implemented for it.
    return "cuda" if torch.cuda.is_available() else "cpu"
