"""Load stored camera intrinsics and undistort images.

Shared by the dataset preparation, inference and measurement stages so every
image passes through exactly the same correction.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_PARAMS = Path(__file__).parent / "camera_params.yaml"


def load_params(path: Path = DEFAULT_PARAMS) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        "K": np.array(data["camera_matrix"], dtype=np.float64),
        "dist": np.array(data["dist_coeffs"], dtype=np.float64),
        "image_size": tuple(data["image_size"]),  # (width, height)
    }


def undistort(image: np.ndarray, params: dict) -> np.ndarray:
    """Remove radial/tangential distortion, keeping the original camera matrix and image size."""
    h, w = image.shape[:2]
    if (w, h) != params["image_size"]:
        raise ValueError(
            f"Image is {w}x{h} but calibration was done at {params['image_size'][0]}x{params['image_size'][1]}. "
            "Intrinsics are resolution-specific — capture at the calibrated resolution."
        )
    return cv2.undistort(image, params["K"], params["dist"])
