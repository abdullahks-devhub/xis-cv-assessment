"""Load stored camera intrinsics and undistort images.

Shared by the dataset preparation, inference and measurement stages so every
image passes through exactly the same correction.

Calibration is performed at a working resolution (capture size x working_scale).
Full-resolution captures are resized to that size before undistortion so the
intrinsics always match the pixels they are applied to.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_PARAMS = Path(__file__).parent / "camera_params.yaml"


def to_working_size(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale with area interpolation (averages sensor noise instead of aliasing it)."""
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def load_params(path: Path = DEFAULT_PARAMS) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        "K": np.array(data["camera_matrix"], dtype=np.float64),
        "dist": np.array(data["dist_coeffs"], dtype=np.float64),
        "image_size": tuple(data["image_size"]),  # working (width, height)
        "capture_size": tuple(data.get("capture_size", data["image_size"])),
        "scale": float(data.get("working_scale", 1.0)),
    }


def undistort(image: np.ndarray, params: dict) -> np.ndarray:
    """Resize a capture to the working resolution (if needed) and remove radial/tangential distortion."""
    h, w = image.shape[:2]
    if (w, h) == params["capture_size"]:
        image = to_working_size(image, params["scale"])
    elif (w, h) != params["image_size"]:
        raise ValueError(
            f"Image is {w}x{h}; expected a {params['capture_size'][0]}x{params['capture_size'][1]} capture "
            f"or a {params['image_size'][0]}x{params['image_size'][1]} working image. "
            "Intrinsics are resolution-specific — capture at the calibrated resolution."
        )
    return cv2.undistort(image, params["K"], params["dist"])
