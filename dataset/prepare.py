"""Undistort raw captures with the stored calibration.

Every raw 4160x3120 capture is resized to the 2080x1560 working resolution and
undistorted with calibration/camera_params.yaml. The undistorted images are the
ones that get labelled and trained on.

Usage:
    python -m dataset.prepare
    python -m dataset.prepare --src measurement/images --dst measurement/undistorted
"""

import argparse
from pathlib import Path

import cv2

from calibration.undistort import load_params, undistort

ROOT = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "raw")
    parser.add_argument("--dst", type=Path, default=ROOT / "undistorted")
    parser.add_argument("--exclude", nargs="*", default=[], help="file names to skip")
    args = parser.parse_args()

    params = load_params()
    args.dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.src.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.name not in args.exclude)

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip {p.name}: unreadable")
            continue
        cv2.imwrite(str(args.dst / p.name), undistort(img, params), [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Undistorted {len(paths)} images -> {args.dst}")


if __name__ == "__main__":
    main()
