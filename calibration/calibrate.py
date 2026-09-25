"""Intrinsic camera calibration from checkerboard images.

Detects inner corners, runs cv2.calibrateCamera, optionally drops outlier
images, and writes:
  calibration/camera_params.yaml        intrinsics, distortion, reprojection error
  calibration/output/corners/*.jpg      detected corners per image
  calibration/output/coverage.png       where corners fell across the frame
  calibration/output/undistort_*.jpg    before/after comparison

Usage:
    python -m calibration.calibrate --square-mm 22.4
    python -m calibration.calibrate --square-mm 22.4 --max-error 1.0
"""

import argparse
from datetime import date
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """3D corner coordinates on the board plane (Z = 0), in mm."""
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return grid


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Sector-based detector (sub-pixel accurate); falls back to the classic detector + cornerSubPix."""
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if found:
        return corners.reshape(-1, 1, 2).astype(np.float32)
    found, corners = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist) -> list[float]:
    errors = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(o, r, t, K, dist)
        diff = projected.reshape(-1, 2) - i.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return errors


def save_coverage(img_pts, size, path: Path) -> None:
    w, h = size
    pts = np.concatenate([p.reshape(-1, 2) for p in img_pts])
    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f"Corner coverage ({len(img_pts)} images)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_undistort_comparison(image_path: Path, K, dist, path: Path) -> None:
    img = cv2.imread(str(image_path))
    und = cv2.undistort(img, K, dist)
    side = np.hstack([img, und])
    cv2.putText(side, "original", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
    cv2.putText(side, "undistorted", (img.shape[1] + 40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 200, 0), 8)
    scale = 2400 / side.shape[1]
    cv2.imwrite(str(path), cv2.resize(side, None, fx=scale, fy=scale))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    parser.add_argument("--cols", type=int, default=9, help="inner corners along the board width")
    parser.add_argument("--rows", type=int, default=7, help="inner corners along the board height")
    parser.add_argument("--square-mm", type=float, required=True, help="measured square size in mm")
    parser.add_argument("--max-error", type=float, default=None,
                        help="drop images whose reprojection error exceeds this (px) and recalibrate")
    parser.add_argument("--out", type=Path, default=ROOT / "camera_params.yaml")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    out_dir = ROOT / "output"
    (out_dir / "corners").mkdir(parents=True, exist_ok=True)

    board = object_points(args.cols, args.rows, args.square_mm)
    obj_pts, img_pts, used, size = [], [], [], None

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip {p.name}: unreadable")
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            print(f"  skip {p.name}: size {w}x{h} differs from {size[0]}x{size[1]}")
            continue
        corners = detect_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
        if corners is None:
            print(f"  skip {p.name}: board not found")
            continue
        obj_pts.append(board)
        img_pts.append(corners)
        used.append(p)
        vis = cv2.drawChessboardCorners(img.copy(), pattern, corners, True)
        cv2.imwrite(str(out_dir / "corners" / p.name), cv2.resize(vis, None, fx=0.3, fy=0.3))

    print(f"Board detected in {len(used)}/{len(paths)} images")
    if len(used) < 10:
        raise SystemExit("Need at least 10 usable images (20+ recommended).")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    errors = per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

    if args.max_error is not None:
        keep = [i for i, e in enumerate(errors) if e <= args.max_error]
        dropped = [used[i].name for i in range(len(used)) if i not in keep]
        if dropped:
            print(f"Dropping {len(dropped)} outlier(s) > {args.max_error} px: {', '.join(dropped)}")
            obj_pts = [obj_pts[i] for i in keep]
            img_pts = [img_pts[i] for i in keep]
            used = [used[i] for i in keep]
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
            errors = per_image_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

    print("\nPer-image reprojection error (px):")
    for p, e in sorted(zip(used, errors), key=lambda x: -x[1]):
        print(f"  {p.name:30s} {e:.3f}{'  <-- high' if e > 1.0 else ''}")

    verdict = "excellent" if rms < 0.3 else "acceptable" if rms < 0.5 else "too high — retake worst images"
    print(f"\nRMS reprojection error: {rms:.4f} px ({verdict})")
    print(f"fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}")
    print(f"dist (k1 k2 p1 p2 k3) = {np.round(dist.ravel(), 5).tolist()}")

    params = {
        "date": date.today().isoformat(),
        "image_size": list(size),
        "pattern_inner_corners": list(pattern),
        "square_mm": args.square_mm,
        "num_images": len(used),
        "rms_reprojection_error_px": float(rms),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "per_image_error_px": {p.name: round(e, 4) for p, e in zip(used, errors)},
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    save_coverage(img_pts, size, out_dir / "coverage.png")
    worst = used[int(np.argmax(errors))]
    save_undistort_comparison(worst, K, dist, out_dir / f"undistort_{worst.stem}.jpg")
    print(f"\nSaved {args.out} and visualisations in {out_dir}")


if __name__ == "__main__":
    main()
