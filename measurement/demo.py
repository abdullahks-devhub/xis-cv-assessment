"""End-to-end demo: one raw photo -> book width and height in millimetres.

Pipeline: resize to working resolution -> undistort (calibration) -> Mask R-CNN
(book + card) -> quadrilaterals from masks -> card-based px/mm with height correction
-> width, height, confidence.

Place the card flat beside the book. For best accuracy, raise the card to the height
of the book's top cover; otherwise pass both heights so the scale is corrected.

Usage:
    python -m measurement.demo --image photo.jpg
    python -m measurement.demo --image photo.jpg --book-thickness-mm 12 --card-height-mm 16.3
"""

import argparse
import json
from pathlib import Path

import cv2

from calibration.undistort import load_params
from inference.predict import Segmenter
from measurement.measure import draw, measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="raw photo from the calibrated camera")
    parser.add_argument("--out", type=Path, default=Path("measurement/output/demo"))
    parser.add_argument("--weights", default="models/output/best.pth")
    parser.add_argument("--book-thickness-mm", type=float, default=0.0,
                        help="height of the book's top cover above the table")
    parser.add_argument("--card-height-mm", type=float, default=0.0,
                        help="height of the card above the table (0 = lying on the table)")
    args = parser.parse_args()

    raw = cv2.imread(str(args.image))
    if raw is None:
        raise SystemExit(f"Cannot read {args.image}")

    seg = Segmenter(args.weights)
    K = load_params()["K"]
    image, detections = seg(raw)
    m = measure(image, detections, focal_px=(K[0, 0] + K[1, 1]) / 2,
                cover_above_card_mm=args.book_thickness_mm - args.card_height_mm)

    args.out.mkdir(parents=True, exist_ok=True)
    overlay_path = args.out / f"{args.image.stem}_measured.jpg"
    cv2.imwrite(str(overlay_path), draw(image, detections, m))
    result = {
        "image": args.image.name,
        "width_mm": round(m.width_mm, 1),
        "height_mm": round(m.height_mm, 1),
        "confidence": round(m.book_score, 3),
        "card_confidence": round(m.card_score, 3),
        "px_per_mm": round(m.px_per_mm, 4),
        "camera_distance_mm": round(m.camera_distance_mm, 1),
        "notes": m.notes,
        "overlay": str(overlay_path),
    }
    (args.out / f"{args.image.stem}_measured.json").write_text(json.dumps(result, indent=2))

    print(f"Width:      {result['width_mm']} mm")
    print(f"Height:     {result['height_mm']} mm")
    print(f"Confidence: {result['confidence']}")
    for note in m.notes:
        print(f"Note:       {note}")
    print(f"Overlay:    {overlay_path}")


if __name__ == "__main__":
    main()
