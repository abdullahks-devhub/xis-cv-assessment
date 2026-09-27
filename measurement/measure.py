"""Pixel-to-millimetre measurement from segmentation masks.

Method
------
1. The card (standard trading-card size, 63 x 88 mm) and the book are segmented
   by Mask R-CNN on the undistorted image.
2. Each mask is converted to a quadrilateral:
     * "mask" (default): robust lines fitted to the mask contour, one per side of its
       minimum-area rectangle; the side lengths are the means of opposite sides.
     * "edge": each mask line is additionally snapped to the strongest nearby image
       gradient with sub-pixel precision. Evaluated as an alternative; it gave no
       measurable gain over "mask" on this data (see docs/MEASUREMENT_REPORT.md).
3. Card side lengths give the scale at the card's plane: s_card = mean(long_px / 88, short_px / 63).
4. Height correction (pinhole model). With focal length f (px) from calibration, the card's
   distance is Z_card = f / s_card. If the book's top cover is dz mm closer to the camera than
   the card (dz = book thickness - card support height), the scale at the cover is
       s_book = f / (Z_card - dz) = s_card * Z_card / (Z_card - dz).
5. Book width / height = its short / long side in px divided by s_book.

Assumptions: the card and the book cover are parallel to the table, and the camera looks
roughly straight down.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

CARD_MM = (63.0, 88.0)  # (short, long) — standard trading-card dimensions


@dataclass
class Quad:
    corners: np.ndarray                # 4x2, ordered around the shape
    sides_px: tuple[float, float]      # (short, long): mean of opposite side lengths
    method: str
    edge_residual_px: float | None = None  # RMS distance of edge points to fitted lines


@dataclass
class Measurement:
    width_mm: float
    height_mm: float
    px_per_mm: float                   # scale at the book's top cover (after height correction)
    card_px_per_mm: float              # scale at the card's plane
    camera_distance_mm: float | None   # Z_card, when the focal length is known
    scale_anisotropy: float            # (long/88) / (short/63); 1.0 = card seen without distortion
    book_score: float
    card_score: float
    book: Quad
    card: Quad
    notes: list[str] = field(default_factory=list)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask")
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust (Huber) line fit. Returns (unit direction, point on line)."""
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([vx, vy]), np.array([x0, y0])


def _intersect(l1, l2) -> np.ndarray:
    (d1, p1), (d2, p2) = l1, l2
    a = np.array([d1, -d2]).T
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def _quad_from_lines(lines) -> np.ndarray:
    return np.array([_intersect(lines[i], lines[(i + 1) % 4]) for i in range(4)])


def _sides(corners: np.ndarray) -> tuple[float, float]:
    lengths = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    a, b = (lengths[0] + lengths[2]) / 2, (lengths[1] + lengths[3]) / 2
    return (min(a, b), max(a, b))


def mask_lines(mask: np.ndarray) -> list:
    """One line per side: contour points are assigned to the nearest side of the min-area rectangle."""
    contour = _largest_contour(mask)
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.float64)
    dists = []
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        dists.append(np.abs((contour - a) @ n))
    side_of = np.argmin(np.stack(dists, axis=1), axis=1)
    lines = []
    for i in range(4):
        pts = contour[side_of == i]
        a, b = box[i], box[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        t = (pts - a) @ d / np.linalg.norm(b - a)
        pts = pts[(t > 0.15) & (t < 0.85)]  # skip rounded / clipped corners
        lines.append(_fit_line(pts) if len(pts) >= 10 else (d, a))
    return lines


def mask_quad(mask: np.ndarray) -> Quad:
    corners = _quad_from_lines(mask_lines(mask))
    return Quad(corners, _sides(corners), "mask")


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def edge_quad(image_bgr: np.ndarray, mask: np.ndarray, search_px: float = 12.0, samples: int = 80) -> Quad:
    """Refine each mask side to the strongest nearby image edge (sub-pixel), then re-fit the lines."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    init = mask_lines(mask)
    corners0 = _quad_from_lines(init)
    offsets = np.arange(-search_px, search_px + 0.25, 0.5)
    prior = np.exp(-offsets ** 2 / (2 * (search_px / 2) ** 2))  # prefer edges close to the mask boundary

    lines, residuals = [], []
    for i in range(4):
        a, b = corners0[i], corners0[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        ts = np.linspace(0.15, 0.85, samples)
        base = a[None] + ts[:, None] * (b - a)[None]
        xs = base[:, 0:1] + offsets[None] * n[0]
        ys = base[:, 1:2] + offsets[None] * n[1]
        g = np.abs(_bilinear(gx, xs, ys) * n[0] + _bilinear(gy, xs, ys) * n[1]) * prior[None]

        k = np.clip(np.argmax(g, axis=1), 1, len(offsets) - 2)
        rows = np.arange(len(k))
        g0, g1, g2 = g[rows, k - 1], g[rows, k], g[rows, k + 1]
        denom = g0 - 2 * g1 + g2
        sub = np.where(np.abs(denom) > 1e-6, 0.5 * (g0 - g2) / denom, 0.0)  # parabola peak
        s = offsets[k] + np.clip(sub, -0.5, 0.5) * 0.5
        strong = g1 > 0.25 * np.median(g1)
        pts = base[strong] + s[strong, None] * n[None]

        if len(pts) < 10:
            lines.append(init[i])
            continue
        line = _fit_line(pts)
        dist = np.abs((pts - line[1]) @ np.array([-line[0][1], line[0][0]]))
        inliers = pts[dist < 2.0]
        if len(inliers) >= 10:
            line = _fit_line(inliers)
            dist = np.abs((inliers - line[1]) @ np.array([-line[0][1], line[0][0]]))
        residuals.append(float(np.sqrt(np.mean(dist ** 2))))
        lines.append(line)

    corners = _quad_from_lines(lines)
    return Quad(corners, _sides(corners), "edge", float(np.mean(residuals)) if residuals else None)


def measure(image_bgr: np.ndarray, detections: list[dict], method: str = "mask",
            focal_px: float | None = None, cover_above_card_mm: float = 0.0) -> Measurement:
    """Measure the highest-confidence book using the highest-confidence card as the scale reference.

    focal_px: camera focal length in working-resolution pixels (enables the height correction).
    cover_above_card_mm: how much higher the book's top cover is than the card (book thickness minus
        card support height); negative when the card sits higher.
    """
    books = [d for d in detections if d["class"] == "book"]
    cards = [d for d in detections if d["class"] == "card"]
    if not books or not cards:
        raise ValueError(f"need a book and a card, got {len(books)} book(s) and {len(cards)} card(s)")
    book, card = books[0], cards[0]

    to_quad = (lambda m: edge_quad(image_bgr, m)) if method == "edge" else mask_quad
    bq, cq = to_quad(book["mask"]), to_quad(card["mask"])

    scale_short = cq.sides_px[0] / CARD_MM[0]
    scale_long = cq.sides_px[1] / CARD_MM[1]
    card_scale = (scale_short + scale_long) / 2
    distance = focal_px / card_scale if focal_px else None
    px_per_mm = card_scale * distance / (distance - cover_above_card_mm) if distance else card_scale

    notes = []
    if len(books) > 1:
        notes.append(f"{len(books)} books detected; measured the most confident")
    anisotropy = scale_long / scale_short
    if abs(anisotropy - 1) > 0.08:
        notes.append(f"card aspect off by {100 * (anisotropy - 1):+.1f}% — camera may be tilted")

    return Measurement(
        width_mm=bq.sides_px[0] / px_per_mm,
        height_mm=bq.sides_px[1] / px_per_mm,
        px_per_mm=px_per_mm,
        card_px_per_mm=card_scale,
        camera_distance_mm=distance,
        scale_anisotropy=anisotropy,
        book_score=book["score"],
        card_score=card["score"],
        book=bq,
        card=cq,
        notes=notes,
    )


def draw(image: np.ndarray, detections: list[dict], m: Measurement) -> np.ndarray:
    """Mask overlay + fitted quadrilaterals + metric labels."""
    from inference.predict import annotate

    out = annotate(image, detections)
    for q, color in ((m.book, (0, 255, 0)), (m.card, (0, 255, 255))):
        cv2.polylines(out, [q.corners.round().astype(np.int32)], True, color, 3)
    c = m.book.corners.mean(axis=0).astype(int)
    lines = [f"W {m.width_mm:.1f} mm", f"H {m.height_mm:.1f} mm", f"conf {m.book_score:.2f}"]
    for k, text in enumerate(lines):
        org = (int(c[0]) - 170, int(c[1]) - 40 + 60 * k)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(out, f"scale {m.px_per_mm:.3f} px/mm at book cover (from card)", (30, out.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return out
