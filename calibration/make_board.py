"""Generate checkerboard calibration targets.

Produces:
  * a full-screen PNG for displaying on a monitor/laptop (used when no printer is available)
  * an A4 PDF at exact physical scale for printing

Usage:
    python -m calibration.make_board
    python -m calibration.make_board --screen-width 3072 --screen-height 1920
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT_DIR = Path(__file__).parent / "boards"

# 10 x 8 squares -> 9 x 7 inner corners (assessment minimum is 7 x 9)
SQUARES_X = 10
SQUARES_Y = 8


def make_screen_png(width: int, height: int, path: Path) -> int:
    """Draw the largest board that fits the screen with a white quiet zone. Returns square size in px."""
    square = int(min(width / (SQUARES_X + 2), height / (SQUARES_Y + 2)))
    img = np.full((height, width), 255, np.uint8)
    x0 = (width - square * SQUARES_X) // 2
    y0 = (height - square * SQUARES_Y) // 2
    for r in range(SQUARES_Y):
        for c in range(SQUARES_X):
            if (r + c) % 2 == 0:
                y, x = y0 + r * square, x0 + c * square
                img[y:y + square, x:x + square] = 0
    cv2.imwrite(str(path), img)
    return square


def make_a4_pdf(square_mm: float, path: Path) -> None:
    """Vector A4 landscape PDF. Must be printed at 100% / actual size."""
    page_w, page_h = 297.0, 210.0
    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal")
    ax.axis("off")

    x0 = (page_w - square_mm * SQUARES_X) / 2
    y0 = (page_h - square_mm * SQUARES_Y) / 2
    for r in range(SQUARES_Y):
        for c in range(SQUARES_X):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * square_mm, y0 + r * square_mm),
                                       square_mm, square_mm, color="black", lw=0))
    ax.text(page_w / 2, 6, f"{SQUARES_X - 1}x{SQUARES_Y - 1} inner corners | square = {square_mm} mm | "
            "print at 100% (actual size)", ha="center", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-width", type=int, default=3072, help="native screen width in px")
    parser.add_argument("--screen-height", type=int, default=1920, help="native screen height in px")
    parser.add_argument("--square-mm", type=float, default=22.0, help="square size for the printable PDF")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    png = OUT_DIR / "checkerboard_screen.png"
    square_px = make_screen_png(args.screen_width, args.screen_height, png)
    pdf = OUT_DIR / "checkerboard_A4.pdf"
    make_a4_pdf(args.square_mm, pdf)

    print(f"Screen board: {png}  ({square_px} px squares)")
    print(f"Print board:  {pdf}  ({args.square_mm} mm squares)")
    print(f"Inner corners: {SQUARES_X - 1} x {SQUARES_Y - 1}")


if __name__ == "__main__":
    main()
