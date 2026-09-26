"""Split the CVAT COCO export into train / val / test (70 / 20 / 10).

The split is stratified by physical book, so every book appears in every split
where its image count allows it. Category names are normalised to lowercase
("book", "card"). Output:
  dataset/splits/{train,val,test}.json   COCO files referencing dataset/undistorted/
  dataset/splits/stats.json              counts used in DATASET_CARD.md

Usage:
    python -m dataset.split
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

# Images left out of every split, with the reason (see docs/DATASET_CARD.md).
EXCLUDE = {
    "img_009.jpg": "a second, partially visible book (register) at the left edge is unlabelled",
}

# Raw capture ranges per physical book (ids match measurement/ground_truth.csv).
BOOK_RANGES = [
    (1, 9, 10),     # Computer education
    (11, 18, 8),    # Oxford dictionary
    (19, 28, 9),    # Register
    (29, 41, 7),    # English grammar
    (42, 52, 6),    # Oxley worksheet
    (53, 62, 5),    # Ahkam e Ilahi
    (63, 73, 4),    # White diary
    (74, 85, 3),    # Maroon diary (photographed open)
    (86, 98, 2),    # Gray diary
    (99, 110, 1),   # Physics
]


def book_of(file_name: str) -> int:
    n = int(Path(file_name).stem.split("_")[1])
    for lo, hi, book in BOOK_RANGES:
        if lo <= n <= hi:
            return book
    raise ValueError(f"{file_name} is outside the known capture ranges")


def split_counts(n: int) -> dict:
    """Per-book allocation: at least one val/test image once a book has enough images."""
    test = max(1, round(n * SPLITS["test"])) if n >= 5 else 0
    val = max(1, round(n * SPLITS["val"])) if n >= 3 else 0
    return {"train": n - val - test, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ROOT / "annotations" / "instances_default.json")
    parser.add_argument("--out", type=Path, default=ROOT / "splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text())
    for c in coco["categories"]:
        c["name"] = c["name"].lower()
        c["supercategory"] = ""

    excluded_ids = {i["id"] for i in coco["images"] if i["file_name"] in EXCLUDE}
    coco["images"] = [i for i in coco["images"] if i["id"] not in excluded_ids]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] not in excluded_ids]
    for name, reason in EXCLUDE.items():
        print(f"excluded {name}: {reason}")

    by_book = defaultdict(list)
    for img in coco["images"]:
        by_book[book_of(img["file_name"])].append(img)

    rng = random.Random(args.seed)
    assignment = {}
    for book in sorted(by_book):
        imgs = sorted(by_book[book], key=lambda i: i["file_name"])
        rng.shuffle(imgs)
        counts = split_counts(len(imgs))
        i = 0
        for name in ("test", "val", "train"):
            for img in imgs[i:i + counts[name]]:
                assignment[img["id"]] = name
            i += counts[name]

    args.out.mkdir(parents=True, exist_ok=True)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    stats = {"seed": args.seed, "excluded": EXCLUDE, "splits": {}}
    for name in SPLITS:
        images = [i for i in coco["images"] if assignment[i["id"]] == name]
        ids = {i["id"] for i in images}
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
        out.update(images=images, annotations=anns)
        (args.out / f"{name}.json").write_text(json.dumps(out))
        stats["splits"][name] = {
            "images": len(images),
            "instances": dict(Counter(cat_names[a["category_id"]] for a in anns)),
            "images_per_book": dict(sorted(Counter(book_of(i["file_name"]) for i in images).items())),
        }
        print(f"{name:5s}: {len(images):3d} images, {stats['splits'][name]['instances']}")

    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved splits to {args.out}")


if __name__ == "__main__":
    main()
