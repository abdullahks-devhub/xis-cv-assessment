# Dataset Card: Books and Reference Card

## 1. Summary

| Item | Value |
|---|---|
| Task | Instance segmentation (polygon masks) |
| Classes | `book` (measurement target), `card` (scale reference) |
| Images captured | 110 (training set) + 20 (measurement validation, kept separate) |
| Images in splits | **108** (see Section 6 for the 2 excluded images) |
| Instances in splits | 109 `book`, 107 `card` |
| Resolution | Captured at 4160 × 3120, undistorted and resized to 2080 × 1560 |
| Labelling tool | CVAT (app.cvat.ai), polygon tool, exported as COCO 1.0 |
| Split | 70 / 20 / 10, stratified by physical book, seed 42 |

## 2. Object choice

**Target: books.** They are widely available, and a set of ten gives a range of sizes (133–210 mm wide, 210–346 mm tall) for the accuracy study. Their near-rectangular shape suits polygon labelling (4–7 points) and gives a clear definition of "width" and "height". The set is also visually varied: glossy, matte, plastic-wrapped and spiral-bound covers, plus one diary photographed open.

**Reference: a trading card.** A Pokémon card, made to a standard size of **63 × 88 mm**, confirmed with a ruler. It carries no personal information. That matters because ID cards (CNIC) and bank cards have the same ISO size but contain sensitive data, and these images are published. A printer was not available for an ArUco marker. The card is segmented by the same model as a second class, so no separate detector is needed.

## 3. The ten books

Ground truth was measured with a ruler (±2 mm) and is stored in [`measurement/ground_truth.csv`](../measurement/ground_truth.csv).

| ID | Book | W × H × T (mm) | Raw image range | Images in splits |
|---|---|---|---|---|
| 1 | Physics textbook | 175 × 235 × 12 | img_099–110 | 12 |
| 2 | Gray diary | 187 × 250 × 11 | img_086–098 | 13 |
| 3 | Maroon diary (photographed open) | 197 × 250 × 11 | img_074–085 | 12 |
| 4 | White diary (spiral) | 146 × 210 × 16 | img_063–073 | 11 |
| 5 | Ahkam e Ilahi | 186 × 246 × 18 | img_053–062 | 10 |
| 6 | Oxley worksheet | 202 × 288 × 21 | img_042–052 | 11 |
| 7 | English grammar | 175 × 233 × 16 | img_029–041 | 13 |
| 8 | Oxford dictionary | 133 × 213 × 22 | img_011–018 | 8 |
| 9 | Register (spiral, plastic cover) | 210 × 346 × 18 | img_019–028 | 10 |
| 10 | Computer education | 178 × 232 × 8 | img_001–009 | 8 |

## 4. Collection strategy

- **Camera:** The calibrated Huawei Y7 Prime 2019, with the same Open Camera settings as calibration (max resolution, landscape lock, standard mode, no auto-level, tap-to-focus at about 40 cm). See [CALIBRATION_REPORT.md](CALIBRATION_REPORT.md).
- **Planned variation per book (about 8–13 shots):** centred and straight; rotated about 30° and 90°; near each edge of the frame; a slightly tilted camera; and some shots with a second book in view. The card is placed in a different position around the book in each shot.
- **Backgrounds:** A black bag, a blue patterned bed sheet, a green-and-gold patterned rug, and a white floor at the frame edges.
- **Lighting:** Indoor daylight and ceiling lights. Several images contain glare on glossy or plastic covers. These were kept on purpose so the model learns to handle real conditions.
- **Separate measurement set:** 20 images (2 per book) were taken with the camera pointing straight down and the card raised on a 16 mm support. They are in `measurement/images/` and are **never used for training**. See [MEASUREMENT_REPORT.md](MEASUREMENT_REPORT.md).

## 5. Pre-processing and labelling

1. Raw captures (`dataset/raw/`, 4160 × 3120) were resized to 2080 × 1560 and undistorted with the Step 1 calibration by [`dataset/prepare.py`](../dataset/prepare.py). The output is in `dataset/undistorted/`. **Labels and training use only undistorted images.**
2. Undistorted images were uploaded to CVAT. Each book and card was traced with a polygon (184 × 4 points, 25 × 5, 7 × 6, 2 × 7). Books include the spine and spiral binding. Card corners are placed just inside the rounded corners.
3. The export is stored unchanged in `dataset/annotations/instances_default.json`.
4. [`dataset/split.py`](../dataset/split.py) lowercases the class names, applies the documented label fix, removes the excluded image, and writes the splits.

**Quality control:** Every label was checked visually against the image as an overlay. An area check was also run: after the correction below, every `card` covers 2.4–7.6% of the frame and every `book` covers 16.6–53.2%, so the two classes cannot be confused by size.

## 6. Exclusions and corrections

| Image | Issue | Action |
|---|---|---|
| `img_010.jpg` | An open book held by hand, which is not representative of the use case | Not uploaded for labelling |
| `img_009.jpg` | A second book (the register) is partly visible at the left edge but was not labelled | Excluded from all splits |
| `img_002.jpg` | The `book` and `card` labels were swapped. Found by the area check (a "card" covering 45% of the frame) | Corrected in `split.py` via `LABEL_FIXES` |

**Note:** The swap in `img_002` (a training image) was found *after* the model was trained, so the delivered model saw this one image with swapped labels. That is 2 of 152 training instances. Validation and test labels were not affected. The reported metrics are therefore computed against correct labels. See [TRAINING_REPORT.md](TRAINING_REPORT.md), Section 7.

## 7. Splits and class distribution

Each book's images are shuffled (seed 42) and allocated about 10% to test (at least 1), about 20% to val (at least 1) and the rest to train. Stratifying by book guarantees that every book appears in every split, and that the test set contains exactly one image of each book.

| Split | Images | `book` | `card` | Share |
|---|---|---|---|---|
| Train | 76 | 77 | 75 | 70.4% |
| Val | 22 | 22 | 22 | 20.4% |
| Test | 10 | 10 | 10 | 9.3% |
| **Total** | **108** | **109** | **107** | |

The classes are balanced, with one card per image. `img_019` has no card in frame, and `img_018` has two books.

Per-book counts for each split are in [`dataset/splits/stats.json`](../dataset/splits/stats.json).

## 8. Limitations

- **Small dataset:** 108 images of **10 physical books**. Test images are new photos of books that also appear in training, so test metrics measure generalisation to new views and conditions, **not to unseen books**.
- **Near-duplicate shots:** Consecutive photos of the same book on the same background are similar. This inflates validation and test scores compared with a truly independent set.
- **Mostly flat, closed books on a few backgrounds.** Open books (except book 3), stacked books, cluttered scenes and very different lighting are not covered.
- **One card instance:** The same physical card appears in every image, so the model has learned *this* card, not trading cards in general.
