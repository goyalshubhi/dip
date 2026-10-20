# Decisions log

Format: choice | alternatives | reason.

## Phase 1: environment and data
- **Python 3.11 venv** | 3.13/3.14 | Python 3.11 is the most mature wheel target for the pinned scientific stack on Windows.
- **Use the repo's own ImageSets/Main split (train 205 / val 87 / test 72)** | fresh 70/15/15 seed-42 split | CLAUDE.md says to use repo lists if provided; they are image-level, disjoint, and cover all 364 images (checked in code). Saved to data/splits.json.
- **Treat VOC coordinates as 1-based and subtract 1** | keep as is and clip | BCCD boxes have min coordinate 1 and max 640/480 on 640x480 images, which only fits a 1-based convention; this is a convention conversion, not counted as clipping.
- **Image id = XML file stem; image size read from the JPEG** | trust `<filename>` / `<size>` in XML | Robust to stale XML metadata (all 364 currently agree).
- **Drop the later box of same-class pairs with IoU > 0.9 (7 boxes)** | keep them | They are double annotations of one cell; keeping them creates misses no segmenter can avoid.
- **Drop 2 point boxes (xmin == xmax, ymin == ymax)** | inflate to a default size | Zero-size boxes carry no extent; CLAUDE.md asks to drop/fix them.
