# Decisions log

Format: choice | alternatives | reason.

## Phase 1: environment and data
- **Python 3.11 venv** | 3.13/3.14 | Python 3.11 is the most mature wheel target for the pinned scientific stack on Windows.
- **Use the repo's own ImageSets/Main split (train 205 / val 87 / test 72)** | fresh 70/15/15 seed-42 split | CLAUDE.md says to use repo lists if provided; they are image-level, disjoint, and cover all 364 images (checked in code). Saved to data/splits.json.
- **Treat VOC coordinates as 1-based and subtract 1** | keep as is and clip | BCCD boxes have min coordinate 1 and max 640/480 on 640x480 images, which only fits a 1-based convention; this is a convention conversion, not counted as clipping.
- **Image id = XML file stem; image size read from the JPEG** | trust `<filename>` / `<size>` in XML | Robust to stale XML metadata (all 364 currently agree).
- **Drop the later box of same-class pairs with IoU > 0.9 (7 boxes)** | keep them | They are double annotations of one cell; keeping them creates misses no segmenter can avoid.
- **Drop 2 point boxes (xmin == xmax, ymin == ymax)** | inflate to a default size | Zero-size boxes carry no extent; CLAUDE.md asks to drop/fix them.

## Phase 2: preprocessing and segmentation
- **Threshold the CLAHE + Gaussian-blurred + min-max green channel with Otsu** | LAB L*, saturation | Cells are darker than background in green (train medians at box centres: background 198, RBC 170 incl. pale centres, WBC 110; `results/tables/segmentation_design_checks.csv`).
- **WBC/platelet channel = B - R (smoothed)** | HSV saturation | Train medians at box centres: background -3.8, RBC -16.8, WBC 65.4, platelet 43.7 (`results/tables/segmentation_design_checks.csv`); a fixed threshold of 20 separates stain from RBCs.
- **Shared foreground, WBC and platelet path for all four methods** | method-specific pipelines | The comparison then isolates how the RBC-like foreground is split.
- **Fill holes also from a radius-7 closing, but keep only the filled holes** | plain radius-5 closing | On 205 train images the share of GT RBC centres inside the mask rises from 0.911 to 0.938 with 16.14 components/image, vs 0.931 and 14.89 (more merging) for a radius-5 closing (`results/tables/segmentation_design_checks.csv`).
- **watershed_plain = watershed on -EDT with every regional maximum as a seed** | watershed on the Sobel gradient | Gradient basins are mostly < 100 px slivers; the EDT version over-segments visibly and differs from watershed_fixed only in marker selection.
- **WBC = convex hull of the grouped nucleus grown by 12 px; platelet = stain blob grown by 6 px** | fixed-size boxes | On train, GT boxes are about 25-30 px (WBC) and 12 px (platelet) wider than the stained blob.
- **Two nucleus-sized blobs in one stain group become two WBCs** | always one WBC per group | Adjacent WBCs (e.g. BloodImage_00176) were merged by the lobe-grouping dilation.
- **Isolated-cell filter: circularity >= 0.70, solidity >= 0.85, not on border, area >= 1500; min_isolated = 1** | 0.80 / 0.90 / 3 | Dense smears rarely contain 3 isolated round RBCs; with min_isolated = 1, 72.2% of train images get their own estimate (`results/tables/radius_estimator_check.csv`), the rest use the training median.
- **Adaptive markers: peaks of the Gaussian-smoothed EDT (sigma = 0.1 r) at least k1*r apart that top an h-dome (h = k2*r); highest peak per dome** | h-maxima regions only | Combines both controls named in the spec; one marker per dome avoids splitting plateaus.
- **Every RBC-like component without a marker gets one at its EDT maximum** | leave it unlabelled | Prevents small cells from vanishing when h exceeds their height.
- **Area cutoff (1500 px) applied to connected components before splitting; only < 100 px slivers dropped after** | cutoff after splitting | A post-split cutoff would hide watershed_plain's over-segmentation.
