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

## Phase 3: matching, tuning, segmentation evaluation
- **Hungarian cost maximises the number of IoU >= thr matches first, total IoU second** | maximise total IoU (1 - IoU cost) | Pure IoU-sum assignment can trade a valid match for a higher-IoU one; the count of matches is what TP measures.
- **Border handling: drop regions touching the border; GT boxes touching the border are "don't care" (an interior region matched to one is neither TP nor FP; they are never misses)** | drop border regions only | 1130 of 4879 GT boxes (23%) touch the border; keeping them as misses would cap recall near 77% for every method.
- **Matching is class-agnostic; per-class recall uses the GT class of the matched box** | class-aware matching | Spec: "Matched -> GT class"; coarse types are for analysis only.
- **Count MAE uses interior regions (by coarse type) vs interior GT boxes per image** | all regions | Same border rule as P/R/F1.
- **Stage A (shared): wbc_dilate, (plt_min_area, plt_dilate), rbc_min_area tuned on train with class-specific F1 at IoU 0.5; RBC cutoff scored with watershed_fixed at default min_distance** | tune jointly with splitting params | Shared params affect all methods equally; separate stages keep the grid small. purple_thr, morphology radii and sigma_frac stay at train-derived defaults.
- **Stage A grid extended after the first run had its optimum on the edge (plt_min_area 250, plt_dilate 9, rbc_min_area 3000)** | keep the first grid | An edge optimum means the grid was too narrow.
- **Stage B: fixed min_distance and adaptive (k1, k2) selected by train F1 at IoU 0.5; val F1 reported as a held-out check, not used for selection** | select on val | Two to three parameters on 205 images overfit little; val stays an honest check.
- **Method used end-to-end in Phase 4 = best validation F1 at IoU 0.5** | best test F1 | Choosing by test would leak.
- **Stage B k2 grid extended down to 0 (h-dome test off) after the first run picked k2 = 0.025, its smallest value** | keep k2 >= 0.025 | Edge optimum; k2 = 0 tells whether the h-maxima step adds anything over min_distance alone. The run was stopped before the test step, so test was not touched with the earlier choice.
- **Final grids trimmed to the region around the optima (Stage A 30 configs, Stage B 38)** | full widened grids (61 + 59 configs, 446 s + 1136 s with 4 workers) | CLAUDE.md asks for ~30 min end to end; every optimum of the wide grids lies inside the trimmed grids.
- **Default 4 worker processes** | all 12 cores | A 12-worker run was killed by the OS for low memory on this laptop (other apps held most of 16 GB).
- **`--reuse` keeps tuning rows whose parameter hash matches and computes only missing configs** | always recompute | Makes an interrupted tuning run resumable; results are deterministic, so a reused table equals a full run. The rows from the interrupted run were tagged after checking its log (same Stage A picks, unchanged defaults).

## Phase 4: features and classification
- **GT-crop mask = the connected component of (Otsu foreground OR B - R stain) that covers most of the box's central half** | the whole rectangle | Shape/texture features need a pixel mask; using the segmenter's own masks keeps one feature definition for both settings.
- **GLCM on region pixels only (outside pixels get a reserved level 0 whose pairs are dropped), 32 levels, distance 1, 4 angles averaged** | GLCM of the full bounding box | The background around a cell would otherwise dominate the texture.
- **compactness = perimeter^2 / area** (redundant with circularity up to 4*pi) | extent or other variants | Spec lists both; the standard definition is kept and the redundancy noted.
- **hue_mean is a circular mean** | arithmetic mean of OpenCV hue | Pink/purple hues straddle the 0/360 wrap.
- **Classifiers tuned with GroupKFold(5) over train + val images (group = image) and refit on them; test used once** | train only | Val was used only for segmentation tuning; more images for CV.
- **class_weight='balanced' for SVM and RandomForest; none for GaussianNB/KNN** | resampling | Those two estimators do not support class weights; resampling would change the data the GaussianNB replication sees.
- **Setting 2 labels: Hungarian matching of all regions to all GT boxes at IoU >= 0.5; unmatched regions are excluded from classifier training and scoring; end-to-end recall = matched AND correctly classified / all test GT boxes of the class** | classify unmatched regions as "background" | The spec's three classes have no background class; misses and false regions are already measured in Evaluation A.
- **Extra row: GT-crop models applied to segmented test regions** | omit | Separates the effect of region shape (segmented masks vs box-derived masks) from that of the training data.
- **Leakage checks added (scripts/leakage_checks.py) because test accuracy exceeded 99%** | accept as is | CLAUDE.md requires it; the checks found no leakage and a depth-2 tree reaches 0.989 accuracy, so the coarse 3-class task is genuinely easy.

## Phase 5: interpretability and sensitivity
- **Permutation importance for SVM-RBF on GT crops (best CV macro-F1), test split, macro-F1 scoring, 20 repeats** | impurity importance of RandomForest | Model-agnostic and measured on held-out data, as the spec asks.
- **Per-class feature means z-scored against all train+val GT crops** | raw means | Features have very different units; z-scores make one heatmap readable.
- **Perturbations: rescale 0.7x / 1.3x (INTER_AREA down, INTER_LINEAR up; GT boxes mapped through pixel centres), hue +-10 deg (HSV), brightness x0.8 / x1.2 (RGB gain)** | larger shifts | Moderate shifts that a different microscope/stain batch could plausibly produce.
- **No re-tuning under perturbation; shared cutoffs stay in absolute pixels** | scale the area cutoffs with r | The spec's adaptive method scales only min_distance and h; changing that after seeing test results would be test tuning. The consequence (46.1% of test RBC boxes fall under rbc_min_area at 0.7x) is reported in results/tables/sensitivity_area_cutoff_check.csv.
