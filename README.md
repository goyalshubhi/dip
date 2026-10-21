# Watershed-Based Blood Cell Segmentation and Classification on BCCD

Course project, VIT BCSE403L Digital Image Processing. A classical (no deep learning) pipeline on the
BCCD dataset: preprocessing -> Otsu threshold + morphology -> distance transform -> marker-controlled
watershed -> cell extraction -> 22 handcrafted features -> classical classifiers -> evaluation.

Our modest extension is an **adaptive marker-controlled watershed**: it estimates a typical RBC radius
`r` per image from isolated, roughly circular cells and sets the marker spacing (`k1 * r`) and the
h-maxima height (`k2 * r`) from it. On unscaled BCCD it does **not** beat a well-tuned fixed watershed;
it helps only when images are shrunk (0.7x). Details and caveats below.

Every number in this README is copied from a file in `results/`; the file is named next to it.

## Reproduce everything

Requirements: Python 3.10+ (3.11.9 used), `git`, and a bash shell (Git Bash on Windows).

```bash
bash scripts/run_all.sh
```

This creates `.venv`, installs the pinned `requirements.txt`, clones BCCD into `data/` (skipped if
present), and regenerates every table, figure and log in `results/`, then runs the tests. It uses 4
worker processes (about 250 MB each); use `JOBS=2 bash scripts/run_all.sh` if memory is tight. Set
`PYTHON=/path/to/python3.11` to choose the interpreter.

Individual steps (from the repository root, after activating `.venv`):

```bash
bash scripts/download_data.sh                      # BCCD -> data/BCCD_Dataset (idempotent)
python -m src.loader                               # cleaning, splits, dataset stats, GT overlays
python scripts/run_segmentation_eval.py            # design checks, radius, tuning, test, sensitivity
python scripts/run_segmentation_eval.py --steps tune --reuse   # resume an interrupted tuning run
python scripts/run_classification.py               # features, classifiers, interpretability
python scripts/leakage_checks.py                   # checks behind the >99 % accuracy
python scripts/make_figures.py                     # figures
python -m pytest -q tests                          # 27 tests
```

`notebooks/01_explore.ipynb` reads the results and re-runs the pipeline on a training image.

## Repository layout

```
src/        loader.py preprocess.py segment.py match.py features.py classify.py evaluate.py utils.py
scripts/    download_data.sh run_segmentation_eval.py run_classification.py leakage_checks.py
            make_figures.py run_all.sh
tests/      test_loader.py test_segment.py test_match.py test_features.py
notebooks/  01_explore.ipynb
results/    figures/ tables/ logs/   (cache/ is git-ignored: feature tables and fitted models)
PROGRESS.md DECISIONS.md
```

## Data

BCCD (https://github.com/Shenggan/BCCD_Dataset): 364 JPEGs (640x480) with Pascal VOC boxes for RBC,
WBC and Platelets. Cleaning (`results/logs/cleaning_log.json`): 4888 raw boxes -> 4879 (2 zero-size
boxes and 7 same-class near-duplicates with IoU > 0.9 dropped; no class names needed fixing; no box
was outside the image). The repository's own image-level split is used: train 205 / val 87 / test 72
images (`results/tables/dataset_stats.csv`; test: 804 RBC, 71 WBC, 69 Platelets).

BCCD has **no pixel masks**, so segmentation is scored against boxes. Annotators skipped many RBCs
(especially at the border and in dense regions), so **precision against BCCD boxes is a lower bound**
throughout.

## Method

- **Preprocessing** (`src/preprocess.py`): green channel and LAB, CLAHE, Gaussian blur, min-max
  normalisation, and a B - R "stain" channel for the purple WBC nuclei and platelets.
- **Foreground**: Otsu on the processed green channel, closing, hole filling (also holes sealed only by
  a larger closing, for pale-centred RBCs), opening.
- **WBCs and platelets** (shared by all methods): B - R > 20; lobes grouped; a group with at least
  4000 stained pixels is a WBC (convex hull grown by `wbc_dilate`), smaller groups are platelets.
- **Four ways to split the RBC-like foreground** (`src/segment.py`):
  `otsu_morph` (connected components), `watershed_plain` (watershed on the distance transform with
  every regional maximum as a seed: the over-segmentation baseline), `watershed_fixed`
  (`peak_local_max` with a fixed `min_distance`), `watershed_adaptive` (per-image `r`, markers at
  least `k1*r` apart that top an h-dome of height `k2*r` on the smoothed distance transform; falls
  back to the training-set median `r` when an image has no isolated round cell).
- **Matching** (`src/match.py`): region boxes vs GT boxes, Hungarian assignment at IoU >= 0.5 (and 0.3).
  Regions touching the border are ignored; GT boxes touching the border are "don't care".
- **Tuning** on train with val as a check (the test split is used once per method):
  `results/tables/seg_tuning_shared.csv`, `results/tables/seg_tuning_splitting.csv`,
  chosen values in `results/tables/seg_params.json` (fixed `min_distance` 36; adaptive `k1` 0.7,
  `k2` 0.01; `rbc_min_area` 4000, `wbc_dilate` 15, `plt_min_area` 250, `plt_dilate` 9).
- **Features** (`src/features.py`, list in `results/tables/feature_list.csv`): exactly 22 features:
  mean R/G/B; gray mean/std/min/max; GLCM contrast/correlation/energy/homogeneity; area, perimeter,
  circularity, aspect ratio, compactness; Canny edge density; solidity, eccentricity, equivalent
  diameter, circular hue mean, saturation mean.
- **Classifiers** (`src/classify.py`): GaussianNB (the setting of Rokhani 2026), KNN, SVM-RBF,
  RandomForest, each in a StandardScaler pipeline, `class_weight='balanced'` where supported,
  hyperparameters by GroupKFold (group = image) over train+val images.

## Results

### A. Segmentation (test split, 72 images; `results/tables/segmentation_comparison.csv`)

| method | precision@0.5 | recall@0.5 | F1@0.5 | F1@0.3 | RBC recall@0.5 | RBC count MAE | overall count MAE | regions / image |
|---|---|---|---|---|---|---|---|---|
| otsu_morph | 0.728 | 0.354 | 0.476 | 0.549 | 0.273 | 5.264 | 5.389 | 4.93 |
| watershed_plain | 0.032 | 0.374 | 0.058 | 0.116 | 0.296 | 107.319 | 107.333 | 117.18 |
| watershed_fixed | 0.646 | 0.652 | **0.649** | **0.733** | 0.625 | **2.514** | **2.806** | 10.21 |
| watershed_adaptive | 0.630 | 0.659 | 0.644 | 0.730 | 0.634 | 2.597 | 2.917 | 10.60 |

Interior GT boxes per image: 9.85. WBC recall@0.5 (0.900) and platelet recall@0.5 (0.705) are the
same for all four methods because that path is shared. Precision is a lower bound (unannotated RBCs).
Plain watershed over-segments as expected (117 regions per image); otsu_morph merges touching cells.
Figure: `results/figures/segmentation_grid_test.png` (original | otsu_morph | plain | fixed | adaptive,
first row = the densest RBC cluster in the test split).

### B. Classification (test split; `results/tables/classification_comparison.csv`)

Setting 1 = one sample per GT box (comparable to the literature). Setting 2 = regions from
`watershed_fixed` (best on val), labelled by IoU >= 0.5 matching; "e2e recall" = share of all test GT
boxes of a class that were both found and correctly classified.

| setting | model | accuracy | macro-F1 | recall RBC | recall WBC | recall Platelets | e2e recall RBC | e2e recall WBC | e2e recall Platelets |
|---|---|---|---|---|---|---|---|---|---|
| GT crops (n = 944) | GaussianNB | 0.997 | 0.990 | 0.999 | 1.000 | 0.971 | - | - | - |
| GT crops | KNN | 0.998 | 0.993 | 1.000 | 1.000 | 0.971 | - | - | - |
| GT crops | SVM-RBF | 0.997 | 0.990 | 1.000 | 1.000 | 0.957 | - | - | - |
| GT crops | RandomForest | 0.999 | 0.995 | 1.000 | 1.000 | 0.986 | - | - | - |
| segmented (n = 680) | GaussianNB | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
| segmented | KNN | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
| segmented | SVM-RBF | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
| segmented | RandomForest | 0.996 | 0.988 | 0.998 | 0.986 | 0.980 | 0.695 | 0.972 | 0.710 |

**The gap.** On regions the segmenter did find, classification is as accurate as on GT crops; the
end-to-end loss is almost entirely segmentation: only 560 of the 804 test RBC boxes and 50 of the 69
platelet boxes are matched by a region at all (`n_RBC`, `n_Platelets` in the CSV vs
`results/tables/dataset_stats.csv`). GT-crop models applied to segmented regions (extra rows in the
CSV) lose platelet recall (0.840 to 0.960) because segmented platelet masks differ in shape from
box-derived ones.

**Why > 99 %, and why it is not leakage** (`results/tables/leakage_checks.csv`): no image is shared
between train+val and test; with shuffled labels the CV macro-F1 drops to 0.256; and a depth-2 decision
tree (size + one colour threshold) already reaches 0.989 test accuracy. RBC vs WBC vs platelet is easy
with size and stain colour; WBC subtypes, the hard problem, are not part of BCCD's labels here.
Confusion matrices: `results/figures/confusion_matrices.png`.

### C. Interpretability (`results/tables/permutation_importance.csv`, `results/tables/class_feature_means.csv`)

Permutation importance (SVM-RBF on GT crops, test split, drop in macro-F1) ranks equivalent diameter
(0.0129), mean blue (0.0072), saturation (0.0041) and perimeter (0.0039) highest; circularity, solidity,
eccentricity and hue mean score 0.0000. Per-class z-scored means explain this: WBCs are large (area
+2.59) and strongly stained (saturation +2.79, mean blue +1.82); platelets are small (equivalent
diameter -2.05) and stained (mean blue +1.63) with the coarsest texture (GLCM contrast +2.27,
homogeneity -2.54); RBCs sit near the average. Roundness barely differs between classes. Individual
importances are small because size and colour features come in redundant groups: permuting one leaves
its correlates intact. Figures: `results/figures/permutation_importance.png`,
`results/figures/class_feature_means.png`.

### D. Sensitivity: fixed vs adaptive watershed (test split; `results/tables/sensitivity.csv`)

| condition | fixed F1@0.5 | adaptive F1@0.5 | fixed RBC count MAE | adaptive RBC count MAE |
|---|---|---|---|---|
| original | 0.649 | 0.644 | 2.514 | 2.597 |
| rescaled 0.7x | 0.482 | **0.512** | 3.889 | **3.417** |
| rescaled 1.3x | 0.639 | 0.640 | 2.958 | 2.736 |
| hue -10 deg | 0.655 | 0.656 | 2.444 | 2.458 |
| hue +10 deg | 0.650 | 0.646 | 2.569 | 2.597 |
| brightness x0.8 | 0.647 | 0.639 | 2.403 | 2.389 |
| brightness x1.2 | 0.637 | 0.636 | 2.431 | 2.333 |

The adaptive watershed wins only on shrunk images (0.7x). At 1.3x and under colour shifts the two
methods are within 0.008 F1. Both lose recall at 0.7x because the shared RBC area cutoff is in
absolute pixels: 46.1 % of test RBC boxes fall below it at 0.7x vs 0.4 % at 1.0x
(`results/tables/sensitivity_area_cutoff_check.csv`). That cutoff also gates the isolated-cell filter,
so only 38.9 % of 0.7x images get their own radius estimate (63.9 % unscaled). Figure:
`results/figures/sensitivity.png`.

## What worked, what did not

- Worked: marker-controlled watershed roughly halves the RBC count error of plain Otsu + morphology
  (2.514 vs 5.264 per image) and lifts F1@0.5 from 0.476 to 0.649; the shared stain path finds WBCs
  reliably (recall@0.5 0.900, count MAE 0.111); classical features separate the three classes almost
  perfectly once a cell is found.
- Did not work as hoped: the adaptive watershed is not better than a tuned fixed one on BCCD, where
  all images share one magnification, so a single well-tuned spacing already fits. Its advantage
  under rescaling is small and limited by the non-adaptive area cutoffs.
- Tuning picked `k2 = 0.01`, i.e. an almost flat h-dome test: on this data the marker spacing
  (`k1 * r`) does most of the work.

## Limitations

- Scored against boxes, not masks; box tightness differs between hand-drawn boxes and region boxes
  (hence also IoU 0.3). Precision is a lower bound because of unannotated RBCs.
- All area cutoffs and dilation radii are absolute pixel values tuned on unscaled images.
- Pale-centred RBCs whose rim has a wide gap stay ring/C-shaped; cells touching the border are ignored.
- A bilobed eosinophil can be split into two WBC regions by the rule that separates adjacent WBCs
  (seen in `segmentation_grid_test.png`); left as is because the test split had already been used.
- WBC regions are a grown convex hull of the nucleus, not a true cytoplasm boundary.
- The three BCCD classes are coarse; the near-perfect classification says little about harder tasks
  such as WBC subtyping.

## Future work

- Hand-annotate pixel masks for a subset of test images and report Dice/IoU (skipped here: needs human
  annotation).
- Scale every area cutoff and dilation with the estimated radius, then re-tune on train/val and evaluate
  the rescaled test sets once.
- A cytoplasm model for WBCs; a second-pass split of ring-shaped RBCs.

See `DECISIONS.md` for every design choice (with alternatives and reasons) and `PROGRESS.md` for the
phase log and open issues.
