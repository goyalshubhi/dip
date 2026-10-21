# Progress

## Status
- [x] Phase 1: env, download, loader, stats, box overlays
- [x] Phase 2: preprocess + 4 segmenters + visual check
- [x] Phase 3: matching, tuning on train/val, segmentation eval on test
- [ ] Phase 4: features + classifiers (GT crops, segmented regions)
- [ ] Phase 5: interpretability + sensitivity + final figures
- [ ] Phase 6: run_all.sh, notebook, README

## Phase 1 (done)
- `.venv` (Python 3.11.9), pinned `requirements.txt`.
- `scripts/download_data.sh` clones BCCD into `data/BCCD_Dataset` (idempotent).
- `python -m src.loader` writes `data/splits.json`, `results/tables/dataset_stats.csv`,
  `results/tables/box_size_stats.csv`, `results/tables/annotations_clean.csv`,
  `results/logs/cleaning_log.json`, `results/figures/gt_box_overlays.png`.
- Key numbers (source: `results/logs/cleaning_log.json`, `results/tables/dataset_stats.csv`):
  364 JPEGs, 364 XMLs; 4888 raw boxes -> 4879 clean (2 zero-size dropped,
  7 near-duplicates dropped, 0 class names fixed, 0 clipped).
  Clean counts: RBC 4146, WBC 372, Platelets 361. Split: train 205 / val 87 / test 72 images
  (test: 804 RBC, 71 WBC, 69 Platelets).
- Visual check: boxes align with cells; many RBCs (especially at borders and in dense regions)
  are not annotated, so precision against BCCD boxes is a lower bound.

## Phase 2 (done)
- `src/preprocess.py`: green, LAB, HSV saturation, CLAHE, Gaussian/median blur, min-max
  normalisation, and a B - R "purple" stain channel; returns all intermediates.
- `src/segment.py`: `segment(rgb, method, **params) -> Segmentation(labels, types, info)` with
  `otsu_morph`, `watershed_plain`, `watershed_fixed`, `watershed_adaptive`. Shared Otsu
  foreground + morphology + purple-stain path (WBC nucleus -> convex hull grown; platelets);
  methods differ only in how the RBC-like foreground is split. Every region carries a coarse
  type (RBC-like / WBC-nucleus / platelet-sized).
- `scripts/run_segmentation_eval.py --steps design,radius` -> `results/tables/segmentation_design_checks.csv`
  (train-set evidence for channel/hole-filling choices), `results/tables/train_radius.json`
  (fallback r = 41.54 px from 432 isolated cells on 205 training images) and
  `results/tables/radius_estimator_check.csv` (72.2% of training images have at least one
  isolated round RBC; per-image r follows image scale: median ratio 0.70 at 0.7x, 1.30 at 1.3x).
- `scripts/make_figures.py` -> `results/figures/preprocess_steps.png`,
  `results/figures/segmentation_check_train.png` (10 train images, default parameters).
- Visual check: on the densest training image (BloodImage_00038) the methods give
  5 / 200 / 32 / 16 RBC-like regions (otsu_morph / plain / fixed / adaptive, default params,
  `results/logs/make_figures.log`); watershed_plain over-segments as expected; fixed with
  min_distance 20 over-splits; otsu_morph merges clumps.
- `tests/test_segment.py` (synthetic smears): 16/16 tests pass with `tests/test_loader.py`.

## Phase 3 (done)
- `src/match.py`: IoU matrix, Hungarian matching (max #matches at IoU >= thr, then total IoU),
  per-image scoring with border handling (border regions dropped; border GT boxes are
  "don't care"). `src/evaluate.py`: per-image counts, micro P/R/F1 at IoU 0.5 and 0.3,
  per-class recall, per-class and overall count MAE, and the train/val tuning grids.
- `scripts/run_segmentation_eval.py --steps tune,test` -> `results/tables/seg_tuning_shared.csv`,
  `results/tables/seg_tuning_splitting.csv`, `results/tables/segmentation_val.csv`,
  `results/tables/seg_params.json`, `results/tables/segmentation_comparison.csv`,
  `results/tables/segmentation_test_per_image.csv`; figure `results/figures/segmentation_grid_test.png`.
- Tuned on train (val as check; `results/logs/seg_tune.log`): wbc_dilate 15, plt_min_area 250,
  plt_dilate 9, rbc_min_area 4000; fixed min_distance 36 (train F1 0.663, val 0.670); adaptive
  k1 0.7, k2 0.01 (train F1 0.661, val 0.669). Best method on val: watershed_fixed.
- Test split, 72 images, each method run once (`results/tables/segmentation_comparison.csv`;
  precision is a lower bound because BCCD under-annotates RBCs):

  | method | P@0.5 | R@0.5 | F1@0.5 | F1@0.3 | RBC count MAE | overall count MAE |
  |---|---|---|---|---|---|---|
  | otsu_morph | 0.728 | 0.354 | 0.476 | 0.549 | 5.264 | 5.389 |
  | watershed_plain | 0.032 | 0.374 | 0.058 | 0.116 | 107.319 | 107.333 |
  | watershed_fixed | 0.646 | 0.652 | 0.649 | 0.733 | 2.514 | 2.806 |
  | watershed_adaptive | 0.630 | 0.659 | 0.644 | 0.730 | 2.597 | 2.917 |

  WBC recall@0.5 0.900 and platelet recall@0.5 0.705 are identical for all methods (shared
  stain path). On unscaled BCCD the adaptive watershed does **not** beat the fixed one
  (F1 0.644 vs 0.649); 46 of 72 test images got a per-image radius, 26 used the training fallback
  (`results/tables/segmentation_test_per_image.csv`).
- Visual check (`segmentation_grid_test.png`): on the densest test cluster (BloodImage_00018,
  21 GT RBCs) the methods give 2 / 177 / 19 / 19 RBC-like regions; plain over-segments every image.
- 27 tests pass (`tests/test_loader.py`, `test_segment.py`, `test_match.py`, `test_features.py`).

## Next
Phase 4: 22 features, four classifiers, GT-crop and end-to-end settings
(`scripts/run_classification.py`, end-to-end with watershed_fixed, the best method on val).

## Open issues
- On unscaled BCCD the per-image radius estimate barely varies with true cell size
  (magnification is constant), so adaptive is not expected to beat fixed on plain BCCD;
  any advantage should show up only under rescaling (Evaluation D).
- With default parameters platelets were over-detected on train (400 predicted vs 209 GT,
  `results/tables/segmentation_design_checks.csv`); tuning raised plt_min_area to 250.
- An eosinophil with two large nuclear lobes is split into two WBC regions by the
  "two nucleus-sized blobs = two WBCs" rule (seen on test image BloodImage_00253 in
  `segmentation_grid_test.png`). Not fixed, because changing the segmenter after the test
  run would mean evaluating on test twice.
- A 12-worker tuning run was killed by the OS for low memory; scripts now default to 4 workers.
- Pale-centred RBCs whose rim has a wide gap, or that touch the border, stay ring/C-shaped.
