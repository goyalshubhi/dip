# Progress

## Status
- [x] Phase 1: env, download, loader, stats, box overlays
- [x] Phase 2: preprocess + 4 segmenters + visual check
- [x] Phase 3: matching, tuning on train/val, segmentation eval on test
- [x] Phase 4: features + classifiers (GT crops, segmented regions)
- [x] Phase 5: interpretability + sensitivity + final figures
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

## Phase 4 (done)
- `src/features.py`: exactly 22 features (asserted), definitions in `results/tables/feature_list.csv`.
  GT boxes get a pixel mask from the segmenter's foreground/stain masks.
- `src/classify.py` + `scripts/run_classification.py --steps features,classify`: GaussianNB, KNN,
  SVM-RBF, RandomForest in StandardScaler pipelines; GridSearchCV with GroupKFold(5) over
  train+val images; test used once. Outputs `results/tables/classification_comparison.csv`,
  `results/tables/confusion_matrices.csv`, `results/tables/classifier_cv_results.csv`,
  `results/tables/segmented_regions_summary.csv`, `results/figures/confusion_matrices.png`.
- Samples (`results/logs/clf_features.log`): GT crops 3935 train+val / 944 test; segmented
  regions (watershed_fixed) matched at IoU 0.5: 2914 train+val / 680 test. No NaN/inf features.
- Test results (`results/tables/classification_comparison.csv`; e2e recall = matched by a
  region AND correctly classified, over all test GT boxes of the class):

  | setting | model | CV macro-F1 | test n | accuracy | macro-F1 | recall RBC | recall WBC | recall Platelets | e2e recall RBC | e2e recall WBC | e2e recall Platelets |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | gt_crops | GaussianNB | 0.995 | 944 | 0.997 | 0.990 | 0.999 | 1.000 | 0.971 | - | - | - |
  | gt_crops | KNN | 0.996 | 944 | 0.998 | 0.993 | 1.000 | 1.000 | 0.971 | - | - | - |
  | gt_crops | SVM-RBF | 0.997 | 944 | 0.997 | 0.990 | 1.000 | 1.000 | 0.957 | - | - | - |
  | gt_crops | RandomForest | 0.996 | 944 | 0.999 | 0.995 | 1.000 | 1.000 | 0.986 | - | - | - |
  | segmented | GaussianNB | 0.993 | 680 | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
  | segmented | KNN | 0.997 | 680 | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
  | segmented | SVM-RBF | 0.998 | 680 | 0.997 | 0.992 | 1.000 | 0.986 | 0.980 | 0.697 | 0.972 | 0.710 |
  | segmented | RandomForest | 0.998 | 680 | 0.996 | 0.988 | 0.998 | 0.986 | 0.980 | 0.695 | 0.972 | 0.710 |
  | gt_crops_model_on_segmented | GaussianNB | - | 680 | 0.990 | 0.970 | 0.998 | 0.986 | 0.900 | 0.695 | 0.972 | 0.652 |
  | gt_crops_model_on_segmented | KNN | - | 680 | 0.985 | 0.961 | 1.000 | 0.971 | 0.840 | 0.697 | 0.958 | 0.609 |
  | gt_crops_model_on_segmented | SVM-RBF | - | 680 | 0.994 | 0.985 | 1.000 | 0.971 | 0.960 | 0.697 | 0.958 | 0.696 |
  | gt_crops_model_on_segmented | RandomForest | - | 680 | 0.991 | 0.976 | 1.000 | 0.986 | 0.900 | 0.697 | 0.972 | 0.652 |

- Gap: on matched regions the classifiers are as accurate as on GT crops; the end-to-end loss
  is almost entirely segmentation (e2e RBC recall about 0.70 vs 1.00 on GT crops). GT-crop
  models applied to segmented regions lose platelet recall (0.840-0.960), because segmented
  platelet masks (dilated stain blobs) differ in shape from box-derived masks.
- Accuracy > 99% triggered leakage checks (`scripts/leakage_checks.py`,
  `results/tables/leakage_checks.csv`): 0 images shared between train+val and test; CV macro-F1
  with shuffled labels 0.256 (GT crops) / 0.249 (segmented); a depth-2 decision tree (size +
  one colour threshold) reaches test accuracy 0.989 / macro-F1 0.973 in both settings; no
  near-duplicate images across splits (max thumbnail correlation 0.637). The coarse 3-class
  task is easy with size + stain colour; no leakage found.

## Phase 5 (done)
- `scripts/run_classification.py --steps interpret` -> `results/tables/permutation_importance.csv`,
  `results/tables/class_feature_means.csv`; figures `results/figures/permutation_importance.png`,
  `results/figures/class_feature_means.png`.
- Permutation importance (SVM-RBF, GT crops, test split, drop in macro-F1): equivalent_diameter
  0.0129, mean_B 0.0072, saturation_mean 0.0041, perimeter 0.0039, area 0.0025; shape features
  circularity, solidity, eccentricity and hue_mean 0.0000.
- Interpretation: size separates WBCs (z-scored area +2.59, equivalent diameter +2.23) from
  platelets (equivalent diameter -2.05, area -1.22) with RBCs in between; stain colour
  separates both from RBCs (mean_B +1.82 WBC / +1.63 platelets, mean_R -1.99 / -1.35,
  saturation +2.79 / +0.83). Platelets are the most textured (GLCM contrast +2.27,
  homogeneity -2.54); WBCs have the highest intensity spread (gray_std +2.37, nucleus vs
  cytoplasm). Roundness features barely differ between classes. Single-feature importances
  are small because size (area / perimeter / equivalent diameter) and colour (mean_B /
  saturation / mean_R) come in redundant groups: permuting one leaves its correlates intact.
  (Sources: `results/tables/class_feature_means.csv`, `results/tables/permutation_importance.csv`.)
- Sensitivity (`scripts/run_segmentation_eval.py --steps sensitivity` -> `results/tables/sensitivity.csv`,
  `results/tables/sensitivity_per_image.csv`, `results/tables/sensitivity_area_cutoff_check.csv`,
  figure `results/figures/sensitivity.png`; 72 test images, precision is a lower bound):

  | condition | fixed F1@0.5 | adaptive F1@0.5 | fixed RBC MAE | adaptive RBC MAE | adaptive: share of images with own r |
  |---|---|---|---|---|---|
  | original | 0.649 | 0.644 | 2.514 | 2.597 | 0.639 |
  | scale 0.7 | 0.482 | 0.512 | 3.889 | 3.417 | 0.389 |
  | scale 1.3 | 0.639 | 0.640 | 2.958 | 2.736 | 0.653 |
  | hue -10 deg | 0.655 | 0.656 | 2.444 | 2.458 | 0.667 |
  | hue +10 deg | 0.650 | 0.646 | 2.569 | 2.597 | 0.625 |
  | brightness 0.8 | 0.647 | 0.639 | 2.403 | 2.389 | 0.667 |
  | brightness 1.2 | 0.637 | 0.636 | 2.431 | 2.333 | 0.597 |

  The adaptive watershed wins only when images are shrunk (0.7x: F1 +0.030, RBC MAE -0.472);
  at 1.3x and under colour shifts the two are within 0.008 F1. Both methods lose much recall
  at 0.7x because the shared, absolute RBC area cutoff tuned on unscaled images (4000 px)
  removes small cells: 46.1% of test RBC boxes have an inscribed-ellipse area below it at 0.7x
  vs 0.4% at 1.0x. The same cutoff gates the isolated-cell filter, so only 38.9% of 0.7x
  images get their own radius estimate (the rest use the unscaled training fallback), which
  limits how much the adaptive method can adapt.

## Phase 6 (in progress)
- `scripts/run_all.sh` (env, data, all steps, figures, tests; `JOBS` sets worker processes),
  `notebooks/01_explore.ipynb` (all code cells executed without error against the current
  results), `README.md` (run commands, results copied from `results/tables`, limitations,
  future work).
- Every stage has been run with the same scripts and arguments `run_all.sh` uses; the committed
  results come from those runs.
- A full end-to-end `run_all.sh` run was stopped twice by Claude Code because the laptop ran low
  on memory (other applications held most of the 16 GB). Before it was stopped, the loader,
  design-check and radius stages had regenerated `data/splits.json` and every table in
  `results/tables/` byte-for-byte identical to the committed versions.

## Next
Complete one uninterrupted `run_all.sh` run (e.g. `JOBS=2`), record its runtime, final commit.

## Open issues
- Adaptive vs fixed: no gain on plain BCCD; gain only at 0.7x, limited by the absolute RBC area
  cutoff (see Phase 5). Making all area cutoffs scale with r is the obvious next step, but it
  was not done here because it would be designed after seeing test results.
- With default parameters platelets were over-detected on train (400 predicted vs 209 GT,
  `results/tables/segmentation_design_checks.csv`); tuning raised plt_min_area to 250.
- An eosinophil with two large nuclear lobes is split into two WBC regions by the
  "two nucleus-sized blobs = two WBCs" rule (seen on test image BloodImage_00253 in
  `segmentation_grid_test.png`). Not fixed, because changing the segmenter after the test
  run would mean evaluating on test twice.
- Runs were stopped three times for low system memory (one 12-worker tuning run, two
  4-worker `run_all.sh` runs); scripts default to 4 workers and `run_all.sh` takes `JOBS`.
  The end-to-end runtime of `run_all.sh` is therefore not yet measured.
- Pale-centred RBCs whose rim has a wide gap, or that touch the border, stay ring/C-shaped.
