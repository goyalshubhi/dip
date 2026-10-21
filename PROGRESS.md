# Progress

## Status
- [x] Phase 1: env, download, loader, stats, box overlays
- [x] Phase 2: preprocess + 4 segmenters + visual check
- [ ] Phase 3: matching, tuning on train/val, segmentation eval on test
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

## Next
Phase 3: box matching (Hungarian, IoU 0.5 / 0.3), tuning of min_distance, k1, k2 (and the
platelet area cutoff / dilations) on train with val as a check, then test evaluation.

## Open issues
- On unscaled BCCD the per-image radius estimate barely varies with true cell size
  (magnification is constant), so adaptive is not expected to beat fixed on plain BCCD;
  any advantage should show up only under rescaling (Evaluation D).
- With default parameters platelets are over-detected on train (400 predicted vs 209 GT,
  `results/tables/segmentation_design_checks.csv`);
  to be addressed by tuning the platelet cutoffs on train in Phase 3.
- Pale-centred RBCs whose rim has a wide gap, or that touch the border, stay ring/C-shaped.
