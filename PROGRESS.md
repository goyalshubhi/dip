# Progress

## Status
- [x] Phase 1: env, download, loader, stats, box overlays
- [ ] Phase 2: preprocess + 4 segmenters + visual check
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

## Next
Phase 2: preprocessing and the four segmenters.

## Open issues
- None yet.
