# Project: Watershed-Based Blood Cell Segmentation and Classification (BCCD)

Course project, VIT BCSE403L Digital Image Processing, team of three.
Title: Watershed-Based Blood Cell Segmentation and Classification on BCCD,
with an Adaptive Marker-Controlled Watershed.

## OPERATING MODE: AUTONOMOUS
You are running unattended. Build the whole project, Phases 1-6, end to end,
WITHOUT stopping to ask for approval or opinions.
- Never pause for confirmation between phases. Finish a phase, verify it,
  commit, and move straight to the next one.
- When a design choice is ambiguous, pick the most defensible option, record
  it in DECISIONS.md (choice, alternatives, reason, one line each), and keep
  going.
- When something fails, debug it yourself: read the traceback, inspect the
  data, check the library docs or source, fix, re-run. Try at least 3 distinct
  fixes before declaring a blocker.
- The ONLY reasons to stop early:
  (1) BCCD cannot be downloaded at all (no network, repo gone). Do not
      substitute another dataset; stop and report.
  (2) A required package cannot be installed.
  In either case write the blocker to PROGRESS.md and stop.
- Anything that cannot be finished for another reason: skip it, log it under
  "Open issues" in PROGRESS.md with what was tried, and continue with the
  remaining phases.
- Do not ask me questions at any point. The final summary is the only
  message I need.

## Goal
Reproducible, classical-pipeline Python project:
preprocessing -> threshold + morphology -> distance transform ->
marker-controlled watershed -> cell extraction -> 22 handcrafted features ->
classical ML classifiers -> evaluation. No deep learning.

## Dataset (do not substitute)
BCCD: https://github.com/Shenggan/BCCD_Dataset
- Expected: ~364 JPEGs (640x480), Pascal VOC XML boxes, classes
  RBC / WBC / Platelets. Treat these as expectations, not facts.
- Clone into data/ via scripts/download_data.sh (idempotent: skip if present).
- After cloning, inspect the actual folder layout and split files, and use
  what is actually there. Print and save the real counts.
- Known quirks to check and handle: some XMLs may have boxes with zero or
  negative size, coordinates outside the image, or mislabelled class names
  (case/whitespace). Clean them, and log how many were fixed or dropped.
- BCCD has NO pixel masks. Never compute Dice/IoU against masks on BCCD.
  Segmentation is scored against boxes (see Evaluation).
- Annotators often skipped RBCs touching the image border, and some dense RBC
  regions are under-annotated. Precision against BCCD boxes is therefore a
  lower bound; say so wherever precision is reported.

## Splits (no leakage, no test tuning)
- If the repo provides train/val/test lists, use them. Otherwise create an
  image-level split 70/15/15 with seed 42 and save it to data/splits.json.
- Every split is BY IMAGE, never by cell crop.
- k1, k2 (adaptive watershed), min_distance (fixed watershed), thresholds,
  area cutoffs, and classifier hyperparameters are tuned on train/val ONLY.
  The test split is touched exactly once per method for final numbers.
  Add an assertion in code that tuning functions never receive test images.

## Constraints
- Python 3.10+. Use a virtualenv at .venv. Allowed libs: numpy, pandas,
  opencv-python, scikit-image, scipy, scikit-learn, matplotlib, seaborn,
  tqdm, joblib (plus nbformat only to write the notebook). Pin versions in
  requirements.txt after confirming they install.
- Seed 42 everywhere. CPU only. Full pipeline should finish in under ~30 min
  on a laptop; if a grid search is slower, shrink the grid and log it.
- Complete files, no TODO placeholders. Practitioner style: clear names,
  comments only where the reason is non-obvious, no over-engineering.
- Every number reported anywhere (README, PROGRESS.md, final summary) must
  come from a file in results/. Never invent, estimate, or round up metrics.
- If unsure about a library API or dataset detail, check it (run it, read
  the docstring/source) instead of guessing.

## Structure
src/: loader.py preprocess.py segment.py match.py features.py classify.py
      evaluate.py utils.py
scripts/: download_data.sh run_segmentation_eval.py run_classification.py
          make_figures.py run_all.sh
tests/: test_loader.py test_segment.py test_match.py test_features.py
notebooks/01_explore.ipynb
results/figures, results/tables, results/logs
README.md, requirements.txt, PROGRESS.md, DECISIONS.md

## Pipeline spec
1. loader: XML -> DataFrame(image, class, xmin, ymin, xmax, ymax); cleaning
   as above; box-drawing helper; class counts and boxes-per-image stats saved
   to results/tables/dataset_stats.csv.
2. preprocess: green channel + LAB, CLAHE, Gaussian/median blur,
   normalisation. Return intermediates for before/after figures.
3. segment: one interface segment(image, method, **params) -> labelled mask,
   FOUR methods:
   - otsu_morph: Otsu + opening/closing + fill holes (RBCs have pale centres)
   - watershed_plain: Otsu + watershed with no markers (over-segmentation
     baseline)
   - watershed_fixed: Otsu + distance transform + peak_local_max with fixed
     min_distance + marker-controlled watershed
   - watershed_adaptive (OUR CONTRIBUTION): estimate typical cell radius r per
     image from distance-transform maxima of isolated, roughly circular
     components (filter by circularity and area; fall back to the median r
     from the training set if too few isolated cells); min_distance = k1*r,
     h-maxima height = k2*r on the smoothed distance transform. Expose k1, k2.
   WBC: segment the dark purple nucleus with its own threshold (saturation or
   a purple-sensitive channel), dilate to approximate the cell. Platelets:
   much smaller area cutoff than RBC. Each returned region carries a
   predicted coarse type (RBC-like / WBC-nucleus / platelet-sized) used only
   for analysis, not as the final class.
4. match: match region bounding boxes to GT boxes by IoU >= 0.5 using the
   Hungarian algorithm (scipy.optimize.linear_sum_assignment). Matched ->
   GT class. Unmatched region = FP, unmatched box = miss. Also report results
   at IoU 0.3 as a secondary threshold, since region boxes and hand-drawn
   boxes differ in tightness.
5. features: exactly 22 (mean R,G,B; gray mean/std/min/max; GLCM contrast,
   correlation, energy, homogeneity; area, perimeter, circularity, aspect
   ratio, compactness; Canny edge density; fill the rest with defensible
   choices such as solidity, eccentricity, equivalent diameter, hue mean,
   saturation mean). Assert the count is 22. Save the list with one-line
   definitions to results/tables/feature_list.csv.
6. classify: GaussianNB (replicates Rokhani 2026), KNN, SVM-RBF,
   RandomForest (reference). All in a Pipeline with StandardScaler;
   class_weight where supported. Hyperparameters via GroupKFold on the
   training images (group = image ID).

## Evaluation
A. Segmentation on the test split: precision/recall/F1 at IoU 0.5 (and 0.3)
   and cell-count MAE (per class + overall) for all four methods. Ignore
   regions touching the image border when scoring. One comparison table
   (results/tables/segmentation_comparison.csv) + figure grid
   (original | otsu_morph | plain | fixed | adaptive) on 4 test images,
   including at least one dense RBC cluster.
B. Classification: report accuracy, macro P/R/F1, per-class recall,
   confusion matrices (accuracy alone is misleading; RBC dominates).
   Two settings: (1) GT box crops (comparable to literature), (2) our
   segmented regions from the best segmentation method (end-to-end). Table
   results/tables/classification_comparison.csv and a short discussion of
   the gap.
C. Interpretability: permutation importance (on the test split, best model)
   + per-class mean-feature plot + a short written interpretation.
D. Sensitivity: adaptive vs fixed watershed on test images rescaled 0.7x and
   1.3x and colour-shifted (hue/brightness shift). Same metrics as A.
   Table results/tables/sensitivity.csv + line plot.
E. Skip hand-annotated Dice/IoU (needs human annotation). Mention it in
   README as future work.

## Verification gates (replace human review)
After each phase, before committing, check all of these yourself:
- Code runs end to end on real BCCD data; output is saved to results/logs.
- pytest passes for the modules touched so far.
- Open and visually inspect the saved figures (read the PNGs). Confirm,
  for example, that box overlays line up with cells, that the four
  segmentation variants actually differ on a dense RBC cluster, and that
  watershed_plain over-segments as expected.
- Sanity checks on numbers: no metric is exactly 0 or 1 unless explained;
  per-class counts add up; no NaNs in feature tables; test split size
  matches splits.json.
- If a check fails, fix and re-run before moving on. If a result looks too
  good (e.g. >99% accuracy), look for leakage before accepting it.

## Phases (run all, in order, no stopping)
1. env (.venv, requirements) + download + loader + stats + 5 box overlays
2. preprocess + 4 segmenters + visual check on 10 train images
3. matching + tune k1, k2 and fixed min_distance on train/val + segmentation
   eval on test
4. features + classifiers + both classification settings
5. interpretability + sensitivity + final figures/tables
6. scripts/run_all.sh (reproduces everything from a fresh clone),
   notebooks/01_explore.ipynb, README with exact run commands, results
   summary pulled from results/tables, limitations section.

At the end of every phase: update PROGRESS.md (done / next / open issues /
key numbers with their source file), commit with a clear message
("phase N: ..."), and continue.

## Honesty rules
- Contributions are modest extensions; never call them state of the art.
- If adaptive watershed does not beat fixed on plain BCCD, report that
  plainly. If it only wins under rescaling (Evaluation D), say exactly that.
- Report negative and mixed results as found. Do not re-tune on test to
  improve a number.
- Do not touch any report PDF, LaTeX, or Word files.

## Final summary (last message)
Keep it short: what was built, the key tables (copied from results/ files,
with file paths), what worked, what did not, open issues, every entry in
DECISIONS.md that the team should double-check, and the single command to
reproduce everything.
