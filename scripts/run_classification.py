"""Feature extraction and classification in two settings.

(1) GT box crops: one sample per cleaned BCCD box (comparable to the literature).
(2) End-to-end: regions from the best segmentation method (chosen by validation F1), labelled
    by Hungarian matching to GT boxes at IoU >= 0.5; unmatched regions have no label.
Classifiers are tuned with GroupKFold over train+val images; the test split is used once.

Usage: python scripts/run_classification.py [--steps features,classify,interpret] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from src import classify as C
from src import segment as S
from src.evaluate import TYPE_TO_CLASS, region_arrays
from src.features import FEATURE_NAMES, ImageContext, feature_list_table, mask_from_box, region_features
from src.loader import load_image
from src.match import match_boxes
from src.utils import CACHE_DIR, CLASSES, FIG_DIR, SEED, TABLE_DIR, Timer, get_logger, load_splits, set_seed

FEAT_GT = CACHE_DIR / "features_gt_crops.csv"
FEAT_SEG = CACHE_DIR / "features_segmented.csv"
MODELS_PATH = CACHE_DIR / "classifiers.joblib"
BOX_COLS = ["xmin", "ymin", "xmax", "ymax"]


def _features_one(image_id: str, split: str, gt: pd.DataFrame, method: str, params: dict) -> tuple[list, list]:
    ctx = ImageContext(load_image(image_id), **params)
    gt_rows = []
    for k, box in enumerate(gt[BOX_COLS].to_numpy()):
        feats = region_features(ctx, mask_from_box(ctx, box))
        gt_rows.append({"image": image_id, "split": split, "label": gt["class"].iloc[k],
                        **dict(zip(BOX_COLS, box)), **feats})

    seg = S.segment_prepared(ctx.prep, method, **params)
    labels, boxes, border, kinds = region_arrays(seg)
    pairs, ious = match_boxes(boxes, gt[BOX_COLS].to_numpy(), 0.5)
    matched = {int(p): (int(g), float(iou)) for (p, g), iou in zip(pairs, ious)}
    seg_rows = []
    for idx, lab in enumerate(labels):
        g, iou = matched.get(idx, (-1, 0.0))
        feats = region_features(ctx, seg.labels == lab)
        seg_rows.append({"image": image_id, "split": split,
                         "label": gt["class"].iloc[g] if g >= 0 else "none",
                         "coarse_type": kinds[idx], "touches_border": bool(border[idx]), "match_iou": iou,
                         **dict(zip(BOX_COLS, boxes[idx])), **feats})
    return gt_rows, seg_rows


def step_features(jobs: int, logger) -> None:
    with open(TABLE_DIR / "seg_params.json", encoding="utf-8") as f:
        cfg = json.load(f)
    params, method = cfg["params"], cfg["best_method_by_val_f1@0.5"]
    splits = load_splits()
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    jobs_list = [(i, s) for s in ("train", "val", "test") for i in splits[s]]
    timer = Timer()
    out = Parallel(n_jobs=jobs)(delayed(_features_one)(i, s, df[df["image"] == i], method, params)
                                for i, s in jobs_list)
    gt_feats = pd.DataFrame([r for g, _ in out for r in g])
    seg_feats = pd.DataFrame([r for _, s in out for r in s])
    for name, tab in (("GT crops", gt_feats), ("segmented", seg_feats)):
        n_bad = int((~np.isfinite(tab[FEATURE_NAMES].to_numpy())).sum())
        assert n_bad == 0, f"{n_bad} non-finite feature values in {name}"
    assert len(gt_feats) == len(df), "one feature row per cleaned GT box"
    gt_feats.to_csv(FEAT_GT, index=False)
    seg_feats.to_csv(FEAT_SEG, index=False)
    feature_list_table().to_csv(TABLE_DIR / "feature_list.csv", index=False)
    summary = (seg_feats.assign(matched=seg_feats["label"] != "none")
               .groupby(["split", "coarse_type"])["matched"].agg(["size", "sum"]).reset_index()
               .rename(columns={"size": "regions", "sum": "matched_iou0.5"}))
    summary.to_csv(TABLE_DIR / "segmented_regions_summary.csv", index=False)
    logger.info("Segmentation method for setting 2: %s", method)
    logger.info("GT-crop samples: %s", gt_feats.groupby(["split", "label"]).size().to_dict())
    logger.info("Segmented regions (all, matched at IoU 0.5):\n%s", summary.to_string(index=False))
    logger.info("Features: %d per sample, no NaN/inf; extraction took %.0f s", len(FEATURE_NAMES), timer.elapsed())


def _e2e_recall(test_regions: pd.DataFrame, y_pred: np.ndarray, gt_test: pd.DataFrame) -> dict:
    """Share of all test GT boxes of each class that were matched by a region AND classified correctly."""
    correct = test_regions["label"].to_numpy() == y_pred
    out = {}
    for c in CLASSES:
        n_gt = int((gt_test["label"] == c).sum())
        out[f"e2e_recall_{c}"] = int((correct & (test_regions["label"].to_numpy() == c)).sum()) / n_gt
    return out


def step_classify(jobs: int, logger) -> None:
    gt_feats = pd.read_csv(FEAT_GT)
    seg_feats = pd.read_csv(FEAT_SEG)
    seg_matched = seg_feats[seg_feats["label"] != "none"]
    data = {
        "gt_crops": (gt_feats[gt_feats["split"] != "test"], gt_feats[gt_feats["split"] == "test"]),
        "segmented": (seg_matched[seg_matched["split"] != "test"], seg_matched[seg_matched["split"] == "test"]),
    }
    gt_test = gt_feats[gt_feats["split"] == "test"]
    rows, cms, cv_rows, models = [], [], [], {}
    timer = Timer()
    for setting, (train_df, test_df) in data.items():
        X, y, groups = C.xy(train_df)
        Xt, yt, _ = C.xy(test_df)
        for name in C.model_grid():
            search = C.tune_and_fit(name, X, y, groups, set(groups), jobs)
            pred = search.predict(Xt)
            models[(setting, name)] = search.best_estimator_
            row = {"setting": setting, "model": name, "cv_macro_f1": search.best_score_,
                   "best_params": json.dumps(search.best_params_), "n_train": len(y), **C.metrics(yt, pred)}
            if setting == "segmented":
                row.update(_e2e_recall(test_df, pred, gt_test))
            rows.append(row)
            cm = C.confusion(yt, pred)
            cms.append(cm.assign(setting=setting, model=name))
            cv_rows.append(pd.DataFrame(search.cv_results_).assign(setting=setting, model=name))
            logger.info("%s / %s: cv macro-F1 %.3f, test acc %.3f macro-F1 %.3f (%s)", setting, name,
                        search.best_score_, row["accuracy"], row["macro_f1"], search.best_params_)

    # The GT-crop models applied to segmented test regions: isolates the region-shape shift.
    seg_test = data["segmented"][1]
    Xs, ys, _ = C.xy(seg_test)
    for name in C.model_grid():
        pred = models[("gt_crops", name)].predict(Xs)
        rows.append({"setting": "gt_crops_model_on_segmented", "model": name, "cv_macro_f1": np.nan,
                     "best_params": "", "n_train": int((gt_feats["split"] != "test").sum()),
                     **C.metrics(ys, pred), **_e2e_recall(seg_test, pred, gt_test)})
        cms.append(C.confusion(ys, pred).assign(setting="gt_crops_model_on_segmented", model=name))

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "classification_comparison.csv", index=False)
    pd.concat(cms).rename_axis("true").reset_index().to_csv(TABLE_DIR / "confusion_matrices.csv", index=False)
    pd.concat(cv_rows).drop(columns=["params"]).to_csv(TABLE_DIR / "classifier_cv_results.csv", index=False)
    best = {s: table[table["setting"] == s].sort_values("cv_macro_f1", ascending=False)["model"].iloc[0]
            for s in data}
    joblib.dump({"models": models, "best_by_cv": best}, MODELS_PATH)
    logger.info("Best model by CV macro-F1: %s", best)
    logger.info("Classification comparison (test):\n%s",
                table.drop(columns=["best_params"]).round(3).to_string(index=False))
    _plot_confusions(pd.concat(cms))
    logger.info("step_classify done in %.0f s", timer.elapsed())


def _plot_confusions(cms: pd.DataFrame) -> None:
    settings = list(dict.fromkeys(cms["setting"]))
    models = list(dict.fromkeys(cms["model"]))
    fig, axes = plt.subplots(len(settings), len(models), figsize=(4.2 * len(models), 3.6 * len(settings)))
    for i, s in enumerate(settings):
        for j, m in enumerate(models):
            cm = cms[(cms["setting"] == s) & (cms["model"] == m)][[f"pred_{c}" for c in CLASSES]].to_numpy()
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax = axes[i, j]
            annot = np.array([[f"{n}\n{r:.0%}" for n, r in zip(rn, rr)] for rn, rr in zip(cm, norm)])
            sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                        xticklabels=CLASSES, yticklabels=CLASSES)
            ax.set_title(f"{s}\n{m}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=90)
    plt.close(fig)


def step_interpret(jobs: int, logger) -> None:
    """Evaluation C: permutation importance (test split, best GT-crop model) and per-class feature means."""
    from sklearn.inspection import permutation_importance

    saved = joblib.load(MODELS_PATH)
    name = saved["best_by_cv"]["gt_crops"]
    model = saved["models"][("gt_crops", name)]
    gt_feats = pd.read_csv(FEAT_GT)
    test = gt_feats[gt_feats["split"] == "test"]
    Xt, yt, _ = C.xy(test)
    result = permutation_importance(model, Xt, yt, scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=jobs)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std, "model": name})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)
    logger.info("Permutation importance (%s, test split, drop in macro-F1, 20 repeats):\n%s", name,
                imp.round(4).to_string(index=False))

    fit = gt_feats[gt_feats["split"] != "test"]
    mu, sd = fit[FEATURE_NAMES].mean(), fit[FEATURE_NAMES].std()
    raw = fit.groupby("label")[FEATURE_NAMES].mean().reindex(CLASSES)
    z = ((raw - mu) / sd).reindex(CLASSES)
    pd.concat({"raw_mean": raw, "z_mean": z}).rename_axis(["stat", "class"]).to_csv(
        TABLE_DIR / "class_feature_means.csv")
    logger.info("Per-class z-scored feature means (train+val GT crops):\n%s", z.T.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="features,classify,interpret")
    parser.add_argument("--jobs", type=int, default=4, help="worker processes (each needs ~250 MB)")
    args = parser.parse_args()
    set_seed()
    steps = {"features": step_features, "classify": step_classify, "interpret": step_interpret}
    for name in args.steps.split(","):
        steps[name](args.jobs, get_logger(f"clf_{name}"))


if __name__ == "__main__":
    main()
