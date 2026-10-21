"""Checks run because classification accuracy came out above 99% (CLAUDE.md: look for leakage).

1. No image appears in both the training data and the test data of either setting.
2. Label-permutation control: GroupKFold CV with shuffled labels must fall to chance.
3. A depth-2 decision tree (two thresholds) on the same features: if it is almost as good,
   the classes are simply easy to separate with colour + size, not leaked.
4. Near-duplicate images across splits (tiny grayscale thumbnails, normalised correlation).

Usage: python scripts/leakage_checks.py   (after run_classification.py --steps features)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from src import classify as C
from src.features import FEATURE_NAMES
from src.loader import load_image
from src.utils import CACHE_DIR, SEED, TABLE_DIR, get_logger, load_splits, set_seed


def main() -> None:
    set_seed()
    logger = get_logger("leakage_checks")
    rows = []
    gt = pd.read_csv(CACHE_DIR / "features_gt_crops.csv")
    seg = pd.read_csv(CACHE_DIR / "features_segmented.csv")
    seg = seg[seg["label"] != "none"]
    splits = load_splits()

    for setting, df in (("gt_crops", gt), ("segmented", seg)):
        train, test = df[df["split"] != "test"], df[df["split"] == "test"]
        overlap = set(train["image"]) & set(test["image"])
        assert set(test["image"]) <= set(splits["test"])
        rows.append({"setting": setting, "check": "images_in_both_train_and_test", "value": len(overlap)})

        X, y, groups = C.xy(train)
        Xt, yt, _ = C.xy(test)
        svm = Pipeline([("scale", StandardScaler()), ("clf", SVC(class_weight="balanced", random_state=SEED))])
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(y)
        cv = GroupKFold(n_splits=C.N_FOLDS)
        perm_f1 = cross_val_score(svm, X, shuffled, groups=groups, cv=cv, scoring="f1_macro").mean()
        rows.append({"setting": setting, "check": "cv_macro_f1_with_shuffled_labels", "value": round(perm_f1, 3)})
        majority = pd.Series(y).value_counts(normalize=True).iloc[0]
        rows.append({"setting": setting, "check": "majority_class_share_train", "value": round(majority, 3)})

        tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED).fit(X, y)
        pred = tree.predict(Xt)
        rows.append({"setting": setting, "check": "depth2_tree_test_accuracy", "value": round((pred == yt).mean(), 3)})
        rows.append({"setting": setting, "check": "depth2_tree_test_macro_f1",
                     "value": round(f1_score(yt, pred, average="macro"), 3)})
        logger.info("%s depth-2 tree:\n%s", setting, export_text(tree, feature_names=FEATURE_NAMES))

    # near-duplicate images: 32x24 grayscale thumbnails, Pearson correlation between splits
    ids = {s: splits[s] for s in ("train", "val", "test")}
    thumbs = {}
    for s, lst in ids.items():
        for i in lst:
            g = cv2.cvtColor(load_image(i), cv2.COLOR_RGB2GRAY)
            t = cv2.resize(g, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
            thumbs[i] = (t - t.mean()) / (t.std() + 1e-9)
    test_mat = np.array([thumbs[i] for i in ids["test"]])
    fit_ids = ids["train"] + ids["val"]
    fit_mat = np.array([thumbs[i] for i in fit_ids])
    corr = test_mat @ fit_mat.T / test_mat.shape[1]
    best = corr.max(axis=1)
    for thr in (0.95, 0.98):
        rows.append({"setting": "images", "check": f"test_images_with_train_val_thumbnail_corr_gt_{thr}",
                     "value": int((best > thr).sum())})
    rows.append({"setting": "images", "check": "max_test_to_train_val_thumbnail_corr", "value": round(float(best.max()), 3)})
    worst = np.argsort(-best)[:5]
    logger.info("Most similar test/train-val pairs: %s",
                [(ids["test"][k], fit_ids[int(corr[k].argmax())], round(float(best[k]), 3)) for k in worst])

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "leakage_checks.csv", index=False)
    logger.info("Leakage checks:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
