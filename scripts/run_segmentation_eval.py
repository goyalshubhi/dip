"""Segmentation pipeline: training-set radius, tuning on train/val, final test evaluation.

Usage: python scripts/run_segmentation_eval.py [--steps design,radius,tune,test,sensitivity] [--jobs N] [--reuse]
Each step logs to results/logs/seg_<step>.log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import evaluate as E
from src import segment as S
from src.loader import load_image
from src.utils import TABLE_DIR, Timer, assert_not_test, get_logger, load_splits, set_seed


def _radius_check_one(image_id: str, p: dict) -> list[dict]:
    rgb = load_image(image_id)
    rows = []
    for scale in (1.0, 0.7, 1.3):
        img = rgb if scale == 1.0 else cv2.resize(
            rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        radii = S.isolated_cell_radii(S.prepare(img, **p).rbc_mask, p)
        rows.append({"image": image_id, "scale": scale, "n_isolated": len(radii),
                     "r_image": float(np.median(radii)) if len(radii) else np.nan})
    return rows


def step_radius(splits: dict, jobs: int, logger) -> None:
    """Fallback radius from training images, plus a check that the per-image estimate follows image scale."""
    train = splits["train"]
    assert_not_test(train, "step_radius")
    timer = Timer()
    rad = S.estimate_train_radius(train)
    with open(S.TRAIN_RADIUS_PATH, "w", encoding="utf-8") as f:
        json.dump(rad, f, indent=1)
    logger.info("Training radius: %s", rad)

    p = {**S.DEFAULTS, "fallback_r": rad["fallback_r"]}
    rows = Parallel(n_jobs=jobs)(delayed(_radius_check_one)(i, p) for i in train)
    per_image = pd.DataFrame([r for chunk in rows for r in chunk])
    base = per_image[per_image.scale == 1.0].set_index("image")["r_image"]
    summary = []
    for scale, grp in per_image.groupby("scale"):
        g = grp.set_index("image")
        ratio = (g["r_image"] / base).dropna()
        summary.append({
            "split": "train", "scale": scale, "images": len(g),
            "share_images_with_isolated_cell": round(float((g["n_isolated"] >= p["min_isolated"]).mean()), 3),
            "isolated_cells_per_image_mean": round(float(g["n_isolated"].mean()), 3),
            "r_image_median": round(float(g["r_image"].median()), 2),
            "r_ratio_to_unscaled_median": round(float(ratio.median()), 3) if len(ratio) else np.nan,
            "r_ratio_p10": round(float(ratio.quantile(0.1)), 3) if len(ratio) else np.nan,
            "r_ratio_p90": round(float(ratio.quantile(0.9)), 3) if len(ratio) else np.nan,
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(TABLE_DIR / "radius_estimator_check.csv", index=False)
    logger.info("Radius estimator check (train):\n%s", summary.to_string(index=False))
    logger.info("step_radius done in %.1f s", timer.elapsed())


HOLE_VARIANTS = {"close3": (3, 0), "close5": (5, 0), "close3+holes7": (3, 7)}


def _design_check_one(image_id: str, boxes: pd.DataFrame) -> dict:
    rgb = load_image(image_id)
    pre = S.preprocess(rgb)
    out = {"image": image_id, "values": [], "holes": []}
    fg, _ = S.foreground_mask(pre["normalized"], 3, 7, 3)
    out["values"].append(("background", float(np.median(pre["green"][~fg])), float(np.median(pre["purple"][~fg]))))
    for b in boxes.itertuples():
        cy, cx = (b.ymin + b.ymax) // 2, (b.xmin + b.xmax) // 2
        hh, hw = max(1, (b.ymax - b.ymin) // 6), max(1, (b.xmax - b.xmin) // 6)
        win = (slice(cy - hh, cy + hh + 1), slice(cx - hw, cx + hw + 1))
        out["values"].append((b._2, float(np.median(pre["green"][win])), float(np.median(pre["purple"][win]))))
    rbc = boxes[boxes["class"] == "RBC"]
    cy = ((rbc.ymin + rbc.ymax) // 2).to_numpy()
    cx = ((rbc.xmin + rbc.xmax) // 2).to_numpy()
    for name, (close_r, hole_r) in HOLE_VARIANTS.items():
        m, _ = S.foreground_mask(pre["normalized"], close_r, hole_r, 3)
        out["holes"].append((name, float(m[cy, cx].mean()) if len(rbc) else np.nan,
                             int(cv2.connectedComponents(m.astype(np.uint8))[0] - 1)))
    prep = S.prepare(rgb)
    out["counts"] = (len(prep.wbc_regions), int((boxes["class"] == "WBC").sum()),
                     len(prep.plt_regions), int((boxes["class"] == "Platelets").sum()))
    return out


def step_design(splits: dict, jobs: int, logger) -> None:
    """Training-set evidence behind the default channel, hole-filling and stain-path choices."""
    train = splits["train"]
    assert_not_test(train, "step_design")
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    res = Parallel(n_jobs=jobs)(delayed(_design_check_one)(i, df[df["image"] == i]) for i in train)
    rows = []
    vals = pd.DataFrame([v for r in res for v in r["values"]], columns=["region", "green", "purple"])
    for region, g in vals.groupby("region"):
        for ch in ("green", "purple"):
            rows.append({"check": f"{ch}_median_at_box_centre", "variant": region,
                         "value": round(float(g[ch].median()), 2), "n": len(g)})
    holes = pd.DataFrame([h for r in res for h in r["holes"]], columns=["variant", "centre_cov", "n_comp"])
    for name, g in holes.groupby("variant", sort=False):
        rows.append({"check": "share_rbc_centres_in_mask", "variant": name,
                     "value": round(float(g["centre_cov"].mean()), 3), "n": len(g)})
        rows.append({"check": "fg_components_per_image", "variant": name,
                     "value": round(float(g["n_comp"].mean()), 2), "n": len(g)})
    counts = np.array([r["counts"] for r in res])
    for k, name in enumerate(["wbc_regions_default", "wbc_gt", "platelet_regions_default", "platelet_gt"]):
        rows.append({"check": "train_total_count", "variant": name, "value": int(counts[:, k].sum()), "n": len(res)})
    rows.append({"check": "share_images_wbc_count_exact", "variant": "default", "n": len(res),
                 "value": round(float((counts[:, 0] == counts[:, 1]).mean()), 3)})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "segmentation_design_checks.csv", index=False)
    logger.info("Design checks (train):\n%s", table.to_string(index=False))


# Grids were widened until every optimum was interior, then trimmed to the region around the
# optima to keep the full pipeline near 30 min (DECISIONS.md, Phase 3).
SHARED_GRID = {
    "wbc_dilate": [9, 12, 15, 18, 21],
    "plt_min_area": [160, 200, 250, 300, 400],
    "plt_dilate": [5, 7, 9, 11],
    "rbc_min_area": [1500, 2000, 3000, 4000, 5000],
}
FIXED_GRID = [16, 20, 24, 28, 32, 36, 40, 48]
K1_GRID = [0.5, 0.6, 0.7, 0.8, 1.0]
K2_GRID = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]   # k2 = 0 switches the h-dome test off
PARAMS_PATH = TABLE_DIR / "seg_params.json"
SHARED_TABLE = TABLE_DIR / "seg_tuning_shared.csv"
SPLIT_TABLE = TABLE_DIR / "seg_tuning_splitting.csv"


def params_tag(params: dict) -> str:
    """Short hash of the parameters a tuning row was computed with."""
    blob = json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _grid_table(fn, keys: list[str], wanted: pd.DataFrame, path: Path, reuse: bool, splits: dict,
                df: pd.DataFrame, jobs: int, make_args, tag: str, logger) -> pd.DataFrame:
    """Train/val TP/FP/FN/P/R/F1 per config. With reuse, rows already in `path` that were computed
    with the same parameters (same tag) are kept and only missing configs are computed; the
    results are deterministic, so this equals a full run."""
    old = None
    if reuse and path.exists():
        old = pd.read_csv(path, dtype={k: str for k in keys})
        old = old[old["params_tag"].astype(str) == tag] if "params_tag" in old else old.iloc[:0]
    todo = wanted if old is None else wanted.merge(old[keys], on=keys, how="left", indicator=True).query(
        "_merge == 'left_only'")[keys]
    logger.info("%s: %d configs wanted, %d to compute", path.name, len(wanted), len(todo))
    parts = {}
    for name in ("train", "val"):
        if len(todo):
            rows = Parallel(n_jobs=jobs)(delayed(fn)(i, df[df["image"] == i], *make_args(todo))
                                         for i in splits[name])
            parts[name] = E.aggregate_grid(pd.DataFrame([r for chunk in rows for r in chunk]), keys)
    new = parts["train"].merge(parts["val"], on=keys, suffixes=("_train", "_val")).assign(params_tag=tag) if parts else None
    table = pd.concat([t for t in (old, new) if t is not None], ignore_index=True)
    for k in keys:
        table[k] = table[k].astype(str)
    table = wanted.astype(str).merge(table, on=keys, how="left")
    assert table["f1_train"].notna().all()
    table.to_csv(path, index=False)
    return table


def step_tune(splits: dict, jobs: int, logger, reuse: bool = False) -> dict:
    """Stage A: shared stain-path/area parameters (class-specific F1). Stage B: min_distance and
    (k1, k2) by overall F1 at IoU 0.5. Selection uses train only; val is reported as a check."""
    train, val = splits["train"], splits["val"]
    assert_not_test(train + val, "step_tune")
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    with open(S.TRAIN_RADIUS_PATH, encoding="utf-8") as f:
        fallback_r = json.load(f)["fallback_r"]
    base = {**S.DEFAULTS, "fallback_r": fallback_r}
    timer = Timer()

    # Stage A
    configs = E.shared_configs(SHARED_GRID)
    keys = ["param", "value", "class"]
    wanted = pd.DataFrame([(c[0], c[1], c[4]) for c in configs], columns=keys)
    by_key = {(c[0], c[1], c[4]): c for c in configs}

    def stage_a_args(todo):
        return ([by_key[tuple(r)] for r in todo.itertuples(index=False)], base)

    shared = _grid_table(E.stain_path_grid_one, keys, wanted, SHARED_TABLE, reuse, splits, df, jobs,
                         stage_a_args, params_tag(base), logger)
    chosen = {}
    for param, g in shared.groupby("param", sort=False):
        best = g.loc[g["f1_train"].idxmax()]
        for name, v in zip(param.split(","), str(best["value"]).split(",")):
            chosen[name] = int(float(v))
        logger.info("Stage A %s -> %s (train F1 %.3f, val F1 %.3f)", param, best["value"], best["f1_train"], best["f1_val"])
    params = {**base, **chosen}
    logger.info("Stage A done in %.0f s", timer.elapsed())

    # Stage B
    keys = ["method", "config"]
    wanted = pd.DataFrame(E.splitting_configs(FIXED_GRID, K1_GRID, K2_GRID), columns=keys)
    split_tab = _grid_table(E.splitting_grid_one, keys, wanted, SPLIT_TABLE, reuse, splits, df, jobs,
                            lambda todo: (params, list(todo.itertuples(index=False, name=None))),
                            params_tag(params), logger)
    for method, g in split_tab.groupby("method", sort=False):
        best = g.loc[g["f1_train"].idxmax()]
        for item in best["config"].split(","):
            name, v = item.split("=")
            params[name] = int(v) if name == "min_distance" else float(v)
        logger.info("Stage B %s -> %s (train F1 %.3f, val F1 %.3f)", method, best["config"], best["f1_train"], best["f1_val"])
    logger.info("Stage B done in %.0f s", timer.elapsed())

    # Validation scores of all four tuned methods pick the method used end-to-end in Phase 4.
    val_rows = []
    for method in S.METHODS:
        per_image = pd.DataFrame(Parallel(n_jobs=jobs)(
            delayed(_eval_one)(i, df[df["image"] == i], method, params) for i in val))
        val_rows.append({"method": method, **E.summarize(per_image)})
    val_tab = pd.DataFrame(val_rows)
    val_tab.to_csv(TABLE_DIR / "segmentation_val.csv", index=False)
    best_method = val_tab.loc[val_tab["f1@0.5"].idxmax(), "method"]
    logger.info("Validation (tuned params):\n%s", val_tab.round(3).to_string(index=False))
    out = {"params": dict(params), "best_method_by_val_f1@0.5": best_method,
           "selected_on": "train F1@0.5 (val reported as a check)"}
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    logger.info("Saved %s; best method on val: %s; tuning took %.0f s", PARAMS_PATH.name, best_method, timer.elapsed())
    return out


def _eval_one(image_id: str, gt: pd.DataFrame, method: str, params: dict) -> dict:
    seg = S.segment(load_image(image_id), method, **params)
    return {"image": image_id, "method": method, **E.evaluate_image(seg, gt),
            **({"r": seg.info["r"], "r_source": seg.info["r_source"]} if "r" in seg.info else {})}


def step_test(splits: dict, jobs: int, logger) -> None:
    """Final numbers: each tuned method runs once on the test split."""
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)["params"]
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    test = splits["test"]
    timer = Timer()
    per_image, rows = [], []
    for method in S.METHODS:
        res = pd.DataFrame(Parallel(n_jobs=jobs)(
            delayed(_eval_one)(i, df[df["image"] == i], method, params) for i in test))
        per_image.append(res)
        rows.append({"method": method, **E.summarize(res)})
    per_image = pd.concat(per_image, ignore_index=True)
    per_image.to_csv(TABLE_DIR / "segmentation_test_per_image.csv", index=False)
    table = pd.DataFrame(rows)
    assert (table["images"] == len(test)).all()
    table.to_csv(TABLE_DIR / "segmentation_comparison.csv", index=False)
    logger.info("Test segmentation comparison (%d images; precision is a lower bound, BCCD under-annotates "
                "RBCs):\n%s", len(test), table.round(3).T.to_string())
    logger.info("step_test done in %.0f s", timer.elapsed())


SENSITIVITY_CONDITIONS = {
    "original": ("scale", 1.0),
    "scale_0.7": ("scale", 0.7),
    "scale_1.3": ("scale", 1.3),
    "hue_-10deg": ("hue", -10),
    "hue_+10deg": ("hue", 10),
    "brightness_0.8": ("brightness", 0.8),
    "brightness_1.2": ("brightness", 1.2),
}


def perturb(rgb: np.ndarray, gt: pd.DataFrame, kind: str, value: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply one perturbation to an image and (for rescaling) its GT boxes."""
    gt = gt.copy()
    if kind == "scale":
        if value == 1.0:
            return rgb, gt
        h, w = rgb.shape[:2]
        new_w, new_h = int(round(w * value)), int(round(h * value))
        interp = cv2.INTER_AREA if value < 1 else cv2.INTER_LINEAR
        out = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
        sx, sy = new_w / w, new_h / h
        for c, s, lim in (("xmin", sx, new_w), ("xmax", sx, new_w), ("ymin", sy, new_h), ("ymax", sy, new_h)):
            # pixel centres map as (i + 0.5) * s - 0.5
            gt[c] = np.clip(np.round((gt[c] + 0.5) * s - 0.5), 0, lim - 1).astype(int)
        return out, gt
    if kind == "hue":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(int) + int(round(value / 2))) % 180).astype(np.uint8)  # OpenCV hue = deg / 2
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), gt
    if kind == "brightness":
        return np.clip(rgb.astype(np.float32) * value, 0, 255).astype(np.uint8), gt
    raise ValueError(kind)


def _sensitivity_one(image_id: str, gt: pd.DataFrame, params: dict) -> list[dict]:
    rgb = load_image(image_id)
    rows = []
    for cond, (kind, value) in SENSITIVITY_CONDITIONS.items():
        img, boxes = perturb(rgb, gt, kind, value)
        prep = S.prepare(img, **params)
        for method in ("watershed_fixed", "watershed_adaptive"):
            seg = S.segment_prepared(prep, method, **params)
            rows.append({"image": image_id, "condition": cond, "kind": kind, "value": value, "method": method,
                         **E.evaluate_image(seg, boxes),
                         **({"r": seg.info["r"], "r_source": seg.info["r_source"]} if "r" in seg.info else {})})
    return rows


def step_sensitivity(splits: dict, jobs: int, logger) -> None:
    """Evaluation D: adaptive vs fixed watershed on perturbed test images (same tuned parameters)."""
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)["params"]
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    test = splits["test"]
    timer = Timer()
    res = Parallel(n_jobs=jobs)(delayed(_sensitivity_one)(i, df[df["image"] == i], params) for i in test)
    per_image = pd.DataFrame([r for chunk in res for r in chunk])
    per_image.to_csv(TABLE_DIR / "sensitivity_per_image.csv", index=False)
    rows = []
    for (cond, method), g in per_image.groupby(["condition", "method"], sort=False):
        extra = {}
        if method == "watershed_adaptive":
            extra = {"r_median": float(g["r"].median()),
                     "share_r_from_image": float((g["r_source"] == "image").mean())}
        rows.append({"condition": cond, "kind": g["kind"].iloc[0], "value": g["value"].iloc[0], "method": method,
                     **E.summarize(g), **extra})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "sensitivity.csv", index=False)

    # The shared RBC area cutoff is in absolute pixels; how many GT RBCs fall under it per scale?
    rbc = df[df["image"].isin(test) & (df["class"] == "RBC")]
    ellipse_area = np.pi / 4 * (rbc["xmax"] - rbc["xmin"] + 1) * (rbc["ymax"] - rbc["ymin"] + 1)
    cutoff_rows = [{"scale": s, "rbc_min_area": params["rbc_min_area"], "test_rbc_boxes": len(rbc),
                    "share_rbc_box_ellipse_area_below_cutoff": round(float((ellipse_area * s ** 2 < params["rbc_min_area"]).mean()), 3)}
                   for s in (0.7, 1.0, 1.3)]
    cutoff = pd.DataFrame(cutoff_rows)
    cutoff.to_csv(TABLE_DIR / "sensitivity_area_cutoff_check.csv", index=False)
    logger.info("Test RBC boxes whose inscribed-ellipse area is below rbc_min_area:\n%s", cutoff.to_string(index=False))
    cols = ["condition", "method", "precision@0.5", "recall@0.5", "f1@0.5", "f1@0.3", "count_mae_RBC",
            "count_mae_overall", "r_median", "share_r_from_image"]
    logger.info("Sensitivity on %d test images (precision is a lower bound):\n%s", len(test),
                table[cols].round(3).to_string(index=False))
    logger.info("step_sensitivity done in %.0f s", timer.elapsed())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="design,radius,tune,test,sensitivity")
    parser.add_argument("--jobs", type=int, default=4, help="worker processes (each needs ~250 MB)")
    parser.add_argument("--reuse", action="store_true", help="tune: keep configs already in the tuning tables")
    args = parser.parse_args()
    set_seed()
    splits = load_splits()
    steps = {"design": step_design, "radius": step_radius, "tune": step_tune, "test": step_test,
             "sensitivity": step_sensitivity}
    for name in args.steps.split(","):
        logger = get_logger(f"seg_{name}")
        if name == "tune":
            step_tune(splits, args.jobs, logger, reuse=args.reuse)
        else:
            steps[name](splits, args.jobs, logger)


if __name__ == "__main__":
    main()
