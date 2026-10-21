"""Segmentation pipeline: training-set radius, tuning on train/val, final test evaluation.

Usage: python scripts/run_segmentation_eval.py [--steps radius,tune,test] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="design,radius,tune,test")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    set_seed()
    logger = get_logger("segmentation_eval")
    splits = load_splits()
    steps = args.steps.split(",")
    if "design" in steps:
        step_design(splits, args.jobs, logger)
    if "radius" in steps:
        step_radius(splits, args.jobs, logger)


if __name__ == "__main__":
    main()
