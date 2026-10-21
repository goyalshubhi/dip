"""Figures for the report. Each figure is a function; run all or pick some with --only.

Usage: python scripts/make_figures.py [--only preprocess,seg_train]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from src import segment as S
from src.loader import draw_boxes, load_image
from src.preprocess import preprocess
from src.utils import FIG_DIR, SEED, TABLE_DIR, get_logger, load_splits

TYPE_COLORS = {S.RBC: (0.95, 0.85, 0.1), S.WBC: (0.2, 0.45, 1.0), S.PLATELET: (0.1, 0.85, 0.3)}


def overlay(rgb: np.ndarray, seg: S.Segmentation) -> np.ndarray:
    """Regions tinted with random colours, boundaries coloured by coarse type."""
    vis = label2rgb(seg.labels, rgb, alpha=0.3, bg_label=0)
    for label, kind in seg.types.items():
        edge = find_boundaries(seg.labels == label, mode="inner")
        vis[edge] = TYPE_COLORS[kind]
    return vis


def fig_preprocess(logger) -> None:
    image_id = load_splits()["train"][0]
    pre = preprocess(load_image(image_id))
    panels = [("rgb", "original RGB", None), ("green", "green channel", "gray"), ("lab_L", "LAB L*", "gray"),
              ("clahe", "CLAHE (green)", "gray"), ("normalized", "blur + min-max", "gray"),
              ("purple", "B - R (stain)", "magma")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(pre[key], cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Preprocessing steps ({image_id})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "preprocess_steps.png", dpi=110)
    plt.close(fig)
    logger.info("saved preprocess_steps.png (%s)", image_id)


def fig_seg_train(logger, n: int = 10) -> None:
    """Visual check of the four segmenters with default parameters on n training images."""
    splits = load_splits()
    df = pd.read_csv(TABLE_DIR / "annotations_clean.csv")
    rng = np.random.default_rng(SEED)
    ids = sorted(rng.choice(splits["train"], size=n, replace=False).tolist())
    # include the training image with the most RBC boxes as a dense-cluster case
    rbc_counts = df[(df["class"] == "RBC") & df["image"].isin(splits["train"])].groupby("image").size()
    densest = rbc_counts.idxmax()
    if densest not in ids:
        ids[-1] = densest
    fig, axes = plt.subplots(len(ids), 5, figsize=(20, 3.1 * len(ids)))
    rows = []
    for row, image_id in zip(axes, ids):
        rgb = load_image(image_id)
        prep = S.prepare(rgb)
        gt = df[df["image"] == image_id]
        row[0].imshow(draw_boxes(rgb, gt, label=False))
        row[0].set_title(f"{image_id}  GT R/W/P = {(gt['class'] == 'RBC').sum()}/"
                         f"{(gt['class'] == 'WBC').sum()}/{(gt['class'] == 'Platelets').sum()}", fontsize=9)
        for ax, method in zip(row[1:], S.METHODS):
            seg = S.segment_prepared(prep, method)
            counts = {k: sum(v == k for v in seg.types.values()) for k in TYPE_COLORS}
            ax.imshow(overlay(rgb, seg))
            extra = f", r={seg.info['r']:.0f} ({seg.info['r_source']})" if "r" in seg.info else ""
            ax.set_title(f"{method}: R/W/P = {counts[S.RBC]}/{counts[S.WBC]}/{counts[S.PLATELET]}{extra}", fontsize=9)
            rows.append({"image": image_id, "method": method, **counts})
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "segmentation_check_train.png", dpi=70)
    plt.close(fig)
    logger.info("saved segmentation_check_train.png for %s", ids)
    logger.info("region counts (default params, train images):\n%s",
                pd.DataFrame(rows).pivot_table(index="image", columns="method", values=S.RBC).to_string())


FIGURES = {"preprocess": fig_preprocess, "seg_train": fig_seg_train}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(FIGURES))
    args = parser.parse_args()
    logger = get_logger("make_figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        FIGURES[name](logger)


if __name__ == "__main__":
    main()
