"""Segmentation scoring against BCCD boxes and the train-only tuning grids.

Precision against BCCD boxes is a lower bound: annotators skipped many RBCs (especially in
dense regions), so some "false positives" are real, unannotated cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import segment as S
from .loader import load_image
from .match import prf, score_image
from .utils import CLASSES, assert_not_test

TYPE_TO_CLASS = {S.RBC: "RBC", S.WBC: "WBC", S.PLATELET: "Platelets"}
IOU_THRESHOLDS = (0.5, 0.3)


def region_arrays(seg: S.Segmentation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(labels, inclusive boxes (N, 4), touches_border (N,), coarse class names (N,)) for every region."""
    h, w = seg.labels.shape
    labels, boxes = [], []
    for i, sl in enumerate(ndi.find_objects(seg.labels), start=1):
        if sl is None:
            continue
        labels.append(i)
        boxes.append((sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1))
    boxes = np.array(boxes, int).reshape(-1, 4)
    border = (boxes[:, 0] == 0) | (boxes[:, 1] == 0) | (boxes[:, 2] == w - 1) | (boxes[:, 3] == h - 1)
    kinds = np.array([TYPE_TO_CLASS[seg.types[l]] for l in labels], dtype=object)
    return np.array(labels, int), boxes, border, kinds


def evaluate_image(seg: S.Segmentation, gt: pd.DataFrame, iou_thresholds=IOU_THRESHOLDS) -> dict:
    """Per-image counts for scoring: TP/FP/FN per IoU threshold, per-class TP, interior counts."""
    _, boxes, border, kinds = region_arrays(seg)
    gt_boxes = gt[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    gt_classes = gt["class"].to_numpy()
    row = {}
    for thr in iou_thresholds:
        s = score_image(boxes, border, gt_boxes, gt_classes, seg.labels.shape, thr)
        row.update({f"tp@{thr}": s.tp, f"fp@{thr}": s.fp, f"fn@{thr}": s.fn})
        for c in CLASSES:
            row[f"tp_{c}@{thr}"] = s.tp_by_class[c]
            row[f"gt_{c}"] = s.gt_by_class[c]
    for c in CLASSES:
        row[f"pred_{c}"] = int(((kinds == c) & ~border).sum())
    return row


def summarize(per_image: pd.DataFrame, iou_thresholds=IOU_THRESHOLDS) -> dict:
    """Micro-averaged P/R/F1 per IoU threshold, per-class recall, and per-image count MAE."""
    out = {"images": len(per_image)}
    for thr in iou_thresholds:
        tp, fp, fn = (int(per_image[f"{k}@{thr}"].sum()) for k in ("tp", "fp", "fn"))
        p, r, f = prf(tp, fp, fn)
        out.update({f"precision@{thr}": p, f"recall@{thr}": r, f"f1@{thr}": f,
                    f"tp@{thr}": tp, f"fp@{thr}": fp, f"fn@{thr}": fn})
        for c in CLASSES:
            gt_c = int(per_image[f"gt_{c}"].sum())
            out[f"recall_{c}@{thr}"] = per_image[f"tp_{c}@{thr}"].sum() / gt_c if gt_c else np.nan
    for c in CLASSES:
        out[f"count_mae_{c}"] = float((per_image[f"pred_{c}"] - per_image[f"gt_{c}"]).abs().mean())
    pred_total = per_image[[f"pred_{c}" for c in CLASSES]].sum(axis=1)
    gt_total = per_image[[f"gt_{c}" for c in CLASSES]].sum(axis=1)
    out["count_mae_overall"] = float((pred_total - gt_total).abs().mean())
    out["pred_regions_per_image"] = float(pred_total.mean())
    out["gt_boxes_per_image"] = float(gt_total.mean())
    return out


# ---------------------------------------------------------------- tuning (train/val only)

def shared_configs(grid: dict) -> list[tuple[str, str, dict, str, str]]:
    """(param, value label, overrides, splitting method, scored class) for every Stage A config.

    WBC regions are scored only against WBC boxes, platelet regions only against platelet
    boxes, and RBC-like regions only against RBC boxes. The RBC area cutoff is scored with
    watershed_fixed at its default min_distance, because otsu_morph leaves clumps unsplit.
    """
    configs = [("wbc_dilate", str(v), {"wbc_dilate": v}, "otsu_morph", "WBC") for v in grid["wbc_dilate"]]
    configs += [("plt_min_area,plt_dilate", f"{a},{d}", {"plt_min_area": a, "plt_dilate": d}, "otsu_morph", "Platelets")
                for a in grid["plt_min_area"] for d in grid["plt_dilate"]]
    configs += [("rbc_min_area", str(v), {"rbc_min_area": v}, "watershed_fixed", "RBC") for v in grid["rbc_min_area"]]
    return configs


def stain_path_grid_one(image_id: str, gt: pd.DataFrame, configs: list, base: dict) -> list[dict]:
    """Class-specific TP/FP/FN at IoU 0.5 for the shared-parameter configs on one image."""
    assert_not_test([image_id], "stain_path_grid_one")
    pre = S.preprocess(load_image(image_id))
    fg, thr = S.foreground_mask(pre["normalized"], base["close_radius"], base["hole_close_radius"],
                                base["open_radius"])
    rows = []
    for param, value, overrides, method, cls in configs:
        p = {**base, **overrides}
        seg = S.segment_prepared(S.assemble(pre, fg, thr, p), method, **p)
        sub = gt[gt["class"] == cls]
        _, boxes, border, kinds = region_arrays(seg)
        keep = kinds == cls
        s = score_image(boxes[keep], border[keep], sub[["xmin", "ymin", "xmax", "ymax"]].to_numpy(),
                        sub["class"].to_numpy(), seg.labels.shape, 0.5)
        rows.append({"image": image_id, "param": param, "value": value, "class": cls,
                     "tp": s.tp, "fp": s.fp, "fn": s.fn})
    return rows


def splitting_configs(fixed_grid: list[int], k1_grid: list[float], k2_grid: list[float]) -> list[tuple[str, str]]:
    return ([("watershed_fixed", f"min_distance={md}") for md in fixed_grid]
            + [("watershed_adaptive", f"k1={k1},k2={k2}") for k2 in k2_grid for k1 in k1_grid])


def splitting_grid_one(image_id: str, gt: pd.DataFrame, params: dict, configs: list[tuple[str, str]]) -> list[dict]:
    """Overall TP/FP/FN at IoU 0.5 for fixed min_distance / adaptive (k1, k2) configs on one image.

    The h-dome map depends on k2 only, so it is computed once per k2 and reused for all k1.
    """
    from skimage.feature import peak_local_max

    assert_not_test([image_id], "splitting_grid_one")
    prep = S.prepare(load_image(image_id), **params)
    mask = prep.rbc_mask
    dist, comps = S.distance_and_components(mask)
    rows = []

    def add(method, key, rbc_labels):
        seg = S.compose(prep, rbc_labels.astype(np.int32), {}, params["min_region_area"])
        r = evaluate_image(seg, gt, iou_thresholds=(0.5,))
        rows.append({"image": image_id, "method": method, "config": key,
                     "tp": r["tp@0.5"], "fp": r["fp@0.5"], "fn": r["fn@0.5"]})

    adaptive = {}
    for method, key in configs:
        values = dict(item.split("=") for item in key.split(","))
        if method == "watershed_fixed":
            md = int(values["min_distance"])
            peaks = peak_local_max(dist, min_distance=md, labels=comps, exclude_border=False)
            add(method, key, S.marker_watershed(dist, comps, mask, peaks))
        else:
            adaptive.setdefault(float(values["k2"]), []).append((float(values["k1"]), key))
    if adaptive:
        r, _, _ = S.estimate_radius(mask, {**S.DEFAULTS, **params})
        smooth = S.smooth_distance(dist, r, params["sigma_frac"])
        for k2, items in adaptive.items():
            domes = S.dome_labels(smooth, k2 * r)
            for k1, key in items:
                peaks = S.adaptive_peaks(smooth, comps, max(1, int(round(k1 * r))), domes)
                add("watershed_adaptive", key, S.marker_watershed(dist, comps, mask, peaks))
    return rows


def aggregate_grid(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = rows.groupby(by, sort=False)[["tp", "fp", "fn"]].sum().reset_index()
    scores = [prf(*t) for t in g[["tp", "fp", "fn"]].itertuples(index=False)]
    g[["precision", "recall", "f1"]] = np.array(scores).reshape(-1, 3)
    return g
