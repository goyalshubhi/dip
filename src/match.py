"""Match segmented regions to GT boxes (Hungarian on IoU) and score one image.

Boxes are inclusive pixel coordinates (xmin, ymin, xmax, ymax), as in the loader.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Any valid match outweighs any IoU difference, so the assignment maximises the number of
# matches first and the total IoU second.
_MATCH_BONUS = 1000.0


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_boxes(pred: np.ndarray, gt: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching with IoU >= iou_thr. Returns (pairs (K, 2) of [pred_idx, gt_idx], their IoUs)."""
    pred = np.asarray(pred).reshape(-1, 4)
    gt = np.asarray(gt).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.empty((0, 2), int), np.empty(0)
    iou = box_iou_matrix(pred, gt)
    valid = iou >= iou_thr
    rows, cols = linear_sum_assignment(-(valid * (_MATCH_BONUS + iou)))
    keep = valid[rows, cols]
    pairs = np.stack([rows[keep], cols[keep]], axis=1)
    return pairs, iou[rows[keep], cols[keep]]


def touches_border(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    boxes = np.asarray(boxes).reshape(-1, 4)
    return (boxes[:, 0] <= 0) | (boxes[:, 1] <= 0) | (boxes[:, 2] >= w - 1) | (boxes[:, 3] >= h - 1)


@dataclass
class ImageScore:
    tp: int
    fp: int
    fn: int
    tp_by_class: dict[str, int]       # matched interior GT boxes per GT class
    gt_by_class: dict[str, int]       # interior GT boxes per class
    matched_pred: np.ndarray          # indices into pred of TP regions
    matched_gt: np.ndarray            # their GT indices
    ignored_pred: np.ndarray          # interior regions matched to a border GT box


def score_image(pred_boxes: np.ndarray, pred_border: np.ndarray, gt_boxes: np.ndarray, gt_classes,
                shape: tuple[int, int], iou_thr: float, classes=("RBC", "WBC", "Platelets")) -> ImageScore:
    """Score one image, ignoring everything at the image border.

    Regions touching the border are dropped. Interior regions are matched against all GT
    boxes; a match to a GT box that touches the border counts neither as TP nor FP (the
    annotation of cut-off cells is unreliable), and only interior GT boxes can be misses.
    """
    pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
    gt_classes = np.asarray(gt_classes)
    interior_pred = np.flatnonzero(~np.asarray(pred_border, bool))
    gt_border = touches_border(gt_boxes, shape)
    pairs, _ = match_boxes(pred_boxes[interior_pred], gt_boxes, iou_thr)
    pred_idx = interior_pred[pairs[:, 0]] if len(pairs) else np.empty(0, int)
    gt_idx = pairs[:, 1] if len(pairs) else np.empty(0, int)
    to_border = gt_border[gt_idx] if len(gt_idx) else np.empty(0, bool)
    tp_pred, tp_gt = pred_idx[~to_border], gt_idx[~to_border]
    n_ignored = int(to_border.sum())
    tp = len(tp_gt)
    fp = len(interior_pred) - tp - n_ignored
    interior_gt_classes = gt_classes[~gt_border]
    fn = len(interior_gt_classes) - tp
    return ImageScore(
        tp=tp, fp=fp, fn=fn,
        tp_by_class={c: int((gt_classes[tp_gt] == c).sum()) for c in classes},
        gt_by_class={c: int((interior_gt_classes == c).sum()) for c in classes},
        matched_pred=tp_pred, matched_gt=tp_gt, ignored_pred=pred_idx[to_border],
    )


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
