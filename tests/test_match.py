import numpy as np
import pytest

from src.match import box_iou_matrix, match_boxes, prf, score_image, touches_border


def test_iou_inclusive_coordinates():
    a = np.array([[0, 0, 9, 9]])           # 10 x 10
    b = np.array([[5, 0, 14, 9], [0, 0, 9, 9], [20, 20, 25, 25]])
    iou = box_iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(50 / 150)
    assert iou[1] == pytest.approx(1.0)
    assert iou[2] == 0.0


def test_hungarian_prefers_more_matches_over_one_greedy_best():
    # pred 0 overlaps both GTs; pred 1 only GT 0. Greedy would give pred0-GT0 and leave GT1 unmatched.
    gt = np.array([[0, 0, 9, 9], [6, 0, 15, 9]])
    pred = np.array([[3, 0, 12, 9], [0, 0, 8, 9]])
    pairs, ious = match_boxes(pred, gt, 0.3)
    assert sorted(map(tuple, pairs)) == [(0, 1), (1, 0)]
    assert (ious >= 0.3).all()


def test_threshold_rejects_weak_overlap():
    pairs, _ = match_boxes(np.array([[0, 0, 9, 9]]), np.array([[5, 0, 14, 9]]), 0.5)
    assert len(pairs) == 0


def test_score_image_ignores_border_regions_and_border_gt():
    shape = (100, 100)
    gt = np.array([[10, 10, 29, 29], [50, 50, 69, 69], [0, 40, 15, 59]])  # last touches the border
    cls = np.array(["RBC", "WBC", "RBC"])
    pred = np.array([
        [11, 11, 29, 30],   # TP for GT 0
        [80, 80, 90, 90],   # FP
        [0, 40, 14, 58],    # border region: ignored
        [2, 41, 15, 59],    # interior region matching the border GT: ignored, not FP
    ])
    border = touches_border(pred, shape)
    s = score_image(pred, border, gt, cls, shape, 0.5)
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)           # GT 1 (WBC) missed; border GT not a miss
    assert s.tp_by_class == {"RBC": 1, "WBC": 0, "Platelets": 0}
    assert s.gt_by_class == {"RBC": 1, "WBC": 1, "Platelets": 0}
    assert list(s.ignored_pred) == [3]


def test_prf_handles_empty():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(3, 1, 2)
    assert (p, r) == (0.75, 0.6) and f == pytest.approx(2 * 0.75 * 0.6 / 1.35)
