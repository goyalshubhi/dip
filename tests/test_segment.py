import numpy as np
import pytest

from src import segment as S
from src.preprocess import preprocess
from src.utils import SPLITS_PATH, load_splits

BACKGROUND = (200, 200, 200)
RBC_COLOR = (190, 140, 165)      # pink: darker green than background, B - R < 0
NUCLEUS_COLOR = (110, 70, 180)   # purple: B - R >> 0


def _canvas(h=240, w=320):
    img = np.empty((h, w, 3), np.float32)
    img[:] = BACKGROUND
    return img


def _disk(img, cy, cx, r, color):
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color


def _finish(img):
    rng = np.random.default_rng(42)
    img = img + rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _n(seg, kind=S.RBC):
    return sum(v == kind for v in seg.types.values())


def test_preprocess_returns_all_intermediates_with_image_shape():
    rgb = _finish(_canvas())
    out = preprocess(rgb)
    for key in ("green", "lab_L", "lab_a", "lab_b", "clahe", "blurred", "normalized", "purple", "saturation"):
        assert out[key].shape == rgb.shape[:2], key
    assert out["normalized"].dtype == np.uint8


def test_single_cell_is_one_region_for_marker_methods():
    img = _canvas()
    _disk(img, 120, 160, 40, RBC_COLOR)
    rgb = _finish(img)
    for method in ("otsu_morph", "watershed_fixed", "watershed_adaptive"):
        assert _n(S.segment(rgb, method, fallback_r=40.0)) == 1, method


def test_touching_cells_split_by_watershed_not_by_otsu():
    img = _canvas()
    _disk(img, 120, 120, 40, RBC_COLOR)
    _disk(img, 120, 190, 40, RBC_COLOR)  # centres 70 px apart: overlapping pair
    rgb = _finish(img)
    assert _n(S.segment(rgb, "otsu_morph")) == 1
    assert _n(S.segment(rgb, "watershed_fixed", min_distance=20)) == 2
    assert _n(S.segment(rgb, "watershed_adaptive", fallback_r=40.0)) == 2
    assert _n(S.segment(rgb, "watershed_plain")) >= 2


def test_pale_centre_is_filled():
    img = _canvas()
    _disk(img, 120, 160, 45, RBC_COLOR)
    _disk(img, 120, 160, 15, BACKGROUND)
    prep = S.prepare(_finish(img))
    assert prep.rbc_mask[120, 160]


def test_purple_blobs_become_wbc_and_platelet():
    img = _canvas()
    _disk(img, 100, 100, 50, NUCLEUS_COLOR)   # ~7850 px of stain -> WBC nucleus
    _disk(img, 200, 260, 10, NUCLEUS_COLOR)   # ~310 px -> platelet
    seg = S.segment(_finish(img), "watershed_adaptive", fallback_r=40.0)
    assert _n(seg, S.WBC) == 1
    assert _n(seg, S.PLATELET) == 1
    assert _n(seg, S.RBC) == 0
    wbc_label = next(k for k, v in seg.types.items() if v == S.WBC)
    # the WBC region is grown beyond the stained nucleus
    assert (seg.labels == wbc_label).sum() > np.pi * 50 ** 2


def test_neighbouring_nuclei_give_two_wbcs():
    img = _canvas()
    _disk(img, 120, 90, 50, NUCLEUS_COLOR)
    _disk(img, 120, 202, 50, NUCLEUS_COLOR)  # 12 px gap: grouped by the lobe dilation
    seg = S.segment(_finish(img), "otsu_morph")
    assert _n(seg, S.WBC) == 2


@pytest.mark.parametrize("radius", [28, 40])
def test_radius_estimate_tracks_cell_size(radius):
    img = _canvas(300, 400)
    _disk(img, 80, 80, radius, RBC_COLOR)
    _disk(img, 200, 250, radius, RBC_COLOR)
    prep = S.prepare(_finish(img))
    r, source, n = S.estimate_radius(prep.rbc_mask, {**S.DEFAULTS, "fallback_r": 999.0})
    assert source == "image" and n == 2
    assert abs(r - radius) <= 0.1 * radius


def test_radius_falls_back_when_no_isolated_cell():
    img = _canvas()
    _disk(img, 120, 0, 40, RBC_COLOR)  # touches the border -> not isolated
    prep = S.prepare(_finish(img))
    r, source, _ = S.estimate_radius(prep.rbc_mask, {**S.DEFAULTS, "fallback_r": 33.0})
    assert source == "train_fallback" and r == 33.0


def test_region_boxes_flag_border_regions():
    img = _canvas()
    _disk(img, 120, 160, 40, RBC_COLOR)
    _disk(img, 120, 10, 30, RBC_COLOR)
    boxes = S.region_boxes(S.segment(_finish(img), "otsu_morph"))
    assert sorted(b["touches_border"] for b in boxes) == [False, True]
    inner = next(b for b in boxes if not b["touches_border"])
    assert abs((inner["xmax"] - inner["xmin"] + 1) - 81) <= 4


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        S.segment(_finish(_canvas()), "magic")


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="splits.json not built")
def test_train_radius_refuses_test_images():
    with pytest.raises(AssertionError):
        S.estimate_train_radius(load_splits()["test"][:1])
