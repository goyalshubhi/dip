import numpy as np
import pytest

from src.features import FEATURE_NAMES, ImageContext, feature_list_table, mask_from_box, region_features


def _image_with_disk(color=(190, 140, 165), r=30):
    rgb = np.full((160, 200, 3), 200, np.uint8)
    yy, xx = np.ogrid[:160, :200]
    disk = (yy - 80) ** 2 + (xx - 100) ** 2 <= r * r
    rgb[disk] = color
    return rgb, disk


def test_exactly_22_features_with_definitions():
    assert len(FEATURE_NAMES) == 22 == len(set(FEATURE_NAMES))
    table = feature_list_table()
    assert len(table) == 22 and table["definition"].str.len().min() > 10


def test_disk_shape_features():
    rgb, disk = _image_with_disk(r=30)
    f = region_features(ImageContext(rgb), disk)
    assert list(f) == FEATURE_NAMES
    assert f["area"] == disk.sum()
    assert f["equivalent_diameter"] == pytest.approx(60, rel=0.03)
    assert f["aspect_ratio"] == pytest.approx(1, abs=0.03)
    assert f["eccentricity"] < 0.2
    assert 0.85 < f["circularity"] <= 1.05
    assert f["compactness"] == pytest.approx(4 * np.pi / f["circularity"])
    assert f["mean_R"] == 190 and f["mean_G"] == 140 and f["gray_std"] == 0
    assert all(np.isfinite(v) for v in f.values())


def test_hue_mean_is_circular():
    # pure red (hue 0) and a slightly blue-ish red (hue ~350) average near 355/0, not near 175
    rgb = np.full((40, 40, 3), 200, np.uint8)
    rgb[:, :20] = (255, 0, 0)
    rgb[:, 20:] = (255, 0, 40)
    mask = np.ones((40, 40), bool)
    hue = region_features(ImageContext(rgb), mask)["hue_mean"]
    assert hue > 340 or hue < 10


def test_thin_region_has_finite_features():
    rgb, _ = _image_with_disk()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[50, 40:60] = True  # one-pixel line: no vertical GLCM pairs, zero minor axis
    f = region_features(ImageContext(rgb), mask)
    assert all(np.isfinite(v) for v in f.values())


def test_mask_from_box_takes_the_central_cell():
    rgb, disk = _image_with_disk(r=30)
    ctx = ImageContext(rgb)
    box = (65, 45, 135, 115)  # around the disk, with margin
    mask = mask_from_box(ctx, box)
    overlap = (mask & disk).sum() / disk.sum()
    assert overlap > 0.9 and (mask & ~disk).sum() < 0.1 * disk.sum()
