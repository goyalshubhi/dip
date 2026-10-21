"""22 handcrafted features per cell region (colour, intensity, GLCM texture, shape, edges).

A region is a boolean mask over the full image. For GT boxes (which have no pixel masks)
`mask_from_box` derives the cell mask inside the box from the same foreground/stain masks
the segmenter uses, so both classification settings share one feature definition.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

from . import segment as S

FEATURES = [
    ("mean_R", "mean red value inside the region"),
    ("mean_G", "mean green value inside the region"),
    ("mean_B", "mean blue value inside the region"),
    ("gray_mean", "mean grayscale intensity inside the region"),
    ("gray_std", "standard deviation of grayscale intensity inside the region"),
    ("gray_min", "minimum grayscale intensity inside the region"),
    ("gray_max", "maximum grayscale intensity inside the region"),
    ("glcm_contrast", "GLCM contrast (distance 1, 4 angles averaged, 32 levels, region pixels only)"),
    ("glcm_correlation", "GLCM correlation (same GLCM)"),
    ("glcm_energy", "GLCM energy, sqrt of angular second moment (same GLCM)"),
    ("glcm_homogeneity", "GLCM homogeneity / inverse difference moment (same GLCM)"),
    ("area", "region area in pixels"),
    ("perimeter", "region perimeter in pixels (skimage 4-connected estimate)"),
    ("circularity", "4*pi*area / perimeter^2 (1 for a disk)"),
    ("aspect_ratio", "major / minor axis length of the moment-equivalent ellipse"),
    ("compactness", "perimeter^2 / area"),
    ("edge_density", "Canny edge pixels inside the region / region area"),
    ("solidity", "area / convex hull area"),
    ("eccentricity", "eccentricity of the moment-equivalent ellipse"),
    ("equivalent_diameter", "diameter of the disk with the same area"),
    ("hue_mean", "circular mean of HSV hue in degrees [0, 360)"),
    ("saturation_mean", "mean HSV saturation (0-255) inside the region"),
]
FEATURE_NAMES = [name for name, _ in FEATURES]
assert len(FEATURE_NAMES) == 22, len(FEATURE_NAMES)

GLCM_LEVELS = 32
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


class ImageContext:
    """Per-image arrays shared by all regions of that image."""

    def __init__(self, rgb: np.ndarray, **seg_params):
        self.rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.hue_rad = hsv[..., 0].astype(np.float64) * (2 * np.pi / 180.0)  # OpenCV hue is 0-179
        self.sat = hsv[..., 1]
        self.edges = cv2.Canny(cv2.GaussianBlur(self.gray, (5, 5), 0), 50, 120) > 0
        self.prep = S.prepare(rgb, **seg_params)
        stain = self.prep.pre["purple"] > {**S.DEFAULTS, **seg_params}["purple_thr"]
        self.cell_mask = self.prep.fg | stain


def mask_from_box(ctx: ImageContext, box) -> np.ndarray:
    """Cell mask inside a GT box: the foreground component covering most of the box centre region.

    Falls back to the full box when the foreground is empty inside it (a few faint cells).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    sub = ctx.cell_mask[y0:y1 + 1, x0:x1 + 1]
    out = np.zeros(ctx.gray.shape, bool)
    labels = measure.label(sub, connectivity=2)
    if labels.max() == 0:
        out[y0:y1 + 1, x0:x1 + 1] = True
        return out
    h, w = sub.shape
    centre = labels[h // 4: h - h // 4 + 1, w // 4: w - w // 4 + 1]
    counts = np.bincount(centre.ravel(), minlength=labels.max() + 1)
    counts[0] = 0
    best = int(counts.argmax()) if counts.max() > 0 else int(np.bincount(labels.ravel())[1:].argmax() + 1)
    out[y0:y1 + 1, x0:x1 + 1] = labels == best
    return out


def _glcm_props(gray_crop: np.ndarray, mask_crop: np.ndarray) -> tuple[float, float, float, float]:
    """GLCM over region pixels only: outside pixels get level 0, whose pairs are discarded."""
    q = (gray_crop.astype(np.int32) * (GLCM_LEVELS - 1) // 255 + 1).astype(np.uint8)
    q[~mask_crop] = 0
    glcm = graycomatrix(q, distances=[1], angles=GLCM_ANGLES, levels=GLCM_LEVELS + 1, symmetric=True)
    glcm = glcm[1:, 1:].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    if (totals == 0).any():
        return 0.0, 0.0, 1.0, 1.0  # region too thin for any in-region pixel pair
    glcm /= totals
    vals = [float(graycoprops(glcm, prop).mean()) for prop in ("contrast", "correlation", "energy", "homogeneity")]
    # correlation is undefined (nan) for a constant region
    vals[1] = 0.0 if not np.isfinite(vals[1]) else vals[1]
    return tuple(vals)


def region_features(ctx: ImageContext, mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    rgb = ctx.rgb[y0:y1, x0:x1][m].astype(np.float64)
    gray_crop = ctx.gray[y0:y1, x0:x1]
    gray = gray_crop[m].astype(np.float64)
    hue = ctx.hue_rad[y0:y1, x0:x1][m]

    props = measure.regionprops(m.astype(np.uint8))[0]
    area = float(props.area)
    perimeter = float(props.perimeter) if props.perimeter > 0 else float(np.sqrt(area) * 4)
    major, minor = props.axis_major_length, props.axis_minor_length
    contrast, correlation, energy, homogeneity = _glcm_props(gray_crop, m)
    hue_mean = np.degrees(np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())) % 360

    feats = {
        "mean_R": rgb[:, 0].mean(), "mean_G": rgb[:, 1].mean(), "mean_B": rgb[:, 2].mean(),
        "gray_mean": gray.mean(), "gray_std": gray.std(), "gray_min": gray.min(), "gray_max": gray.max(),
        "glcm_contrast": contrast, "glcm_correlation": correlation,
        "glcm_energy": energy, "glcm_homogeneity": homogeneity,
        "area": area, "perimeter": perimeter,
        "circularity": 4 * np.pi * area / perimeter ** 2,
        "aspect_ratio": major / minor if minor > 0 else 1.0,
        "compactness": perimeter ** 2 / area,
        "edge_density": ctx.edges[y0:y1, x0:x1][m].mean(),
        "solidity": float(props.solidity), "eccentricity": float(props.eccentricity),
        "equivalent_diameter": float(props.equivalent_diameter_area),
        "hue_mean": hue_mean, "saturation_mean": ctx.sat[y0:y1, x0:x1][m].mean(),
    }
    assert list(feats) == FEATURE_NAMES
    return {k: float(v) for k, v in feats.items()}


def feature_list_table() -> pd.DataFrame:
    return pd.DataFrame(FEATURES, columns=["feature", "definition"])
