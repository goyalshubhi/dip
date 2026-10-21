"""Cell segmentation: one entry point, four methods for splitting the RBC foreground.

segment(rgb, method, **params) -> Segmentation(labels, types, info)

All methods share the same preprocessing, Otsu foreground, morphology and the
purple-stain path for WBCs and platelets; they differ only in how the RBC-like
foreground is split into cells, so the comparison isolates that step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .preprocess import preprocess
from .utils import TABLE_DIR, assert_not_test

METHODS = ("otsu_morph", "watershed_plain", "watershed_fixed", "watershed_adaptive")
RBC, WBC, PLATELET = "RBC-like", "WBC-nucleus", "platelet-sized"
TRAIN_RADIUS_PATH = TABLE_DIR / "train_radius.json"

DEFAULTS = {
    # shared foreground
    "close_radius": 3,
    "hole_close_radius": 7,    # holes sealed by this larger closing are filled (pale RBC centres)
    "open_radius": 3,
    "rbc_min_area": 1500,      # px; connected components below this are not RBC candidates
    "min_region_area": 100,    # px; slivers left after splitting are dropped (all methods)
    # purple-stain path (B - R channel): WBC nuclei and platelets
    "purple_thr": 20.0,
    "group_radius": 8,         # merges the lobes of one nucleus
    "plt_min_area": 80,        # purple px
    "wbc_min_area": 4000,      # purple px; a group at least this large is a WBC nucleus
    "wbc_dilate": 12,          # px grown around the nucleus hull to approximate the whole WBC
    "plt_dilate": 6,
    # watershed_fixed
    "min_distance": 20,
    # watershed_adaptive
    "k1": 0.6,                 # min_distance = k1 * r
    "k2": 0.15,                # h-maxima height = k2 * r
    "sigma_frac": 0.1,         # Gaussian sigma on the distance transform = sigma_frac * r
    "circ_min": 0.70,          # 4*pi*A/P^2 with skimage's perimeter; single round RBCs score ~0.85-0.9
    "solidity_min": 0.85,
    "min_isolated": 1,
    "fallback_r": None,        # None -> training-set median from TRAIN_RADIUS_PATH
}


@dataclass
class Segmentation:
    labels: np.ndarray                      # int32, 0 = background
    types: dict[int, str]                   # label -> RBC-like / WBC-nucleus / platelet-sized
    info: dict = field(default_factory=dict)


@dataclass
class Prepared:
    """Everything that does not depend on the splitting method (cached during tuning)."""
    shape: tuple[int, int]
    pre: dict[str, np.ndarray]
    fg: np.ndarray            # cleaned Otsu foreground (all cells)
    rbc_mask: np.ndarray      # fg minus WBC regions and platelet cores, small components removed
    wbc_regions: list[np.ndarray]
    plt_regions: list[np.ndarray]
    otsu_thr: float


def _disk(radius: int) -> np.ndarray:
    r = max(int(round(radius)), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop 8-connected components with fewer than min_area pixels."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def foreground_mask(normalized: np.ndarray, close_radius: int, hole_close_radius: int,
                    open_radius: int) -> tuple[np.ndarray, float]:
    """Otsu on the contrast-stretched green channel (cells are darker than background).

    RBCs have pale centres, so the thresholded rim must be closed before hole filling.
    A small closing plus fill handles most cells; holes that only a larger closing seals
    (rims with a gap) are added too, but without the larger closing's bridges between
    neighbouring cells. Opening then cuts thin bridges and removes specks.
    """
    thr, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (normalized < thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _disk(close_radius))
    filled = ndi.binary_fill_holes(fg)
    if hole_close_radius > close_radius:
        sealed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _disk(hole_close_radius)).astype(bool)
        filled |= ndi.binary_fill_holes(sealed) & ~sealed
    fg = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, _disk(open_radius))
    return fg.astype(bool), float(thr)


def purple_objects(purple: np.ndarray, p: dict) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Split the purple-stained pixels into WBC regions and platelet regions.

    Lobes of a segmented nucleus are grouped by a small dilation; a group whose purple
    area reaches wbc_min_area is a WBC (convex hull of the nucleus, grown by wbc_dilate),
    smaller groups are platelets (grown by plt_dilate). Also returns the raw platelet cores.
    """
    stain = remove_small(purple > p["purple_thr"], p["plt_min_area"])
    groups = measure.label(cv2.dilate(stain.astype(np.uint8), _disk(p["group_radius"])) > 0)
    wbc_regions, plt_regions = [], []
    plt_cores = np.zeros_like(stain)
    for region in measure.regionprops(groups):
        r0, c0, r1, c1 = region.bbox
        core = stain[r0:r1, c0:c1] & region.image
        area = int(core.sum())
        if area < p["plt_min_area"]:
            continue
        if area >= p["wbc_min_area"]:
            for nucleus in _split_nuclei(core, p["wbc_min_area"]):
                hull = morphology.convex_hull_image(nucleus)
                wbc_regions.append(_grow(hull, (r0, c0), stain.shape, p["wbc_dilate"], fill=False))
        else:
            plt_cores[r0:r1, c0:c1] |= core
            plt_regions.append(_grow(core, (r0, c0), stain.shape, p["plt_dilate"], fill=True))
    return wbc_regions, plt_regions, plt_cores


def _grow(crop_mask: np.ndarray, origin: tuple[int, int], shape: tuple[int, int], radius: int,
          fill: bool) -> np.ndarray:
    """Dilate a mask given as a crop at origin; returns a full-size mask. Works on a padded crop."""
    pad = int(radius) + 1
    h, w = crop_mask.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad:pad + h, pad:pad + w] = crop_mask
    grown = cv2.dilate(canvas, _disk(radius)) > 0
    if fill:
        grown = ndi.binary_fill_holes(grown)
    out = np.zeros(shape, bool)
    r0, c0 = origin[0] - pad, origin[1] - pad
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r0 + grown.shape[0], shape[0]), min(c0 + grown.shape[1], shape[1])
    out[rr0:rr1, cc0:cc1] = grown[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    return out


def _split_nuclei(core: np.ndarray, wbc_min_area: int) -> list[np.ndarray]:
    """Separate neighbouring WBCs whose nuclei fell into one group.

    Each blob that is nucleus-sized on its own seeds a WBC and smaller lobes join the nearest
    seed; a single nucleus-sized blob (or only small lobes) means one WBC.
    """
    blobs = measure.label(core, connectivity=2)
    props = measure.regionprops(blobs)
    seeds = [r for r in props if r.area >= wbc_min_area]
    if len(seeds) < 2:
        return [core]
    seed_xy = np.array([r.centroid for r in seeds])
    nuclei = [blobs == r.label for r in seeds]
    for r in props:
        if r.area < wbc_min_area:
            nearest = int(np.argmin(((seed_xy - np.array(r.centroid)) ** 2).sum(axis=1)))
            nuclei[nearest] |= blobs == r.label
    return nuclei


def prepare(rgb: np.ndarray, **params) -> Prepared:
    p = {**DEFAULTS, **params}
    pre = preprocess(rgb)
    fg, thr = foreground_mask(pre["normalized"], p["close_radius"], p["hole_close_radius"], p["open_radius"])
    return assemble(pre, fg, thr, p)


def assemble(pre: dict, fg: np.ndarray, thr: float, p: dict) -> Prepared:
    """Stain path and RBC mask on top of a computed foreground (reused across tuning configs)."""
    wbc_regions, plt_regions, plt_cores = purple_objects(pre["purple"], p)
    rbc_mask = fg & ~plt_cores
    for w in wbc_regions:
        rbc_mask &= ~w
    rbc_mask = remove_small(rbc_mask, p["rbc_min_area"])
    return Prepared(fg.shape, pre, fg, rbc_mask, wbc_regions, plt_regions, thr)


def _ensure_marker_per_component(markers: np.ndarray, comp_labels: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Give every foreground component without a marker one marker at its distance maximum."""
    present = np.unique(comp_labels[markers > 0])
    next_id = int(markers.max()) + 1
    for comp in range(1, int(comp_labels.max()) + 1):
        if comp in present:
            continue
        idx = np.flatnonzero(comp_labels == comp)
        if idx.size == 0:
            continue
        markers.flat[idx[np.argmax(dist.flat[idx])]] = next_id
        next_id += 1
    return markers


def _markers_from_peaks(peaks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    markers = np.zeros(shape, np.int32)
    if len(peaks):
        markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    return markers


def isolated_cell_radii(rbc_mask: np.ndarray, p: dict) -> np.ndarray:
    """Inscribed radius (max of the distance transform) of isolated, roughly circular RBC components."""
    dist = cv2.distanceTransform(rbc_mask.astype(np.uint8), cv2.DIST_L2, 5)
    comps = measure.label(rbc_mask, connectivity=2)
    h, w = rbc_mask.shape
    radii = []
    for region in measure.regionprops(comps, intensity_image=dist):
        r0, c0, r1, c1 = region.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue
        if region.area < p["rbc_min_area"] or region.perimeter == 0:
            continue
        circularity = 4 * np.pi * region.area / region.perimeter ** 2
        if circularity >= p["circ_min"] and region.solidity >= p["solidity_min"]:
            radii.append(float(region.intensity_max))
    return np.asarray(radii)


def estimate_radius(rbc_mask: np.ndarray, p: dict) -> tuple[float, str, int]:
    radii = isolated_cell_radii(rbc_mask, p)
    if len(radii) >= p["min_isolated"]:
        return float(np.median(radii)), "image", len(radii)
    fallback = p["fallback_r"] if p["fallback_r"] is not None else load_train_radius()
    return float(fallback), "train_fallback", len(radii)


def estimate_train_radius(train_ids: list[str], **params) -> dict:
    """Median isolated-RBC radius over training images; the adaptive method's fallback."""
    from .loader import load_image

    assert_not_test(train_ids, "estimate_train_radius")
    p = {**DEFAULTS, **params}
    per_image, all_radii = [], []
    for image_id in train_ids:
        radii = isolated_cell_radii(prepare(load_image(image_id), **p).rbc_mask, p)
        all_radii.extend(radii.tolist())
        if len(radii) >= p["min_isolated"]:
            per_image.append(float(np.median(radii)))
    return {
        "fallback_r": float(np.median(all_radii)),
        "n_images": len(train_ids),
        "n_isolated_cells": len(all_radii),
        "n_images_with_enough_isolated": len(per_image),
        "per_image_r_p5_p50_p95": np.percentile(per_image, [5, 50, 95]).round(2).tolist(),
    }


def load_train_radius() -> float:
    if not TRAIN_RADIUS_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_RADIUS_PATH} missing; run scripts/run_segmentation_eval.py (or pass fallback_r)"
        )
    with open(TRAIN_RADIUS_PATH, encoding="utf-8") as f:
        return float(json.load(f)["fallback_r"])


def split_rbc(prep: Prepared, method: str, **params) -> tuple[np.ndarray, dict]:
    """Label the RBC-like foreground with one of the four methods."""
    p = {**DEFAULTS, **params}
    mask = prep.rbc_mask
    info: dict = {}
    if method == "otsu_morph":
        labels = measure.label(mask, connectivity=2)
    elif method == "watershed_plain":
        # No markers: every regional maximum of the distance transform seeds its own basin,
        # so this differs from watershed_fixed only in the absence of marker selection.
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        labels = watershed(-dist, mask=mask)
    elif method == "watershed_fixed":
        dist, comps = distance_and_components(mask)
        min_distance = int(p["min_distance"])
        peaks = peak_local_max(dist, min_distance=min_distance, labels=comps, exclude_border=False)
        labels = marker_watershed(dist, comps, mask, peaks)
        info["min_distance"] = min_distance
    elif method == "watershed_adaptive":
        dist, comps = distance_and_components(mask)
        r, source, n_iso = estimate_radius(mask, p)
        smooth = smooth_distance(dist, r, p["sigma_frac"])
        min_distance = max(1, int(round(p["k1"] * r)))
        h = p["k2"] * r
        peaks = adaptive_peaks(smooth, comps, min_distance, dome_labels(smooth, h))
        labels = marker_watershed(dist, comps, mask, peaks)
        info.update({"r": r, "r_source": source, "n_isolated": n_iso, "h": h, "min_distance": min_distance})
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return labels.astype(np.int32), info


def distance_and_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return dist, measure.label(mask, connectivity=2)


def smooth_distance(dist: np.ndarray, r: float, sigma_frac: float) -> np.ndarray:
    return ndi.gaussian_filter(dist, sigma=max(sigma_frac * r, 0.5))


def dome_labels(smooth: np.ndarray, h: float) -> np.ndarray | None:
    """Label the h-maxima of the smoothed distance transform: domes rising at least h.

    h <= 0 switches the dome test off (None).
    """
    if h <= 0:
        return None
    return measure.label(morphology.h_maxima(smooth, h), connectivity=2)


def adaptive_peaks(smooth: np.ndarray, comps: np.ndarray, min_distance: int, domes: np.ndarray | None) -> np.ndarray:
    """Peaks at least min_distance apart that top an h-dome; the highest peak per dome wins."""
    peaks = peak_local_max(smooth, min_distance=min_distance, labels=comps, exclude_border=False)
    if not len(peaks) or domes is None:
        return peaks.reshape(-1, 2)
    dome_of_peak = domes[peaks[:, 0], peaks[:, 1]]
    kept, seen = [], set()
    for i in np.argsort(-smooth[peaks[:, 0], peaks[:, 1]], kind="stable"):
        d = int(dome_of_peak[i])
        if d and d not in seen:
            seen.add(d)
            kept.append(peaks[i])
    return np.array(kept, dtype=int).reshape(-1, 2)


def marker_watershed(dist: np.ndarray, comps: np.ndarray, mask: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    markers = _ensure_marker_per_component(_markers_from_peaks(peaks, mask.shape), comps, dist)
    return watershed(-dist, markers, mask=mask)


def segment_prepared(prep: Prepared, method: str, **params) -> Segmentation:
    p = {**DEFAULTS, **params}
    rbc_labels, info = split_rbc(prep, method, **p)
    return compose(prep, rbc_labels, info, p["min_region_area"])


def compose(prep: Prepared, rbc_labels: np.ndarray, info: dict, min_region_area: int) -> Segmentation:
    """Merge split RBC labels with the WBC and platelet regions into one labelled image."""
    sizes = np.bincount(rbc_labels.ravel())
    keep = sizes >= min_region_area
    keep[0] = False
    remap = np.zeros(len(sizes), np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    labels = remap[rbc_labels]
    types = {i: RBC for i in range(1, int(keep.sum()) + 1)}
    next_id = len(types) + 1
    # WBC regions were removed from the RBC mask; platelets are painted last so a platelet
    # lying on an RBC keeps its own label.
    for region_mask, kind in [(m, WBC) for m in prep.wbc_regions] + [(m, PLATELET) for m in prep.plt_regions]:
        labels[region_mask] = next_id
        types[next_id] = kind
        next_id += 1
    # A later region can overwrite an earlier one completely; drop labels that vanished.
    present = set(np.unique(labels).tolist()) - {0}
    types = {k: v for k, v in types.items() if k in present}
    info.update({"otsu_thr": prep.otsu_thr, "n_wbc": len(prep.wbc_regions), "n_plt": len(prep.plt_regions)})
    return Segmentation(labels, types, info)


def segment(rgb: np.ndarray, method: str, **params) -> Segmentation:
    """Segment an RGB uint8 image into labelled cells with a coarse type per region."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return segment_prepared(prepare(rgb, **params), method, **params)


def region_boxes(seg: Segmentation) -> list[dict]:
    """One dict per region: label, coarse type, inclusive bbox (x/y min/max), area, touches_border."""
    h, w = seg.labels.shape
    out = []
    for region in measure.regionprops(seg.labels):
        r0, c0, r1, c1 = region.bbox
        out.append(
            {
                "label": region.label,
                "type": seg.types.get(region.label, RBC),
                "xmin": c0, "ymin": r0, "xmax": c1 - 1, "ymax": r1 - 1,
                "area": int(region.area),
                "touches_border": bool(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
            }
        )
    return out
