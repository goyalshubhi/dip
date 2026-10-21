"""Preprocessing: green channel + LAB, CLAHE, blur, normalisation, and a purple-stain channel.

All intermediates are returned so figures can show each step.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "clahe_clip": 2.0,
    "clahe_tiles": 8,
    "blur": "gaussian",  # "gaussian" or "median"
    "blur_ksize": 5,
    "purple_blur_ksize": 5,
}


def preprocess(rgb: np.ndarray, **params) -> dict[str, np.ndarray]:
    """Return a dict of intermediates for an RGB uint8 image.

    green      raw green channel (best RBC/background contrast in H&E-like smears)
    lab_L/a/b  CIELAB channels (OpenCV 8-bit encoding, a/b offset by 128)
    clahe      CLAHE-equalised green channel
    blurred    smoothed CLAHE output (Gaussian or median)
    normalized blurred, min-max stretched to [0, 255] uint8: the image thresholded for cells
    purple     B - R, smoothed, float32: high for nuclei and platelets (blue-purple stain),
               negative for pink RBCs, about 0 for background
    saturation HSV saturation, for reference/figures
    """
    p = {**DEFAULTS, **params}
    if rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("expected an RGB uint8 image")

    green = rgb[..., 1].copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(p["clahe_tiles"], p["clahe_tiles"]))
    green_eq = clahe.apply(green)

    k = int(p["blur_ksize"]) | 1
    if p["blur"] == "median":
        blurred = cv2.medianBlur(green_eq, k)
    elif p["blur"] == "gaussian":
        blurred = cv2.GaussianBlur(green_eq, (k, k), 0)
    else:
        raise ValueError(f"unknown blur {p['blur']}")
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    purple = rgb[..., 2].astype(np.float32) - rgb[..., 0].astype(np.float32)
    kp = int(p["purple_blur_ksize"]) | 1
    purple = cv2.GaussianBlur(purple, (kp, kp), 0)

    return {
        "rgb": rgb,
        "green": green,
        "lab_L": lab[..., 0],
        "lab_a": lab[..., 1],
        "lab_b": lab[..., 2],
        "clahe": green_eq,
        "blurred": blurred,
        "normalized": normalized,
        "purple": purple,
        "saturation": hsv[..., 1],
    }
