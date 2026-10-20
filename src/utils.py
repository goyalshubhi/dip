"""Shared paths, seeding, logging and the no-test-leakage guard."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BCCD_DIR = DATA_DIR / "BCCD_Dataset" / "BCCD"
SPLITS_PATH = DATA_DIR / "splits.json"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
CACHE_DIR = RESULTS_DIR / "cache"

CLASSES = ["RBC", "WBC", "Platelets"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for d in (FIG_DIR, TABLE_DIR, LOG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to stdout and results/logs/<name>.log (overwritten per run)."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_splits() -> dict[str, list[str]]:
    with open(SPLITS_PATH, encoding="utf-8") as f:
        return json.load(f)


def assert_not_test(image_ids, context: str = "") -> None:
    """Guard for every tuning function: raise if any test image is passed in."""
    test_ids = set(load_splits()["test"])
    leaked = sorted(set(image_ids) & test_ids)
    assert not leaked, (
        f"Test images passed to a tuning function{' (' + context + ')' if context else ''}: "
        f"{leaked[:5]}{'...' if len(leaked) > 5 else ''}"
    )


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
