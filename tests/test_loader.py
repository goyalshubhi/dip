import numpy as np
import pandas as pd
import pytest

from src import loader
from src.utils import BCCD_DIR


def _raw(rows):
    return pd.DataFrame(rows, columns=["image", "raw_class", "xmin", "ymin", "xmax", "ymax"])


def test_clean_normalises_names_and_converts_to_zero_based():
    raw = _raw([["a", " rbc ", 1, 1, 50, 60], ["a", "Platelet", 100, 100, 130, 125]])
    df, log = loader.clean_annotations(raw, {"a": (480, 640)})
    assert df["class"].tolist() == ["RBC", "Platelets"]
    assert df.loc[0, ["xmin", "ymin", "xmax", "ymax"]].tolist() == [0, 0, 49, 59]
    assert log["class_name_fixed"] == 2


def test_clean_drops_degenerate_unknown_and_duplicates_and_clips():
    raw = _raw(
        [
            ["a", "RBC", 10, 10, 10, 40],      # zero width
            ["a", "RBC", 60, 60, 40, 90],      # negative width
            ["a", "Monocyte", 1, 1, 30, 30],   # unknown class
            ["a", "WBC", 600, 400, 700, 520],  # outside image -> clipped
            ["a", "RBC", 100, 100, 200, 200],
            ["a", "RBC", 100, 100, 200, 200],  # exact duplicate
            ["a", "RBC", 101, 100, 200, 201],  # near duplicate (IoU > 0.9)
        ]
    )
    df, log = loader.clean_annotations(raw, {"a": (480, 640)})
    assert log["dropped_zero_or_negative_size"] == 2
    assert log["dropped_unknown_class"] == 1
    assert log["dropped_exact_duplicates"] == 1
    assert log["dropped_near_duplicates_iou_gt_0.9"] == 1
    assert log["clipped_to_image"] == 1
    wbc = df[df["class"] == "WBC"].iloc[0]
    assert wbc["xmax"] == 639 and wbc["ymax"] == 479
    assert len(df) == 2


def test_draw_boxes_does_not_modify_input():
    img = np.zeros((50, 50, 3), np.uint8)
    boxes = pd.DataFrame([["a", "RBC", 5, 5, 20, 20]], columns=["image", "class", "xmin", "ymin", "xmax", "ymax"])
    out = loader.draw_boxes(img, boxes)
    assert img.sum() == 0 and out.sum() > 0


@pytest.mark.skipif(not (BCCD_DIR / "Annotations").exists(), reason="BCCD not downloaded")
def test_real_dataset_loads_and_splits_are_disjoint():
    df, log = loader.load_annotations()
    assert set(df["class"]) <= {"RBC", "WBC", "Platelets"}
    assert (df["xmax"] > df["xmin"]).all() and (df["ymax"] > df["ymin"]).all()
    assert df[["xmin", "ymin"]].min().min() >= 0
    splits = loader.build_splits()
    ids = splits["train"] + splits["val"] + splits["test"]
    assert len(ids) == len(set(ids)) == log["n_jpegs"]
