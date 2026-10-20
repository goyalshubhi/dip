"""BCCD loading: VOC XML -> cleaned box DataFrame, splits, images, box drawing, stats."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .utils import BCCD_DIR, CLASSES, SPLITS_PATH, TABLE_DIR

CLASS_ALIASES = {"rbc": "RBC", "wbc": "WBC", "platelets": "Platelets", "platelet": "Platelets"}
CLASS_COLORS = {"RBC": (220, 40, 40), "WBC": (40, 90, 230), "Platelets": (30, 170, 60)}  # RGB


def image_path(image_id: str, root: Path = BCCD_DIR) -> Path:
    return root / "JPEGImages" / f"{image_id}.jpg"


def load_image(image_id: str, root: Path = BCCD_DIR) -> np.ndarray:
    """Return the image as RGB uint8."""
    bgr = cv2.imread(str(image_path(image_id, root)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path(image_id, root))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_xml(xml_path: Path) -> list[dict]:
    """Raw boxes from one VOC XML, no cleaning. Image id comes from the XML file name."""
    tree = ET.parse(xml_path)
    rows = []
    for obj in tree.getroot().iter("object"):
        bnd = obj.find("bndbox")
        rows.append(
            {
                "image": xml_path.stem,
                "raw_class": obj.findtext("name", default=""),
                "xmin": float(bnd.findtext("xmin")),
                "ymin": float(bnd.findtext("ymin")),
                "xmax": float(bnd.findtext("xmax")),
                "ymax": float(bnd.findtext("ymax")),
            }
        )
    return rows


def load_raw_annotations(root: Path = BCCD_DIR) -> pd.DataFrame:
    rows = []
    for xml_path in sorted((root / "Annotations").glob("*.xml")):
        rows.extend(parse_xml(xml_path))
    return pd.DataFrame(rows)


def image_sizes(image_ids, root: Path = BCCD_DIR) -> dict[str, tuple[int, int]]:
    """Actual (height, width) read from the JPEG headers, not trusted from the XML."""
    sizes = {}
    for image_id in image_ids:
        img = cv2.imread(str(image_path(image_id, root)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(image_path(image_id, root))
        sizes[image_id] = img.shape[:2]
    return sizes


NEAR_DUP_IOU = 0.9


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (area_a + area_b - inter)


def _near_duplicate_mask(df: pd.DataFrame, iou_thr: float) -> pd.Series:
    """Flag the later box of any same-image, same-class pair with IoU > iou_thr (double annotation)."""
    flags = np.zeros(len(df), dtype=bool)
    all_boxes = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    for positions in df.groupby(["image", "class"]).indices.values():
        for a, i in enumerate(positions):
            if flags[i]:
                continue
            for j in positions[a + 1:]:
                if not flags[j] and _box_iou(all_boxes[i], all_boxes[j]) > iou_thr:
                    flags[j] = True
    return pd.Series(flags, index=df.index)


def clean_annotations(raw: pd.DataFrame, sizes: dict[str, tuple[int, int]]) -> tuple[pd.DataFrame, dict]:
    """Normalise class names, clip boxes to the image, drop degenerate/duplicate/unknown boxes.

    Returns the cleaned DataFrame (image, class, xmin, ymin, xmax, ymax) and a log of what changed.
    Coordinates are pixel indices; xmax/ymax are inclusive, as in VOC.
    """
    df = raw.copy()
    log = {"raw_boxes": len(df)}

    stripped = df["raw_class"].str.strip()
    df["class"] = stripped.str.lower().map(CLASS_ALIASES)
    log["class_name_fixed"] = int(((df["raw_class"] != df["class"]) & df["class"].notna()).sum())
    unknown = df["class"].isna()
    log["dropped_unknown_class"] = int(unknown.sum())
    log["unknown_class_names"] = sorted(df.loc[unknown, "raw_class"].unique().tolist())
    df = df[~unknown].copy()

    missing_img = ~df["image"].isin(list(sizes))
    log["dropped_missing_image"] = int(missing_img.sum())
    df = df[~missing_img].copy()

    # VOC pixel coordinates are 1-based (BCCD: min 1, max 640/480); convert to 0-based indices.
    for c in ("xmin", "ymin", "xmax", "ymax"):
        df[c] = df[c].round().astype(int) - 1

    heights = df["image"].map(lambda i: sizes[i][0]).to_numpy()
    widths = df["image"].map(lambda i: sizes[i][1]).to_numpy()
    before = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy().copy()
    df["xmin"] = np.clip(df["xmin"], 0, widths - 1)
    df["xmax"] = np.clip(df["xmax"], 0, widths - 1)
    df["ymin"] = np.clip(df["ymin"], 0, heights - 1)
    df["ymax"] = np.clip(df["ymax"], 0, heights - 1)
    after = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    log["clipped_to_image"] = int((before != after).any(axis=1).sum())

    degenerate = (df["xmax"] <= df["xmin"]) | (df["ymax"] <= df["ymin"])
    log["dropped_zero_or_negative_size"] = int(degenerate.sum())
    df = df[~degenerate]

    dup = df.duplicated(subset=["image", "class", "xmin", "ymin", "xmax", "ymax"])
    log["dropped_exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    near_dup = _near_duplicate_mask(df, iou_thr=NEAR_DUP_IOU)
    log["dropped_near_duplicates_iou_gt_0.9"] = int(near_dup.sum())
    df = df[~near_dup]

    df = df[["image", "class", "xmin", "ymin", "xmax", "ymax"]].reset_index(drop=True)
    log["clean_boxes"] = len(df)
    return df, log


def load_annotations(root: Path = BCCD_DIR) -> tuple[pd.DataFrame, dict]:
    raw = load_raw_annotations(root)
    jpeg_ids = sorted(p.stem for p in (root / "JPEGImages").glob("*.jpg"))
    sizes = image_sizes(jpeg_ids, root)
    df, log = clean_annotations(raw, sizes)
    log["n_jpegs"] = len(jpeg_ids)
    log["n_xmls"] = int(raw["image"].nunique()) if len(raw) else 0
    log["images_without_boxes"] = sorted(set(jpeg_ids) - set(df["image"]))
    return df, log


def build_splits(root: Path = BCCD_DIR, out_path: Path = SPLITS_PATH) -> dict[str, list[str]]:
    """Use the repo's ImageSets/Main lists; save them as data/splits.json."""
    main = root / "ImageSets" / "Main"
    splits = {}
    for name in ("train", "val", "test"):
        lines = (main / f"{name}.txt").read_text(encoding="utf-8").split()
        splits[name] = sorted(set(line.strip() for line in lines if line.strip()))
    all_ids = [i for s in splits.values() for i in s]
    assert len(all_ids) == len(set(all_ids)), "train/val/test lists overlap"
    jpeg_ids = {p.stem for p in (root / "JPEGImages").glob("*.jpg")}
    missing = set(all_ids) - jpeg_ids
    assert not missing, f"split lists reference missing images: {sorted(missing)[:5]}"
    splits["unassigned"] = sorted(jpeg_ids - set(all_ids))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    return splits


def draw_boxes(image: np.ndarray, boxes: pd.DataFrame, thickness: int = 2, label: bool = True) -> np.ndarray:
    """Draw GT boxes (one colour per class) on a copy of an RGB image."""
    out = image.copy()
    cols = boxes[["class", "xmin", "ymin", "xmax", "ymax"]].to_numpy()
    for cls, x0, y0, x1, y1 in cols:
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
        if label:
            cv2.putText(out, str(cls)[:3], (int(x0) + 2, int(y0) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def dataset_stats(df: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Per split: images, class counts, boxes-per-image summary."""
    rows = []
    for split in ("train", "val", "test", "all"):
        ids = splits["train"] + splits["val"] + splits["test"] if split == "all" else splits[split]
        sub = df[df["image"].isin(ids)]
        per_image = sub.groupby("image").size().reindex(ids, fill_value=0)
        row = {"split": split, "images": len(ids), "boxes": len(sub)}
        for c in CLASSES:
            row[f"n_{c}"] = int((sub["class"] == c).sum())
            row[f"images_with_{c}"] = int(sub.loc[sub["class"] == c, "image"].nunique())
        row.update(
            {
                "boxes_per_image_mean": round(per_image.mean(), 3),
                "boxes_per_image_median": float(per_image.median()),
                "boxes_per_image_min": int(per_image.min()),
                "boxes_per_image_max": int(per_image.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def box_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    w = df["xmax"] - df["xmin"] + 1
    h = df["ymax"] - df["ymin"] + 1
    tmp = df.assign(width=w, height=h)
    return tmp.groupby("class")[["width", "height"]].describe().round(1)


def save_dataset_stats(df: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    stats = dataset_stats(df, splits)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(TABLE_DIR / "dataset_stats.csv", index=False)
    return stats


def save_box_overlays(df: pd.DataFrame, image_ids: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(image_ids), figsize=(4.2 * len(image_ids), 4.2))
    for ax, image_id in zip(np.atleast_1d(axes), image_ids):
        boxes = df[df["image"] == image_id]
        ax.imshow(draw_boxes(load_image(image_id), boxes))
        counts = boxes["class"].value_counts()
        ax.set_title(f"{image_id}\n" + ", ".join(f"{c}:{counts.get(c, 0)}" for c in CLASSES), fontsize=9)
        ax.axis("off")
    handles = [Patch(color=np.array(CLASS_COLORS[c]) / 255, label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    from .utils import FIG_DIR, LOG_DIR, SEED, ensure_dirs, get_logger

    ensure_dirs()
    logger = get_logger("phase1_data")
    df, log = load_annotations()
    splits = build_splits()
    logger.info("Cleaning log: %s", json.dumps(log))
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})
    with open(LOG_DIR / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    df.to_csv(TABLE_DIR / "annotations_clean.csv", index=False)
    stats = save_dataset_stats(df, splits)
    logger.info("Dataset stats:\n%s", stats.to_string())
    sizes = box_size_stats(df)
    sizes.to_csv(TABLE_DIR / "box_size_stats.csv")
    logger.info("Box sizes (px):\n%s", sizes.to_string())

    rng = np.random.default_rng(SEED)
    picks = sorted(rng.choice(splits["train"], size=5, replace=False).tolist())
    save_box_overlays(df, picks, FIG_DIR / "gt_box_overlays.png")
    logger.info("Saved GT box overlays for %s", picks)


if __name__ == "__main__":
    main()
