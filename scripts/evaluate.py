"""Evaluate detections on the held-out test split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, IMAGES_DIR, ROOT
from src.inference import FractureEngine


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def main() -> None:
    meta = pd.read_csv(DATASET_DIR / "metadata.csv")
    test = meta[meta["split"] == "test"]
    engine = FractureEngine()
    hits = 0
    cls_hits = 0
    n = 0
    for _, row in test.iterrows():
        img = cv2.imread(str(IMAGES_DIR / row["file"]))
        if img is None:
            continue
        dets = engine.predict(img)
        gt = [int(v) for v in str(row["bbox_xyxy"]).split(",")]
        n += 1
        if not dets:
            continue
        best = max(dets, key=lambda d: d.confidence)
        if iou(best.box, gt) >= 0.3:
            hits += 1
        if best.class_name == row["detection_class"]:
            cls_hits += 1
    report = {
        "engine": engine.mode,
        "n_test": n,
        "localization_iou03": hits / n if n else 0,
        "class_accuracy": cls_hits / n if n else 0,
    }
    out = ROOT / "runs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
