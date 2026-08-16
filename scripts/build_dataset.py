"""Procedural radiographic-phantom dataset (≥20,000 YOLO-labeled images).

These images are bone-like X-ray phantoms with labeled fracture boxes so the
training pipeline runs without a 20 GB public download. Replace / merge with
GRAZPEDWRI-DX, MURA, FracAtlas, and Roboflow sets for the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATASET_DIR,
    DETECTION_CLASSES,
    IMAGES_DIR,
    IMG_SIZE,
    LABELS_DIR,
    MORPHOLOGY_CLASSES,
    TARGET_IMAGES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    YOLO_DATA_YAML,
)


def _bone_canvas(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = rng.normal(28, 6, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # soft field inhomogeneity (like an X-ray heel effect)
    img += 12 * ((xx / w) ** 1.2)
    img += rng.normal(0, 3, (h, w))
    return img


def _draw_long_bone(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = img.shape
    cx, cy = w // 2 + int(rng.integers(-18, 19)), h // 2 + int(rng.integers(-18, 19))
    bone_w = int(rng.integers(28, 52))
    bone_h = int(rng.integers(140, 220))
    x1, y1 = max(4, cx - bone_w // 2), max(4, cy - bone_h // 2)
    x2, y2 = min(w - 5, cx + bone_w // 2), min(h - 5, cy + bone_h // 2)
    overlay = img.copy()
    cv2.ellipse(
        overlay,
        ((x1 + x2) // 2, (y1 + y2) // 2),
        ((x2 - x1) // 2, (y2 - y1) // 2),
        0,
        0,
        360,
        int(rng.integers(150, 210)),
        -1,
    )
    # cortical edges
    cv2.ellipse(
        overlay,
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (max(6, (x2 - x1) // 2 - 6), max(20, (y2 - y1) // 2 - 8)),
        0,
        0,
        360,
        int(rng.integers(90, 130)),
        -1,
    )
    img[:] = cv2.addWeighted(img, 0.25, overlay, 0.75, 0)
    return x1, y1, x2, y2


def _draw_fracture(
    img: np.ndarray, bone: tuple[int, int, int, int], rng: np.random.Generator, morph: str
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bone
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2 + int(rng.integers(-40, 41))
    cy = int(np.clip(cy, y1 + 8, y2 - 8))
    length = int(rng.integers(18, max(19, x2 - x1 + 8)))
    if morph in {"Spiral", "Oblique"}:
        p1 = (cx - length // 2, cy - 12)
        p2 = (cx + length // 2, cy + 12)
    elif morph == "Longitudinal":
        p1 = (cx, cy - length)
        p2 = (cx, cy + length)
    elif morph == "Comminuted":
        p1 = (cx - length // 2, cy)
        p2 = (cx + length // 2, cy)
        for _ in range(3):
            ox = int(rng.integers(-8, 9))
            oy = int(rng.integers(-8, 9))
            cv2.line(img, (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy), 20, 1)
    else:
        p1 = (cx - length // 2, cy)
        p2 = (cx + length // 2, cy + int(rng.integers(-6, 7)))
    thickness = 2 if morph == "Hairline" else int(rng.integers(2, 4))
    cv2.line(img, p1, p2, 18, thickness, cv2.LINE_AA)
    pad = 8
    bx1 = max(0, min(p1[0], p2[0]) - pad)
    by1 = max(0, min(p1[1], p2[1]) - pad)
    bx2 = min(img.shape[1] - 1, max(p1[0], p2[0]) + pad)
    by2 = min(img.shape[0] - 1, max(p1[1], p2[1]) + pad)
    return bx1, by1, bx2, by2


def _to_yolo(box, w, h, cls_id: int) -> str:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    return f"{cls_id} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"


def generate_one(index: int, split: str, seed: int) -> dict:
    rng = np.random.default_rng(seed + index)
    h = w = IMG_SIZE
    img = _bone_canvas(h, w, rng)
    bone = _draw_long_bone(img, rng)
    det_id = int(index % len(DETECTION_CLASSES))
    morph_id = int(rng.integers(0, len(MORPHOLOGY_CLASSES)))
    morph = MORPHOLOGY_CLASSES[morph_id]
    box = _draw_fracture(img, bone, rng, morph)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 3-channel grayscale (matches the base paper input)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    stem = f"{split}_{index:06d}"
    img_path = IMAGES_DIR / split / f"{stem}.jpg"
    lbl_path = LABELS_DIR / split / f"{stem}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    lbl_path.write_text(_to_yolo(box, w, h, det_id), encoding="utf-8")
    return {
        "file": f"{split}/{stem}.jpg",
        "split": split,
        "detection_class": DETECTION_CLASSES[det_id],
        "detection_id": det_id,
        "morphology_class": morph,
        "morphology_id": morph_id,
        "bbox_xyxy": ",".join(str(int(v)) for v in box),
    }


def write_yaml() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(DETECTION_CLASSES)},
        "nc": len(DETECTION_CLASSES),
    }
    YOLO_DATA_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _job(payload: tuple[int, str, int]) -> dict:
    index, split, seed = payload
    return generate_one(index, split, seed)


def build(n: int = TARGET_IMAGES, seed: int = 42) -> Path:
    write_yaml()
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    jobs: list[tuple[int, str, int]] = []
    cursor = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(count):
            jobs.append((cursor + i, split, seed))
        cursor += count
    with ProcessPoolExecutor() as pool:
        rows = list(tqdm(pool.map(_job, jobs, chunksize=64), total=len(jobs), desc="generate"))

    meta_path = DATASET_DIR / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_images": len(rows),
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "detection_classes": DETECTION_CLASSES,
        "morphology_classes": MORPHOLOGY_CLASSES,
        "note": (
            "Phantom corpus for pipeline training. Merge GRAZPEDWRI-DX (~20,327), "
            "MURA (~40,561), FracAtlas (~4,083), Roboflow 7-class, and Kaggle "
            "10-class morphology for the real ≥20,000 clinical set."
        ),
    }
    (DATASET_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=TARGET_IMAGES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = build(args.images, args.seed)
    print(f"Wrote metadata to {path}")


if __name__ == "__main__":
    main()
