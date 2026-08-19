"""Dataset ingestion, 256×256×3 scaling, augmentation, and class balancing.

Designed for ImageFolder corpora from 1k to >20,000 train *and* >20,000 val
radiographs. Generators stream from disk so the full set need not fit in RAM.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import Config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def scale_rgb(image_bgr: np.ndarray, size: tuple[int, int] = Config.IMAGE_SIZE) -> np.ndarray:
    """Resize to H×W×3 as required by the IEEE paper (256×256×3)."""
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def clahe_enhance(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def augment_bgr(image_bgr: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = image_bgr.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    angle = float(rng.uniform(-12, 12))
    h, w = out.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.4:
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-12, 12))
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    return out


def class_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for klass in sorted(p.name for p in root.iterdir() if p.is_dir()):
        counts[klass] = len(list_images(root / klass))
    return counts


def balance_indices(labels: list[str], strategy: str = "oversample") -> list[int]:
    """Return sample indices that equalize class frequency.

    * oversample — repeat minority classes up to the majority count
    * undersample — downsample majority classes to the minority count
    """
    counts = Counter(labels)
    if not counts:
        return []
    target = max(counts.values()) if strategy == "oversample" else min(counts.values())
    buckets: dict[str, list[int]] = {k: [] for k in counts}
    for idx, label in enumerate(labels):
        buckets[label].append(idx)
    rng = np.random.default_rng(42)
    chosen: list[int] = []
    for label, idxs in buckets.items():
        if strategy == "oversample":
            reps = rng.choice(idxs, size=target, replace=len(idxs) < target)
        else:
            reps = rng.choice(idxs, size=min(target, len(idxs)), replace=False)
        chosen.extend(int(i) for i in reps)
    rng.shuffle(chosen)
    return chosen


def keras_generators(train_dir: Path, val_dir: Path, batch_size: int = Config.BATCH_SIZE):
    """Streaming ImageDataGenerators — suitable for >20k-image splits."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    train = train_gen.flow_from_directory(
        str(train_dir),
        target_size=Config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val = val_gen.flow_from_directory(
        str(val_dir),
        target_size=Config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val


def make_sample_structure(root: Path | None = None, per_class: int = 2) -> Path:
    """Write tiny synthetic radiographs so Colab/local demos run without Kaggle."""
    root = root or (Config.DATASET_DIR / "classification")
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for name in Config.MORPHOLOGY_CLASSES:
            dest = root / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                canvas = np.full((256, 256, 3), 28, np.uint8)
                x0 = int(rng.integers(80, 120))
                cv2.rectangle(canvas, (x0, 30), (x0 + 28, 230), (190, 190, 190), -1)
                if "Hairline" in name:
                    cv2.line(canvas, (x0 + 8, 90), (x0 + 20, 150), (40, 40, 40), 1)
                else:
                    cv2.line(canvas, (x0, 110), (x0 + 28, 150), (20, 20, 20), 2)
                cv2.imwrite(str(dest / f"synth_{i}.png"), canvas)
    return root


def dataset_scale_report(train_dir: Path, val_dir: Path) -> dict:
    train_n = sum(class_counts(train_dir).values())
    val_n = sum(class_counts(val_dir).values())
    return {
        "train_images": train_n,
        "val_images": val_n,
        "meets_20k_train": train_n >= Config.LARGE_DATASET_THRESHOLD,
        "meets_20k_val": val_n >= Config.LARGE_DATASET_THRESHOLD,
        "train_by_class": class_counts(train_dir),
        "val_by_class": class_counts(val_dir),
        "note": (
            "20k+ train/val is supported via streaming generators. "
            "Populate folders from GRAZPEDWRI-DX + YOLO + FracAtlas (see dataset_handler/catalog.py)."
        ),
    }
