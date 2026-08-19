#!/usr/bin/env python3
"""Install the offline FracAtlas bundle that ships in this repository.

No Kaggle account and no internet are required after you clone the repo.

The full CC-BY-4.0 FracAtlas archive (4,083 X-rays, ~323 MB) is stored as
GitHub-safe 80 MB parts under data/offline_bundles/. This script:

  1. Concatenates the parts and verifies the zip
  2. Extracts to data/fracatlas/
  3. Builds ImageFolder train/val (Fractured / Non-fractured)
  4. Builds YOLO train/val from FracAtlas YOLO labels
  5. Copies the bundled 10-class morphology demo set into classification/

GRAZPEDWRI-DX (~15 GB wrist set) cannot fit on GitHub. The 20,327-row CSV
manifest is already in data/manifests/.

Usage:
  python data/install_offline_dataset.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset_handler.preprocess import make_sample_structure  # noqa: E402

BUNDLES = ROOT / "data" / "offline_bundles"
EXTRACT = ROOT / "data" / "fracatlas"
ZIP_OUT = BUNDLES / "FracAtlas.zip"
# sha256 of the Figshare FracAtlas.zip (file 65518038)
EXPECTED_SHA256 = "b67ec2d290a022b3dcf47f78e9a37f7edcc80592c0571f439355bf00bd9f0e23"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def concat_parts() -> Path:
    parts = sorted(BUNDLES.glob("FracAtlas.zip.part*"))
    if not parts:
        raise SystemExit(f"No FracAtlas.zip.part* files in {BUNDLES}. Clone the full git repository.")
    print(f"Concatenating {len(parts)} parts → {ZIP_OUT}")
    with ZIP_OUT.open("wb") as out:
        for part in parts:
            print(" ", part.name, part.stat().st_size)
            out.write(part.read_bytes())
    digest = sha256(ZIP_OUT)
    check = BUNDLES / "SHA256SUMS.txt"
    if check.exists():
        expected = check.read_text().split()[0]
        if digest != expected:
            raise SystemExit(f"Checksum mismatch: got {digest}, expected {expected}")
        print("SHA256 OK", digest)
    else:
        check.write_text(f"{digest}  FracAtlas.zip\n")
        print("Wrote", check, digest)
    return ZIP_OUT


def extract(zip_path: Path) -> Path:
    EXTRACT.mkdir(parents=True, exist_ok=True)
    print("Extracting to", EXTRACT)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT)
    inner = EXTRACT / "FracAtlas"
    if not inner.exists():
        raise SystemExit("Zip did not contain FracAtlas/")
    return inner


def _copy_split(files: list[Path], dest_root: Path, train_ratio: float = 0.8) -> tuple[int, int]:
    n_train = int(len(files) * train_ratio)
    train, val = files[:n_train], files[n_train:]
    for split, group in (("train", train), ("val", val)):
        dest_root.joinpath(split).mkdir(parents=True, exist_ok=True)
        for src in group:
            target = dest_root / split / src.name
            shutil.copy2(src, target)
    return len(train), len(val)


def install_classification(frac: Path) -> None:
    fractured = sorted((frac / "images" / "Fractured").glob("*.jpg"))
    non = sorted((frac / "images" / "Non_fractured").glob("*.jpg"))
    cls = Config.DATASET_DIR / "classification"
    for name, files in (("Fractured", fractured), ("Non-fractured", non)):
        n_train = int(len(files) * 0.8)
        for split, group in (("train", files[:n_train]), ("val", files[n_train:])):
            dest = cls / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for src in group:
                shutil.copy2(src, dest / src.name)
        print(f"  {name}: {len(files)} images")
    bundled = ROOT / "data" / "bundled_morphology"
    if bundled.exists():
        for split in ("train", "val"):
            src = bundled / split
            if not src.exists():
                continue
            for klass_dir in src.iterdir():
                if not klass_dir.is_dir():
                    continue
                dest = cls / split / klass_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for img in klass_dir.glob("*"):
                    shutil.copy2(img, dest / img.name)
        print("  copied bundled 10-class morphology demo images")


def install_yolo(frac: Path) -> None:
    labels_dir = frac / "Annotations" / "YOLO"
    img_dir = frac / "images" / "Fractured"
    pairs = []
    for img in sorted(img_dir.glob("*.jpg")):
        lab = labels_dir / (img.stem + ".txt")
        if not lab.exists():
            continue
        text = lab.read_text().strip()
        if not text:
            continue
        # keep only numeric YOLO rows
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0].replace(".", "", 1).isdigit():
                # force class id 0 = Fracture
                parts[0] = "0"
                rows.append(" ".join(parts))
        if not rows:
            continue
        pairs.append((img, "\n".join(rows) + "\n"))
    n_train = int(len(pairs) * 0.8)
    yolo_root = Config.DATASET_DIR / "yolo"
    for split, group in (("train", pairs[:n_train]), ("val", pairs[n_train:])):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, label in group:
            shutil.copy2(img, yolo_root / "images" / split / img.name)
            (yolo_root / "labels" / split / (img.stem + ".txt")).write_text(label)
    yaml = yolo_root / "fracatlas.yaml"
    yaml.write_text(
        """# FracAtlas fracture vs background (installed offline)
path: .
train: images/train
val: images/val
names:
  0: Fracture
"""
    )
    print(f"  YOLO labelled fracture images: {len(pairs)} (yaml {yaml})")


def main() -> None:
    Config.ensure_directories()
    zip_path = concat_parts()
    frac = extract(zip_path)
    print("Installing classification + YOLO folders…")
    install_classification(frac)
    install_yolo(frac)
    make_sample_structure(Config.DATASET_DIR / "classification", per_class=2)
    print("\nDone. Real FracAtlas X-rays are in data/fracatlas and copied into data/classification and data/yolo.")
    print("Train detector:  python models/train_yolov8.py --data data/yolo/fracatlas.yaml")
    print("Train 10-class:  python models/train_vgg16_rf.py   (uses morphology folders + demo images)")
    print("Research & Diagnostic Support Tool Only — Not an Authorized Medical Device.")


if __name__ == "__main__":
    main()
