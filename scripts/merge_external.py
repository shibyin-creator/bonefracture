"""Copy YOLO-format folders from dataset/external/* into the unified 20k+ corpus."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IMAGES_DIR, LABELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="dataset/external")
    args = parser.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} not found. Put Roboflow/FracAtlas YOLO folders there.")
    copied = 0
    for split in ("train", "val", "test"):
        for img in (src.rglob(f"**/images/{split}/*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            dest = IMAGES_DIR / split / f"ext_{img.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            if lbl.exists():
                ldest = LABELS_DIR / split / f"ext_{img.stem}.txt"
                ldest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lbl, ldest)
            copied += 1
    print(f"Copied {copied} external images into dataset/images")


if __name__ == "__main__":
    main()
