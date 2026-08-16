"""Notes and helpers for merging REAL public datasets into a ≥20,000-image corpus.

The bundled generator creates 20,000 labeled phantoms so training works offline.
For the thesis/report, merge these open clinical sets (licenses apply):

1. GRAZPEDWRI-DX  — ~20,327 pediatric wrist radiographs
   https://www.nature.com/articles/s41597-022-01328-z
2. MURA (Stanford) — ~40,561 upper-extremity studies (elbow, finger, forearm, humerus, shoulder, wrist)
   https://stanfordmlgroup.github.io/competitions/mura/
3. FracAtlas       — ~4,083 X-rays with fracture bounding boxes
   https://www.nature.com/articles/s41597-023-02432-4
4. Roboflow 7-class bone fracture YOLO set (matches this project's detection names)
5. Kaggle 10-class bone-break morphology (base paper): ~1,129 images
   https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset

After download, place images/labels under dataset/external/<source>/ and run:

    python scripts/merge_external.py
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    plan = {
        "minimum_images": 20000,
        "recommended_curated_total": 25000,
        "sources": [
            {"name": "GRAZPEDWRI-DX", "images": 20327, "role": "wrist trauma, fracture labels"},
            {"name": "MURA", "images": 40561, "role": "upper limb abnormal vs normal, body-part tags"},
            {"name": "FracAtlas", "images": 4083, "role": "bounding boxes for detection"},
            {"name": "Roboflow 7-class", "images": 3715, "role": "Elbow/Fingers/Forearm/Humerus/Shoulder/Wrist YOLO"},
            {"name": "Kaggle 10-class morphology", "images": 1129, "role": "base-paper fracture types"},
        ],
        "unify_to_detection_classes": [
            "Elbow Positive",
            "Fingers Positive",
            "Forearm Fracture",
            "Humerus",
            "Humerus Fracture",
            "Shoulder Fracture",
            "Wrist Positive",
        ],
        "offline_augmentation": "Albumentations: rotate ±12°, CLAHE, mild noise, scale 0.9–1.1 (no MixUp on hairline cracks)",
    }
    out = ROOT / "dataset" / "PUBLIC_DATASET_PLAN.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    print("\nDownload the sources above, then keep YOLO folders as dataset/images/{train,val,test}.")


if __name__ == "__main__":
    main()
