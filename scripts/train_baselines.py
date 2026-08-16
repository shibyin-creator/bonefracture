"""Reproduce base-paper hybrid classifiers (VGG-16, RF, SVM, XGBoost) on morphology labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, IMAGES_DIR, ROOT, WEIGHTS_DIR


def _load_split(split: str, max_per_split: int | None) -> tuple[np.ndarray, np.ndarray]:
    meta = pd.read_csv(DATASET_DIR / "metadata.csv")
    part = meta[meta["split"] == split]
    if max_per_split:
        part = part.iloc[:max_per_split]
    xs, ys = [], []
    for _, row in part.iterrows():
        img = cv2.imread(str(IMAGES_DIR / row["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
        xs.append(img.flatten())
        ys.append(int(row["morphology_id"]))
    return np.asarray(xs), np.asarray(ys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=8000)
    parser.add_argument("--max-test", type=int, default=2000)
    args = parser.parse_args()

    x_train, y_train = _load_split("train", args.max_train)
    x_test, y_test = _load_split("test", args.max_test)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=18, n_jobs=-1, random_state=42
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LinearSVC(max_iter=2000, dual=False, random_state=42)),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=int(y_train.max()) + 1,
            tree_method="hist",
        )
    except ImportError:
        print("xgboost not installed; skipping EfficientNetB0+XGBoost analogue")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        reports[name] = classification_report(y_test, pred, output_dict=True, zero_division=0)
        print("\n===", name, "===")
        print(classification_report(y_test, pred, zero_division=0))

    out = ROOT / "runs" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "morphology_baselines.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("Saved", out / "morphology_baselines.json")


if __name__ == "__main__":
    main()
