"""Comparative ensembles from the IEEE Access 2025 study.

  * ResNet-50 (no top) feature vectors -> Linear SVM
  * EfficientNetB0 pooled features -> XGBoost

These scripts reproduce the paper's hybrid heads so published metrics
(ResNet-50+SVM ~0.93, EfficientNetB0+XGBoost ~0.41) can be re-evaluated
on local or Colab datasets of 1k–20k radiographs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


def _features(directory: Path, backbone: str):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if backbone == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        size = (224, 224)
    elif backbone == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

        model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
        size = (224, 224)
    else:
        raise ValueError(backbone)

    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow = gen.flow_from_directory(
        str(directory),
        target_size=size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    feats = model.predict(flow, verbose=1)
    return feats, flow.classes, {v: k for k, v in flow.class_indices.items()}


def train_resnet_svm(train_dir: Path, val_dir: Path, output: Path) -> None:
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    import joblib

    x_train, y_train, idx = _features(train_dir, "resnet50")
    x_val, y_val, _ = _features(val_dir, "resnet50")
    clf = SVC(kernel="linear", probability=True, class_weight="balanced", C=1.0)
    clf.fit(x_train, y_train)
    print(classification_report(y_val, clf.predict(x_val), target_names=[idx[i] for i in sorted(idx)], zero_division=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)
    output.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    print(f"Saved ResNet-50 + SVM -> {output}")


def train_efficientnet_xgb(train_dir: Path, val_dir: Path, output: Path) -> None:
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier
    import joblib

    x_train, y_train, idx = _features(train_dir, "efficientnetb0")
    x_val, y_val, _ = _features(val_dir, "efficientnetb0")
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    print(classification_report(y_val, clf.predict(x_val), target_names=[idx[i] for i in sorted(idx)], zero_division=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)
    output.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    print(f"Saved EfficientNetB0 + XGBoost -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train comparative hybrid ensembles")
    parser.add_argument("--train-dir", type=Path, default=Config.DATASET_DIR / "classification" / "train")
    parser.add_argument("--val-dir", type=Path, default=Config.DATASET_DIR / "classification" / "val")
    parser.add_argument("--mode", choices=["svm", "xgb", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_dir.exists():
        raise SystemExit(f"Training directory not found: {args.train_dir}")
    Config.ensure_directories()
    if args.mode in {"svm", "both"}:
        train_resnet_svm(args.train_dir, args.val_dir, Config.WEIGHT_FILES["resnet_svm"])
    if args.mode in {"xgb", "both"}:
        train_efficientnet_xgb(args.train_dir, args.val_dir, Config.WEIGHT_FILES["efficientnet_xgb"])


if __name__ == "__main__":
    main()
