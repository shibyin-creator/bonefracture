"""Hybrid classification heads from IEEE Access 2025.

* Dense-Softmax on VGG-16 (Model 1)
* Random Forest on VGG-16 features (Model 2)
* Linear SVM on ResNet-50 features (Model 3)
* XGBoost on EfficientNetB0 features (Model 4)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import Config
from dataset_handler.preprocess import keras_generators
from models.efficientnetb0 import build_efficientnetb0_feature_extractor
from models.resnet50 import build_resnet50_feature_extractor
from models.vgg16 import build_vgg16_feature_extractor, build_vgg16_softmax


def _directory_features(directory: Path, backend: str):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if backend == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input

        model = build_vgg16_feature_extractor()
    elif backend == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        model = build_resnet50_feature_extractor()
    elif backend == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input

        model = build_efficientnetb0_feature_extractor()
    else:
        raise ValueError(backend)

    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow = gen.flow_from_directory(
        str(directory),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    feats = model.predict(flow, verbose=1)
    index = {v: k for k, v in flow.class_indices.items()}
    return feats, flow.classes, index


def train_softmax(train_dir: Path, val_dir: Path, epochs: int = Config.VGG_EPOCHS) -> Path:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    train, val = keras_generators(train_dir, val_dir)
    model = build_vgg16_softmax(num_classes=train.num_classes)
    Config.ensure_directories()
    ckpt = Config.LOG_DIR / "vgg16_softmax_best.h5"
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        shuffle=True,
        callbacks=[
            ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            TensorBoard(log_dir=str(Config.LOG_DIR / "tensorboard_vgg16")),
        ],
    )
    dest = Config.WEIGHT_FILES["vgg16"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    dest.with_suffix(".history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2)
    )
    return dest


def train_random_forest(train_dir: Path, val_dir: Path) -> Path:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    x_train, y_train, idx = _directory_features(train_dir, "vgg16")
    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    dest = Config.WEIGHT_FILES["vgg16_rf"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, dest)
    dest.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    return dest


def train_svm(train_dir: Path, val_dir: Path) -> Path:
    from sklearn.svm import SVC
    import joblib

    x_train, y_train, idx = _directory_features(train_dir, "resnet50")
    clf = SVC(kernel="linear", probability=True, class_weight="balanced", C=1.0)
    clf.fit(x_train, y_train)
    dest = Config.WEIGHT_FILES["resnet_svm"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, dest)
    dest.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    return dest


def train_xgboost(train_dir: Path, val_dir: Path) -> Path:
    from xgboost import XGBClassifier
    import joblib

    x_train, y_train, idx = _directory_features(train_dir, "efficientnetb0")
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
    dest = Config.WEIGHT_FILES["efficientnet_xgb"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, dest)
    dest.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    return dest
