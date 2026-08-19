"""Joblib / Keras loaders for the four IEEE hybrid heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import Config


def load_joblib(path: Path) -> Any:
    import joblib

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train with: python models/train_vgg16_rf.py  or  python main.py train"
        )
    return joblib.load(path)


def load_vgg16_softmax():
    from tensorflow.keras.models import load_model

    path = Config.WEIGHT_FILES["vgg16"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return load_model(str(path))


def load_vgg16_random_forest():
    return load_joblib(Config.WEIGHT_FILES["vgg16_rf"])


def load_resnet50_svm():
    return load_joblib(Config.WEIGHT_FILES["resnet_svm"])


def load_efficientnet_xgboost():
    return load_joblib(Config.WEIGHT_FILES["efficientnet_xgb"])


def softmax_predict(model, batch: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(batch, verbose=0))
