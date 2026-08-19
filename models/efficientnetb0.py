"""EfficientNetB0 feature extractor for the XGBoost comparative baseline (Fig. 5)."""

from __future__ import annotations


def build_efficientnetb0_feature_extractor(pooling: str = "avg"):
    from tensorflow.keras.applications import EfficientNetB0

    return EfficientNetB0(weights="imagenet", include_top=False, pooling=pooling)
