"""ResNet-50 feature extractor for the linear-SVM hybrid head (paper Fig. 4)."""

from __future__ import annotations


def build_resnet50_feature_extractor(pooling: str = "avg"):
    from tensorflow.keras.applications import ResNet50

    return ResNet50(weights="imagenet", include_top=False, pooling=pooling)
