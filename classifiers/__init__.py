"""Classification heads (Softmax, Random Forest, SVM, XGBoost)."""

from classifiers.heads import train_random_forest, train_softmax, train_svm, train_xgboost
from classifiers.wrappers import (
    load_efficientnet_xgboost,
    load_resnet50_svm,
    load_vgg16_random_forest,
    load_vgg16_softmax,
)

__all__ = [
    "load_efficientnet_xgboost",
    "load_resnet50_svm",
    "load_vgg16_random_forest",
    "load_vgg16_softmax",
    "train_random_forest",
    "train_softmax",
    "train_svm",
    "train_xgboost",
]
