"""Classification heads (Softmax, Random Forest, SVM, XGBoost)."""

from classifiers.heads import train_random_forest, train_softmax, train_svm, train_xgboost

__all__ = ["train_random_forest", "train_softmax", "train_svm", "train_xgboost"]
