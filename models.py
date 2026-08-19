"""Baseline classifiers, XGBoost, and stacking ensemble for heart-failure survival."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)


def _xgb_classifier(**kwargs: Any) -> XGBClassifier:
    params = {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "eval_metric": "logloss",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params.update(kwargs)
    return XGBClassifier(**params)


class AdvancedHeartModels:
    """
    Implements the SPIN 2024 baseline suite plus the abstract's ensemble stack:

    Naive Bayes, KNN, Decision Tree, SVM, Logistic Regression, Random Forest,
    XGBoost, and a Stacking Classifier (meta: Logistic Regression;
    bases: XGBoost, Random Forest, SVM).
    """

    def __init__(self, random_state: int = config.RANDOM_STATE) -> None:
        self.random_state = random_state
        self.models: dict[str, Any] = {}
        self.results_: pd.DataFrame = pd.DataFrame()
        self.feature_names: list[str] = []
        self.best_model_name: Optional[str] = None

    def build_estimators(self, probability_svm: bool = True) -> dict[str, Any]:
        base_svm = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            random_state=self.random_state,
        )
        svm = (
            CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
            if probability_svm
            else base_svm
        )
        rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb = _xgb_classifier()
        estimators = {
            "Naive Bayes": GaussianNB(var_smoothing=0.043),
            "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10,
                splitter="random",
                criterion="entropy",
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
            ),
            "SVM": svm,
            "Logistic Regression": LogisticRegression(
                C=0.01,
                solver="liblinear",
                max_iter=500,
                random_state=self.random_state,
            ),
            "Random Forest": rf,
            "XGBoost": xgb,
        }
        stack = StackingClassifier(
            estimators=[
                ("xgb", _xgb_classifier()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=20,
                        min_samples_leaf=4,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "svm",
                    CalibratedClassifierCV(
                        SVC(C=1.0, kernel="rbf", random_state=self.random_state),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ],
            final_estimator=LogisticRegression(
                max_iter=500,
                solver="liblinear",
                random_state=self.random_state,
            ),
            cv=config.STACKING_CONFIG["cv_folds"],
            n_jobs=-1,
            passthrough=config.STACKING_CONFIG["passthrough"],
        )
        estimators["Stacking Classifier"] = stack
        self.models = estimators
        return estimators

    def fit_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models: Optional[dict[str, Any]] = None,
    ) -> "AdvancedHeartModels":
        self.models = models or self.models or self.build_estimators()
        for name, model in self.models.items():
            logger.info("Fitting %s ...", name)
            model.fit(X_train, y_train)
        return self

    @staticmethod
    def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
        else:
            y_score = y_pred.astype(float)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, y_pred)),
            "roc_auc": float(roc_auc_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, y_score)) if len(np.unique(y)) > 1 else float("nan"),
        }
        return metrics

    def evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            train_m = self.evaluate_model(model, X_train, y_train)
            test_m = self.evaluate_model(model, X_test, y_test)
            row = {"model": name}
            for k, v in train_m.items():
                row[f"train_{k}"] = v
            for k, v in test_m.items():
                row[f"test_{k}"] = v
            rows.append(row)
        self.results_ = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
        self.best_model_name = str(self.results_.iloc[0]["model"])
        return self.results_

    def confusion_and_report(self, model_name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        model = self.models[model_name]
        y_pred = model.predict(X)
        return {
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, zero_division=0, output_dict=True),
        }

    def feature_importance(self, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
        names = feature_names or self.feature_names
        if not names:
            raise ValueError("feature_names must be provided.")
        importances = None
        source = None
        xgb = self.models.get("XGBoost")
        rf = self.models.get("Random Forest")
        if xgb is not None and hasattr(xgb, "feature_importances_"):
            importances = np.asarray(xgb.feature_importances_, dtype=float)
            source = "XGBoost"
        elif rf is not None and hasattr(rf, "feature_importances_"):
            importances = np.asarray(rf.feature_importances_, dtype=float)
            source = "Random Forest"
        else:
            raise RuntimeError("No tree ensemble is available for feature importance.")
        table = pd.DataFrame({"feature": names, "importance": importances, "source": source})
        return table.sort_values("importance", ascending=False).reset_index(drop=True)

    def predict_proba_best(self, X: np.ndarray) -> np.ndarray:
        name = self.best_model_name or "Stacking Classifier"
        model = self.models[name]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        scores = model.decision_function(X)
        exp = np.exp(scores - np.max(scores))
        p1 = exp / (1 + exp)
        return np.column_stack([1 - p1, p1])

    def save_bundle(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": self.models,
            "results": self.results_,
            "best_model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "extra": extra or {},
        }
        joblib.dump(payload, path)
        metrics_path = config.METRICS_DIR / "model_comparison.csv"
        if not self.results_.empty:
            self.results_.to_csv(metrics_path, index=False)
        return path

    @classmethod
    def load_bundle(cls, path: str | Path) -> tuple["AdvancedHeartModels", dict[str, Any]]:
        payload = joblib.load(path)
        obj = cls()
        obj.models = payload["models"]
        obj.results_ = payload.get("results", pd.DataFrame())
        obj.best_model_name = payload.get("best_model_name")
        obj.feature_names = payload.get("feature_names", [])
        return obj, payload.get("extra", {})

    def export_ieee_metrics_table(self, path: Optional[str | Path] = None) -> pd.DataFrame:
        """Export test metrics in the SPIN 2024 Table III / IV layout."""
        if self.results_.empty:
            raise RuntimeError("No evaluation results available.")
        table = self.results_[
            [
                "model",
                "train_accuracy",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_mcc",
                "test_roc_auc",
                "test_pr_auc",
            ]
        ].copy()
        dest = Path(path) if path else config.METRICS_DIR / "ieee_metrics_table.csv"
        table.to_csv(dest, index=False)
        (config.METRICS_DIR / "ieee_metrics_table.json").write_text(
            json.dumps(table.to_dict(orient="records"), indent=2)
        )
        return table
