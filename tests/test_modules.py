"""Smoke tests for authentication, synthesis, preprocessing, and model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from auth import AdminAuthManager
from data_loader import DatasetHandler
from pipeline import CardiacAssessmentPipeline, synthesize_ecg_waveform
import config


def test_admin_auth(tmp_path: Path) -> None:
    manager = AdminAuthManager(db_path=tmp_path / "auth.sqlite3")
    manager.create_admin("clinician_admin", "SecurePass#2024")
    assert manager.authenticate("clinician_admin", "SecurePass#2024")
    assert not manager.authenticate("clinician_admin", "wrong-password")


def test_synthetic_cohort_exceeds_20k() -> None:
    df = DatasetHandler().generate_synthetic_cohort(n_records=20_100)
    assert len(df) > 20_000
    for col in config.CORE_FEATURES + config.ECG_FEATURES + [config.TARGET_COLUMN]:
        assert col in df.columns
    assert df.isna().sum().sum() == 0
    assert set(df[config.TARGET_COLUMN].unique()) <= {0, 1}


def test_ecg_waveform() -> None:
    t, y = synthesize_ecg_waveform(160, 95, 400, 0.2, 72, 90)
    assert len(t) == len(y)
    assert np.isfinite(y).all()


def test_training_pipeline_and_priority_features() -> None:
    handler = DatasetHandler()
    df = handler.generate_synthetic_cohort(n_records=2500, allow_below_spec=True)
    result = CardiacAssessmentPipeline(fast_tuning=True).run(
        df=df,
        tune=False,
        persist=False,
        max_train_rows=900,
    )
    required = {"accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"}
    for metric in required:
        assert f"test_{metric}" in result.metrics.columns
    assert "Stacking Classifier" in set(result.metrics["model"])
    assert "XGBoost" in set(result.metrics["model"])
    top = set(result.importances.head(8)["feature"])
    hits = [f for f in config.PRIORITY_FEATURES if f in top]
    assert len(hits) >= 3, f"Expected paper-priority features in top ranks, got {list(result.importances.head(8)['feature'])}"
    frame = pd.DataFrame([df.iloc[0][result.models.feature_names]])
    X = result.preprocessor.process_inference_matrix(frame)
    proba = result.models.predict_proba_best(X)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0


if __name__ == "__main__":
    test_admin_auth(Path("artifacts") / "tmp_auth_test")
    test_ecg_waveform()
    test_synthetic_cohort_exceeds_20k()
    test_training_pipeline_and_priority_features()
    print("All smoke tests passed.")
