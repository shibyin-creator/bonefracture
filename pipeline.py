"""End-to-end training, evaluation, and artifact persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import DatasetHandler
from models import AdvancedHeartModels
from preprocessing import DataPreprocessor
from tuning import HyperparameterTuner

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.SYSTEM_LOG_PATH),
            logging.StreamHandler(),
        ],
        force=True,
    )


@dataclass
class TrainResult:
    models: AdvancedHeartModels
    preprocessor: DataPreprocessor
    metrics: pd.DataFrame
    importances: pd.DataFrame
    bundle_path: Path
    n_records: int
    n_train: int
    n_test: int


class CardiacAssessmentPipeline:
    """Orchestrates generation → preprocess → tune → evaluate → persist."""

    def __init__(self, fast_tuning: bool = config.FAST_TUNING) -> None:
        self.fast_tuning = fast_tuning
        self.handler = DatasetHandler()
        self.preprocessor = DataPreprocessor()
        self.models = AdvancedHeartModels()
        self.tuner = HyperparameterTuner(fast=fast_tuning)

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str | Path] = None,
        tune: bool = True,
        persist: bool = True,
        n_records: int = config.TARGET_SYNTHETIC_RECORDS,
        max_train_rows: Optional[int] = None,
    ) -> TrainResult:
        configure_logging()
        logger.info("Starting cardiac risk pipeline (fast_tuning=%s).", self.fast_tuning)
        if df is None:
            df = self.handler.load_or_generate(csv_path=csv_path, n_records=n_records)
        else:
            df = self.handler.validate(df)

        df = self.preprocessor.handle_missing(df)
        X_df, y = self.preprocessor.split_xy(df)
        self.models.feature_names = list(X_df.columns)

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=config.TRAIN_TEST_SPLIT,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )

        if max_train_rows and len(X_train_df) > max_train_rows:
            X_train_df = X_train_df.sample(n=max_train_rows, random_state=config.RANDOM_STATE)
            y_train = y_train.loc[X_train_df.index]

        X_train = self.preprocessor.fit_transform(X_train_df)
        y_train_np = y_train.to_numpy(dtype=int)
        X_train, y_train_np = self.preprocessor.apply_smote(X_train, y_train_np)
        X_test = self.preprocessor.transform(X_test_df)
        y_test_np = y_test.to_numpy(dtype=int)

        estimators = self.models.build_estimators()
        if tune:
            estimators = self.tuner.tune_suite(estimators, X_train, y_train_np)
        self.models.fit_all(X_train, y_train_np, models=estimators)
        metrics = self.models.evaluate_all(X_train, y_train_np, X_test, y_test_np)
        importances = self.models.feature_importance(self.models.feature_names)
        self.models.export_ieee_metrics_table()
        importances.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)

        extra = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_records": int(len(df)),
            "feature_names": self.models.feature_names,
            "tuning_log": self.tuner.search_log,
        }
        bundle_path = config.DEFAULT_MODEL_BUNDLE
        if persist:
            joblib_extra = {"preprocessor": self.preprocessor, **extra}
            bundle_path = self.models.save_bundle(config.DEFAULT_MODEL_BUNDLE, extra=joblib_extra)
            logger.info("Saved model bundle to %s", bundle_path)

        logger.info("Best model: %s (test acc=%.4f)", self.models.best_model_name, metrics.iloc[0]["test_accuracy"])
        return TrainResult(
            models=self.models,
            preprocessor=self.preprocessor,
            metrics=metrics,
            importances=importances,
            bundle_path=bundle_path,
            n_records=len(df),
            n_train=len(y_train_np),
            n_test=len(y_test_np),
        )


def synthesize_ecg_waveform(
    pr_interval_ms: float,
    qrs_duration_ms: float,
    qt_interval_ms: float,
    st_deviation_mm: float,
    heart_rate_bpm: float,
    p_wave_ms: float,
    duration_s: float = 8.0,
    fs: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise P-QRS-T synthetic lead-II waveform from interval features."""
    t = np.linspace(0, duration_s, int(duration_s * fs))
    rr = 60.0 / max(heart_rate_bpm, 40.0)
    signal = np.zeros_like(t)
    n_beats = int(duration_s / rr) + 2

    def gauss(x, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    for i in range(n_beats):
        t0 = i * rr
        p_mu = t0 + (pr_interval_ms / 1000.0) * 0.35
        qrs_mu = t0 + (pr_interval_ms / 1000.0)
        st_mu = qrs_mu + (qrs_duration_ms / 2000.0) + 0.04
        t_mu = qrs_mu + (qt_interval_ms / 1000.0) * 0.72
        signal += gauss(t, p_mu, max(p_wave_ms / 4000.0, 0.015), 0.18)
        signal += gauss(t, qrs_mu - qrs_duration_ms / 4000.0, 0.012, -0.12)
        signal += gauss(t, qrs_mu, max(qrs_duration_ms / 6000.0, 0.01), 1.15)
        signal += gauss(t, qrs_mu + qrs_duration_ms / 4000.0, 0.012, -0.18)
        signal += st_deviation_mm * 0.08 * np.exp(-0.5 * ((t - st_mu) / 0.06) ** 2)
        signal += gauss(t, t_mu, 0.05, 0.28)
    noise = 0.012 * np.random.default_rng(config.RANDOM_STATE).normal(size=t.shape)
    return t, signal + noise
