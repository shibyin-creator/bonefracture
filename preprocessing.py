"""StandardScaler pipeline, missing-value handling, SMOTE, and ECG processing."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """IEEE SPIN 2024 preprocessing: impute, StandardScaler, SMOTE, ECG flags."""

    def __init__(self, random_state: int = config.RANDOM_STATE) -> None:
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.fitted = False

    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if config.TARGET_COLUMN not in df.columns:
            raise ValueError("DEATH_EVENT target is required.")
        features = [c for c in config.ALL_FEATURES if c in df.columns]
        X = df[features].copy()
        y = df[config.TARGET_COLUMN].astype(int)
        return X, y

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        self.feature_names = list(X.columns)
        values = self.imputer.fit_transform(X[self.feature_names])
        self.scaler.fit(values)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("DataPreprocessor must be fit before transform.")
        frame = X.reindex(columns=self.feature_names)
        values = self.imputer.transform(frame)
        if scale:
            values = self.scaler.transform(values)
        return values

    def fit_transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        return self.fit(X).transform(X, scale=scale)

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for col in frame.columns:
            if frame[col].isna().any():
                if col in config.BINARY_FEATURES or col == config.TARGET_COLUMN:
                    frame[col] = frame[col].fillna(frame[col].mode().iloc[0] if not frame[col].mode().empty else 0)
                else:
                    frame[col] = frame[col].fillna(frame[col].median())
        return frame

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: str | float = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logger.warning("SMOTE skipped: only one class present.")
            return X, y
        k = max(1, min(5, int(counts.min()) - 1))
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info("SMOTE resized training set from %s to %s samples.", len(y), len(y_res))
        return X_res, y_res

    def process_training_matrix(
        self,
        df: pd.DataFrame,
        smote: bool = True,
        scale: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = self.handle_missing(df)
        X, y = self.split_xy(cleaned)
        X_arr = self.fit_transform(X, scale=scale)
        y_arr = y.to_numpy(dtype=int)
        if smote:
            X_arr, y_arr = self.apply_smote(X_arr, y_arr)
        return X_arr, y_arr

    def process_inference_matrix(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        features = self.feature_names or [c for c in config.ALL_FEATURES if c in df.columns]
        frame = df.reindex(columns=features)
        frame = self.handle_missing(frame)
        return self.transform(frame, scale=scale)

    def inverse_scale(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)
