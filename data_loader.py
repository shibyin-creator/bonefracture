"""Data ingestion, schema validation, and synthetic multi-modal cohort generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

import config

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    Ingest, validate, and expand the UCI Heart Failure Clinical Records schema.

    Synthetic expansion uses a Gaussian-copula tabular generator (CTGAN-style
    joint sampling without the heavy SDV dependency) followed by optional SMOTE
    oversampling. ECG interval columns enable multi-modal diagnostic checks.
    """

    def __init__(self, random_state: int = config.RANDOM_STATE) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_or_generate(
        self,
        csv_path: Optional[str | Path] = None,
        n_records: int = config.TARGET_SYNTHETIC_RECORDS,
        persist: bool = True,
    ) -> pd.DataFrame:
        path = Path(csv_path) if csv_path else config.SYNTHETIC_DATASET_PATH
        if path.exists():
            df = pd.read_csv(path)
            df = self.validate(df)
            if len(df) >= 20_000:
                logger.info("Loaded existing cohort with %s records from %s", f"{len(df):,}", path)
                return df
            logger.warning("Existing file has %s rows (<20,000); regenerating.", len(df))
        df = self.generate_synthetic_cohort(n_records=n_records)
        if persist:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info("Wrote synthetic cohort (%s rows) to %s", f"{len(df):,}", path)
        return df

    def validate(self, df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        rename_map = {
            "anemia": "anaemia",
            "creatinine phosphokinase": "creatinine_phosphokinase",
            "cpk": "creatinine_phosphokinase",
            "ejection fraction": "ejection_fraction",
            "high blood pressure": "high_blood_pressure",
            "serum creatinine": "serum_creatinine",
            "serum sodium": "serum_sodium",
            "follow-up time": "time",
            "follow_up_time": "time",
            "death_event": "DEATH_EVENT",
            "Death Event": "DEATH_EVENT",
        }
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

        missing_core = [c for c in config.CORE_FEATURES if c not in frame.columns]
        if missing_core:
            raise ValueError(f"Uploaded dataset is missing required clinical columns: {missing_core}")

        if require_target and config.TARGET_COLUMN not in frame.columns:
            raise ValueError("Target column DEATH_EVENT is required for training datasets.")

        if not all(c in frame.columns for c in config.ECG_FEATURES):
            frame = self._attach_ecg_features(frame)

        frame = self._clip_to_clinical_ranges(frame)
        frame = self.compute_ecg_flags(frame)

        numeric_cols = config.ALL_FEATURES + ([config.TARGET_COLUMN] if config.TARGET_COLUMN in frame.columns else [])
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        n_before = len(frame)
        frame = frame.drop_duplicates()
        if len(frame) < n_before:
            logger.info("Removed %s duplicate rows.", n_before - len(frame))
        return frame.reset_index(drop=True)

    def generate_synthetic_cohort(
        self,
        n_records: int = config.TARGET_SYNTHETIC_RECORDS,
        allow_below_spec: bool = False,
    ) -> pd.DataFrame:
        if n_records < 20_000 and not allow_below_spec:
            raise ValueError("Specification requires more than 20,000 synthetic patient records.")

        seed = self.build_physiological_seed(n_seed=1200)
        copula_samples = self._gaussian_copula_sample(seed, n_records=int(n_records * 0.92))
        smote_extra = self._smote_tabular(copula_samples, n_extra=n_records - len(copula_samples))
        df = pd.concat([copula_samples, smote_extra], ignore_index=True)
        df = df.sample(n=n_records, random_state=self.random_state).reset_index(drop=True)
        df = self._clip_to_clinical_ranges(df)
        df = self.compute_ecg_flags(df)
        df[config.TARGET_COLUMN] = self._assign_labels(df)
        logger.info(
            "Generated %s records (event rate=%.3f).",
            f"{len(df):,}",
            float(df[config.TARGET_COLUMN].mean()),
        )
        return df

    def build_physiological_seed(self, n_seed: int = 1200) -> pd.DataFrame:
        """Seed cohort matching UCI / SPIN 2024 Table I ranges and typical moments."""
        rng = self.rng
        age = np.clip(rng.normal(60.8, 11.9, n_seed), 40, 95)
        sex = rng.binomial(1, 0.65, n_seed)  # 194/299 ≈ male majority
        anaemia = rng.binomial(1, 0.43, n_seed)
        diabetes = rng.binomial(1, 0.42, n_seed)
        hbp = rng.binomial(1, 0.35, n_seed)
        smoking = rng.binomial(1, 0.32, n_seed)
        cpk = np.clip(rng.lognormal(5.4, 1.05, n_seed), 23, 7861)
        ef = np.clip(rng.normal(38.1, 11.8, n_seed), 14, 80)
        platelets = np.clip(rng.normal(263358, 97804, n_seed), 25000, 850000)
        creat = np.clip(rng.lognormal(0.25, 0.55, n_seed), 0.5, 9.4)
        sodium = np.clip(rng.normal(136.6, 4.4, n_seed), 114, 148)
        time = np.clip(rng.normal(130.3, 77.6, n_seed), 4, 285)

        seed = pd.DataFrame(
            {
                "age": age,
                "anaemia": anaemia,
                "creatinine_phosphokinase": cpk,
                "diabetes": diabetes,
                "ejection_fraction": ef,
                "high_blood_pressure": hbp,
                "platelets": platelets,
                "serum_creatinine": creat,
                "serum_sodium": sodium,
                "sex": sex,
                "smoking": smoking,
                "time": time,
            }
        )
        seed = self._attach_ecg_features(seed)
        seed[config.TARGET_COLUMN] = self._assign_labels(seed)
        return seed

    def compute_ecg_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["ecg_prolonged_qrs"] = (frame["qrs_duration_ms"] >= 120).astype(int)
        frame["ecg_prolonged_qt"] = (frame["qtc_interval_ms"] >= 460).astype(int)
        frame["ecg_st_abnormality"] = (frame["st_deviation_mm"].abs() >= 1.0).astype(int)
        frame["ecg_bradycardia"] = (frame["heart_rate_bpm"] < 60).astype(int)
        frame["ecg_tachycardia"] = (frame["heart_rate_bpm"] > 100).astype(int)
        return frame

    def summarize(self, df: pd.DataFrame) -> dict:
        return {
            "n_records": int(len(df)),
            "n_features": int(len([c for c in df.columns if c != config.TARGET_COLUMN])),
            "event_rate": float(df[config.TARGET_COLUMN].mean()) if config.TARGET_COLUMN in df.columns else None,
            "missing_values": int(df.isna().sum().sum()),
            "columns": list(df.columns),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _attach_ecg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Synthesize ECG intervals conditionally on age, EF, and electrolytes."""
        frame = df.copy()
        n = len(frame)
        rng = self.rng
        age = frame["age"].to_numpy(dtype=float)
        ef = frame["ejection_fraction"].to_numpy(dtype=float)
        sodium = frame["serum_sodium"].to_numpy(dtype=float)
        creat = frame["serum_creatinine"].to_numpy(dtype=float)

        hr = np.clip(72 + 0.12 * (age - 60) - 0.15 * (ef - 38) + rng.normal(0, 9, n), 42, 145)
        rr = 60000.0 / hr
        pr = np.clip(160 + 0.35 * (age - 60) + rng.normal(0, 14, n), 90, 280)
        qrs = np.clip(95 + 0.35 * np.maximum(0, 40 - ef) + rng.normal(0, 8, n), 60, 180)
        qt = np.clip(390 + 0.4 * (age - 60) + 0.8 * (140 - sodium) + rng.normal(0, 18, n), 300, 520)
        qtc = qt / np.sqrt(rr / 1000.0)
        st = np.clip(-0.015 * (40 - ef) + 0.12 * creat + rng.normal(0, 0.55, n), -3.5, 4.0)
        p_wave = np.clip(90 + 0.15 * (age - 60) + rng.normal(0, 8, n), 60, 140)
        t_amp = np.clip(0.35 - 0.004 * (age - 50) + rng.normal(0, 0.12, n), -0.4, 1.2)

        frame["heart_rate_bpm"] = hr
        frame["rr_interval_ms"] = rr
        frame["pr_interval_ms"] = pr
        frame["qrs_duration_ms"] = qrs
        frame["qt_interval_ms"] = qt
        frame["qtc_interval_ms"] = qtc
        frame["st_deviation_mm"] = st
        frame["p_wave_ms"] = p_wave
        frame["t_wave_amplitude_mv"] = t_amp
        return frame

    def _clip_to_clinical_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for col, (lo, hi) in config.FEATURE_RANGES.items():
            if col in frame.columns:
                frame[col] = frame[col].clip(lo, hi)
        for col in config.BINARY_FEATURES:
            if col in frame.columns:
                frame[col] = (frame[col] >= 0.5).astype(int)
        return frame

    def _assign_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label generator that encodes SPIN 2024 / Chicco-Jurman ranking:
        ejection fraction, serum creatinine, follow-up time, and age dominate.
        """
        z = lambda s: (s - s.mean()) / (s.std() + 1e-6)
        logit = (
            2.55 * z(df["age"])
            - 3.15 * z(df["ejection_fraction"])
            + 3.35 * z(df["serum_creatinine"])
            - 2.75 * z(df["time"])
            + 0.35 * df["anaemia"]
            + 0.28 * df["high_blood_pressure"]
            + 0.22 * df["diabetes"]
            + 0.18 * df["smoking"]
            + 0.12 * z(df["creatinine_phosphokinase"])
            - 0.20 * z(df["serum_sodium"])
            + 0.45 * z(df["qrs_duration_ms"])
            + 0.40 * z(df["qtc_interval_ms"])
            + 0.55 * df["st_deviation_mm"].abs()
            - 1.55
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        return (self.rng.uniform(0, 1, len(df)) < prob).astype(int)

    def _gaussian_copula_sample(self, seed: pd.DataFrame, n_records: int) -> pd.DataFrame:
        cols = config.CORE_FEATURES + config.ECG_FEATURES
        data = seed[cols].to_numpy(dtype=float)
        ranks = np.vstack([stats.rankdata(data[:, j]) / (len(seed) + 1) for j in range(data.shape[1])]).T
        gauss = np.clip(stats.norm.ppf(ranks), -4.5, 4.5)
        cov = np.cov(gauss, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-4
        latent = self.rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n_records)
        u = stats.norm.cdf(latent)
        sampled = np.empty_like(u)
        for j, col in enumerate(cols):
            sampled[:, j] = np.quantile(seed[col].to_numpy(dtype=float), u[:, j])
        out = pd.DataFrame(sampled, columns=cols)
        return out

    def _smote_tabular(self, df: pd.DataFrame, n_extra: int) -> pd.DataFrame:
        """SMOTE-style interpolation on minority DEATH_EVENT class (pre-label via temporary labels)."""
        if n_extra <= 0:
            return df.iloc[0:0].copy()
        labeled = df.copy()
        labeled[config.TARGET_COLUMN] = self._assign_labels(labeled)
        minority = labeled[labeled[config.TARGET_COLUMN] == 1]
        majority = labeled[labeled[config.TARGET_COLUMN] == 0]
        pool = minority if len(minority) >= 8 else labeled
        feature_cols = [c for c in pool.columns if c != config.TARGET_COLUMN]
        X = pool[feature_cols].to_numpy(dtype=float)
        k = min(5, len(pool) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        neigh = nn.kneighbors(X, return_distance=False)
        synth_rows = []
        for _ in range(n_extra):
            i = int(self.rng.integers(0, len(X)))
            j = int(self.rng.choice(neigh[i][1:]))
            gap = float(self.rng.random())
            synth_rows.append(X[i] + gap * (X[j] - X[i]))
        extra = pd.DataFrame(synth_rows, columns=feature_cols)
        extra[config.TARGET_COLUMN] = 1 if len(minority) < len(majority) else 0
        return extra.drop(columns=[config.TARGET_COLUMN])
