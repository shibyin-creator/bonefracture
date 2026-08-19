"""
Project configuration for AI-Powered Cardiac Risk Assessment.

Aligns with Sandilya et al., SPIN 2024 (IEEE) and the submitted abstract:
12 core UCI Heart Failure Clinical Records attributes plus synthetic ECG
interval features for multi-modal cardiac screening.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"

SEED_DATASET_PATH = DATA_DIR / "uci_heart_failure_seed.csv"
SYNTHETIC_DATASET_PATH = DATA_DIR / "heart_failure_multimodal_20k.csv"
DEFAULT_MODEL_BUNDLE = ARTIFACT_DIR / "cardiac_risk_bundle.joblib"
AUTH_DB_PATH = DATA_DIR / "admin_auth.sqlite3"
SYSTEM_LOG_PATH = LOG_DIR / "system_execution.log"

for _path in (DATA_DIR, ARTIFACT_DIR, LOG_DIR, MODEL_DIR, METRICS_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Randomness & dataset scale
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_SYNTHETIC_RECORDS = 20_500  # strictly > 20,000
TRAIN_TEST_SPLIT = 0.20  # 80/20 as in SPIN 2024
SMOTE_RANDOM_STATE = 42

# Fast mode keeps GridSearch tractable in Colab / CI. Set False for paper-scale grids.
FAST_TUNING = True

# ---------------------------------------------------------------------------
# Clinical schema (UCI Heart Failure Clinical Records + ECG extensions)
# ---------------------------------------------------------------------------
CORE_FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

ECG_FEATURES = [
    "pr_interval_ms",
    "qrs_duration_ms",
    "qt_interval_ms",
    "qtc_interval_ms",
    "st_deviation_mm",
    "rr_interval_ms",
    "p_wave_ms",
    "t_wave_amplitude_mv",
    "heart_rate_bpm",
]

DERIVED_ECG_FLAGS = [
    "ecg_prolonged_qrs",
    "ecg_prolonged_qt",
    "ecg_st_abnormality",
    "ecg_bradycardia",
    "ecg_tachycardia",
]

TARGET_COLUMN = "DEATH_EVENT"

ALL_FEATURES = CORE_FEATURES + ECG_FEATURES + DERIVED_ECG_FLAGS

BINARY_FEATURES = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
] + DERIVED_ECG_FLAGS

# Paper-reported clinically dominant predictors (Chicco & Jurman; SPIN 2024 Fig. 2)
PRIORITY_FEATURES = ["ejection_fraction", "serum_creatinine", "time", "age"]

# Physiological ranges used for validation and seed generation (Table I, SPIN 2024)
FEATURE_RANGES = {
    "age": (40.0, 95.0),
    "anaemia": (0, 1),
    "creatinine_phosphokinase": (23.0, 7861.0),
    "diabetes": (0, 1),
    "ejection_fraction": (14.0, 80.0),
    "high_blood_pressure": (0, 1),
    "platelets": (25000.0, 850000.0),
    "serum_creatinine": (0.50, 9.40),
    "serum_sodium": (114.0, 148.0),
    "sex": (0, 1),
    "smoking": (0, 1),
    "time": (4.0, 285.0),
    "pr_interval_ms": (90.0, 280.0),
    "qrs_duration_ms": (60.0, 180.0),
    "qt_interval_ms": (300.0, 520.0),
    "qtc_interval_ms": (320.0, 540.0),
    "st_deviation_mm": (-3.5, 4.0),
    "rr_interval_ms": (400.0, 1500.0),
    "p_wave_ms": (60.0, 140.0),
    "t_wave_amplitude_mv": (-0.4, 1.2),
    "heart_rate_bpm": (40.0, 150.0),
}

# ---------------------------------------------------------------------------
# Hyperparameter search grids (SPIN 2024 Section VI + stacking extension)
# ---------------------------------------------------------------------------
PAPER_PARAM_GRIDS = {
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 0.043, 1e-3]},
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean"],
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20],
        "splitter": ["best", "random"],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
    },
}

FAST_PARAM_GRIDS = {
    "Naive Bayes": {"var_smoothing": [1e-9, 0.043]},
    "KNN": {"n_neighbors": [5, 9], "weights": ["distance"]},
    "Decision Tree": {
        "max_depth": [10],
        "criterion": ["entropy"],
        "min_samples_split": [10],
        "min_samples_leaf": [4],
    },
    "SVM": {"C": [1], "kernel": ["rbf"], "gamma": ["scale"]},
    "Logistic Regression": {"C": [0.01, 1.0], "solver": ["liblinear"], "max_iter": [400]},
    "Random Forest": {
        "n_estimators": [80],
        "max_depth": [20],
        "min_samples_split": [2],
        "min_samples_leaf": [4],
    },
    "XGBoost": {
        "n_estimators": [120],
        "max_depth": [3],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1],
    },
}

STACKING_CONFIG = {
    "meta_learner": "Logistic Regression",
    "base_learners": ["XGBoost", "Random Forest", "SVM"],
    "cv_folds": 3,
    "passthrough": False,
}

TUNING_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1",
    "mcc": "matthews_corrcoef",
    "roc_auc": "roc_auc",
}
TUNING_REFIT_METRIC = "accuracy"

# ---------------------------------------------------------------------------
# Default admin credentials (change after first login in production)
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "CardiacAdmin@2024"
PBKDF2_ITERATIONS = 200_000

# ---------------------------------------------------------------------------
# Streamlit medical theme
# ---------------------------------------------------------------------------
UI_THEME = {
    "background": "#0b1220",
    "panel": "#121a2b",
    "card": "#182238",
    "accent": "#c81e1e",
    "accent_soft": "#e85d5d",
    "text": "#f4f7fb",
    "muted": "#9aa7bd",
    "success": "#1f9d6a",
    "warning": "#d97706",
    "high_risk": "#c81e1e",
    "moderate_risk": "#d97706",
    "low_risk": "#1f9d6a",
    "ecg_trace": "#39d98a",
    "grid": "#243049",
}

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1b2438 0%, #0b1220 45%);
        color: #f4f7fb;
    }
    .block-container { padding-top: 1.2rem; max-width: 1280px; }
    h1, h2, h3 { color: #f4f7fb !important; letter-spacing: 0.2px; }
    .risk-card {
        border: 1px solid #2a3a58;
        background: #182238;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .accent-bar {
        height: 4px;
        background: linear-gradient(90deg, #c81e1e, #e85d5d, #f4f7fb);
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }
    .metric-label { color: #9aa7bd; font-size: 0.82rem; text-transform: uppercase; }
    .stButton>button {
        background: #c81e1e;
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton>button:hover { background: #a31818; color: white; }
    div[data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #243049;
    }
</style>
"""
