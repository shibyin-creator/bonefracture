"""Global configuration for the IEEE-aligned bone-fracture analysis platform.

Hyperparameters follow Torne et al., IEEE Access 2025 (256×256×3 input, Adam
lr=5e-4, categorical cross-entropy, batch 32, 20 epochs) and the project
abstract (YOLOv8, mAP50 = 0.86 on 3,316 / 399 images).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_DATA = ROOT / "bone_fracture_system" / "data"


class Config:
    """Paths, taxonomy, losses, and published benchmarks."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-bone-fracture")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 8 * 60 * 60
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom", "webp"}

    WEB_DIR = ROOT / "web_app"
    DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", WEB_DIR / "database.db"))
    UPLOAD_FOLDER = WEB_DIR / "static" / "uploads"
    OUTPUT_FOLDER = WEB_DIR / "static" / "outputs"
    WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", ROOT / "weights"))
    DATASET_DIR = Path(os.environ.get("DATASET_DIR", ROOT / "data"))
    YOLO_DATA_YAML = DATASET_DIR / "yolo" / "data.yaml"
    LOG_DIR = ROOT / "logs"
    METRICS_DIR = ROOT / "logs" / "metrics"

    DEFAULT_ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")
    DEFAULT_CLINICIAN_USER = os.environ.get("CLINICIAN_USER", "clinician")
    DEFAULT_CLINICIAN_PASSWORD = os.environ.get("CLINICIAN_PASSWORD", "Clinic@123")

    IMAGE_SIZE = (256, 256)
    IMAGE_CHANNELS = 3
    YOLO_IMAGE_SIZE = 640
    BATCH_SIZE = 32
    VGG_EPOCHS = 20
    VGG_LEARNING_RATE = 5e-4
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
    LARGE_DATASET_THRESHOLD = 20_000

    # Categorical cross-entropy (IEEE paper Eq. 1):
    # CCE = - (1/N) Σ_i Σ_c y_{i,c} log(p_{i,c})
    LOSS_NAME = "categorical_crossentropy"
    OPTIMIZER = "adam"

    MORPHOLOGY_CLASSES = [
        "Avulsion Fracture",
        "Comminuted Fracture",
        "Fracture-Dislocation",
        "Greenstick Fracture",
        "Hairline Fracture",
        "Impacted Fracture",
        "Longitudinal Fracture",
        "Oblique Fracture",
        "Pathological Fracture",
        "Spiral Fracture",
    ]

    ANATOMICAL_CLASSES = [
        "Elbow",
        "Fingers",
        "Forearm",
        "Humerus",
        "Shoulder",
        "Wrist",
        "Femur",
        "Knee",
        "Ankle",
        "Spine",
        "Elbow Positive",
        "Fingers Positive",
        "Forearm Fracture",
        "Humerus Fracture",
        "Shoulder Fracture",
        "Wrist Positive",
        "Pelvic Fracture",
    ]

    UNIFIED_CLASSES = MORPHOLOGY_CLASSES + ANATOMICAL_CLASSES

    SEVERE_CLASSES = {
        "Comminuted Fracture",
        "Fracture-Dislocation",
        "Spiral Fracture",
        "Impacted Fracture",
        "Pathological Fracture",
        "Humerus Fracture",
        "Forearm Fracture",
        "Shoulder Fracture",
        "Pelvic Fracture",
        "Femur",
        "Spine",
    }

    PUBLISHED_METRICS = {
        "VGG-16": {"accuracy": 0.95, "macro_f1": 0.95, "weighted_f1": 0.95, "gflops": 15.5},
        "VGG-16 + Random Forest": {"accuracy": 0.95, "macro_f1": 0.95, "weighted_f1": 0.95, "gflops": 15.5},
        "ResNet-50 + SVM": {"accuracy": 0.93, "macro_f1": 0.93, "weighted_f1": 0.93, "gflops": 4.1},
        "EfficientNetB0 + XGBoost": {"accuracy": 0.41, "macro_f1": 0.42, "weighted_f1": 0.43, "gflops": 0.39},
        "YOLOv8": {"mAP50": 0.86, "train_images": 3316, "val_images": 399, "gflops": 8.7},
    }

    WEIGHT_FILES = {
        "vgg16": WEIGHTS_DIR / "vgg16_softmax.h5",
        "vgg16_rf": WEIGHTS_DIR / "vgg16_random_forest.joblib",
        "resnet_svm": WEIGHTS_DIR / "resnet50_svm.joblib",
        "efficientnet_xgb": WEIGHTS_DIR / "efficientnetb0_xgboost.joblib",
        "yolov8": WEIGHTS_DIR / "yolov8_fracture.pt",
    }

    @classmethod
    def ensure_directories(cls) -> None:
        for path in (
            cls.UPLOAD_FOLDER,
            cls.OUTPUT_FOLDER,
            cls.WEIGHTS_DIR,
            cls.LOG_DIR,
            cls.METRICS_DIR,
            cls.DATASET_DIR / "classification" / "train",
            cls.DATASET_DIR / "classification" / "val",
            cls.DATASET_DIR / "yolo" / "images" / "train",
            cls.DATASET_DIR / "yolo" / "images" / "val",
            cls.DATASET_DIR / "yolo" / "labels" / "train",
            cls.DATASET_DIR / "yolo" / "labels" / "val",
        ):
            path.mkdir(parents=True, exist_ok=True)
