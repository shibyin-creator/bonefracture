"""Application configuration for the bone fracture analysis platform.

IEEE Access 2025 (Torne et al.) morphology classes plus YOLOv8 anatomical
classes from the project abstract are registered here so training, inference,
and the clinical UI stay aligned.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Flask and model runtime configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-bone-fracture")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 8 * 60 * 60  # 8 hours clinical shift

    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB DICOM / X-ray uploads
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom", "webp"}

    DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", BASE_DIR / "database.db"))
    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
    WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", BASE_DIR / "weights"))
    DATASET_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "data"))
    YOLO_DATA_YAML = DATASET_DIR / "yolo" / "data.yaml"
    LOG_DIR = BASE_DIR / "logs"

    DEFAULT_ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")

    IMAGE_SIZE = (256, 256)
    YOLO_IMAGE_SIZE = 640
    BATCH_SIZE = 32
    VGG_EPOCHS = 20
    VGG_LEARNING_RATE = 5e-4
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))

    # Paper (IEEE Access 2025): 10 morphological fracture types
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

    # Abstract: YOLOv8 anatomical localizations
    ANATOMICAL_CLASSES = [
        "Elbow Positive",
        "Fingers Positive",
        "Forearm Fracture",
        "Humerus",
        "Humerus Fracture",
        "Shoulder Fracture",
        "Wrist Positive",
        "Pelvic Fracture",
    ]

    # Unified 18-class taxonomy used by the production classifier head
    UNIFIED_CLASSES = MORPHOLOGY_CLASSES + ANATOMICAL_CLASSES

    # Severe / displaced patterns that trigger surgical refixation overlay
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
    }

    # Published benchmark metrics (Torne et al., IEEE Access 2025, Table 2)
    PUBLISHED_METRICS = {
        "VGG-16": {"accuracy": 0.95, "macro_f1": 0.95, "weighted_f1": 0.95},
        "VGG-16 + Random Forest": {"accuracy": 0.95, "macro_f1": 0.95, "weighted_f1": 0.95},
        "ResNet-50 + SVM": {"accuracy": 0.93, "macro_f1": 0.93, "weighted_f1": 0.93},
        "EfficientNetB0 + XGBoost": {"accuracy": 0.41, "macro_f1": 0.42, "weighted_f1": 0.43},
        "YOLOv8": {"mAP50": 0.86, "train_images": 3316, "val_images": 399},
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
            cls.DATASET_DIR / "yolo",
        ):
            path.mkdir(parents=True, exist_ok=True)
