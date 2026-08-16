"""Project configuration for Bone Fracture CAD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DETECTION_CLASSES = [
    "Elbow Positive",
    "Fingers Positive",
    "Forearm Fracture",
    "Humerus",
    "Humerus Fracture",
    "Shoulder Fracture",
    "Wrist Positive",
]

MORPHOLOGY_CLASSES = [
    "Avulsion",
    "Comminuted",
    "Fracture-Dislocation",
    "Greenstick",
    "Hairline",
    "Impacted",
    "Longitudinal",
    "Oblique",
    "Pathological",
    "Spiral",
]

DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
WEIGHTS_DIR = ROOT / "weights"
UPLOAD_DIR = ROOT / "uploads"
RESULT_DIR = ROOT / "static" / "results"

YOLO_DATA_YAML = DATASET_DIR / "data.yaml"
YOLO_WEIGHTS = WEIGHTS_DIR / "fracture_yolov8n.pt"
TINY_WEIGHTS = WEIGHTS_DIR / "tiny_detector.pt"

IMG_SIZE = 256
TARGET_IMAGES = 20000
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

SECRET_KEY = "bone-fracture-cad-change-in-production"
MAX_CONTENT_MB = 16
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
