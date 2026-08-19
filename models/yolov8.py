"""YOLOv8 detector — Stage 1 localization (abstract mAP50 = 0.86).

Mean Average Precision at IoU 0.50:

    AP_c = ∫_0^1 p_c(r) dr
    mAP50 = (1/C) Σ_c AP_c    with IoU threshold 0.50
"""

from __future__ import annotations

from pathlib import Path

from config import Config

YOLO_NAMES = [
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
]


def load_yolo(weights: Path | None = None):
    from ultralytics import YOLO

    path = weights or Config.WEIGHT_FILES["yolov8"]
    if path.exists():
        return YOLO(str(path))
    return YOLO("yolov8n.pt")


def train_yolo(
    data_yaml: Path | None = None,
    epochs: int = 80,
    imgsz: int = Config.YOLO_IMAGE_SIZE,
    batch: int = 8,
    model_name: str = "yolov8n.pt",
):
    from ultralytics import YOLO

    yaml_path = data_yaml or Config.YOLO_DATA_YAML
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if not yaml_path.exists():
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_NAMES))
        yaml_path.write_text(
            f"path: {yaml_path.parent.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
        )
    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(Config.LOG_DIR / "detect"),
        name="yolov8_fracture",
        patience=20,
        optimizer="AdamW",
        pretrained=True,
        exist_ok=True,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = Config.WEIGHT_FILES["yolov8"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        dest.write_bytes(best.read_bytes())
    return results


def predict_yolo(image_bgr, model=None, conf: float = Config.CONFIDENCE_THRESHOLD) -> list[dict]:
    detector = model or load_yolo()
    results = detector.predict(source=image_bgr, imgsz=Config.YOLO_IMAGE_SIZE, conf=conf, verbose=False)
    detections: list[dict] = []
    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "label": str(names.get(cls_id, cls_id)),
                    "confidence": float(box.conf[0]),
                    "box": tuple(xyxy),
                }
            )
    return detections
