"""YOLOv8 multi-class bone fracture detection training pipeline.

Abstract protocol: 3,316 train / 399 val images, seven anatomical classes
(Elbow, Fingers, Forearm, Humerus, Humerus Fracture, Shoulder, Wrist),
extended here with pelvic and morphological labels for a 10+ class detector.

Expected Ultralytics layout referenced by data/yolo/data.yaml:
    data/yolo/images/train
    data/yolo/images/val
    data/yolo/labels/train
    data/yolo/labels/val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


DEFAULT_YAML = """# YOLOv8 fracture detection dataset
path: {root}
train: images/train
val: images/val

names:
  0: Elbow Positive
  1: Fingers Positive
  2: Forearm Fracture
  3: Humerus
  4: Humerus Fracture
  5: Shoulder Fracture
  6: Wrist Positive
  7: Pelvic Fracture
  8: Femur
  9: Knee
  10: Ankle
  11: Spine
  12: Avulsion Fracture
  13: Comminuted Fracture
  14: Fracture-Dislocation
  15: Greenstick Fracture
  16: Hairline Fracture
  17: Impacted Fracture
  18: Longitudinal Fracture
  19: Oblique Fracture
  20: Pathological Fracture
  21: Spiral Fracture
"""


def ensure_yaml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_YAML.format(root=str(path.parent.resolve())))
        print(f"Wrote dataset stub: {path}")
    return path


def train(data_yaml: Path, epochs: int, imgsz: int, model_name: str, project: Path, batch: int) -> None:
    from ultralytics import YOLO

    ensure_yaml(data_yaml)
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name="yolov8_fracture",
        patience=20,
        optimizer="AdamW",
        lr0=1e-3,
        pretrained=True,
        exist_ok=True,
        plots=True,
        val=True,
    )
    Config.ensure_directories()
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = Config.WEIGHT_FILES["yolov8"]
    if best.exists():
        dest.write_bytes(best.read_bytes())
        print(f"Copied best detector weights -> {dest}")
    else:
        print("Training finished but best.pt was not found. Check Ultralytics runs/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fracture detector")
    parser.add_argument("--data", type=Path, default=Config.YOLO_DATA_YAML)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=Config.YOLO_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m/l/x.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.data, args.epochs, args.imgsz, args.model, args.project, args.batch)


if __name__ == "__main__":
    main()
