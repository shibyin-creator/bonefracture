"""YOLOv8 training for 7-class anatomical fracture detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WEIGHTS_DIR, YOLO_DATA_YAML, YOLO_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if not YOLO_DATA_YAML.exists():
        raise SystemExit("dataset/data.yaml missing. Run: python scripts/build_dataset.py")

    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    kwargs = dict(
        data=str(YOLO_DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path("runs") / "detect"),
        name="fracture_yolov8",
        exist_ok=True,
    )
    if args.device:
        kwargs["device"] = args.device
    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        YOLO_WEIGHTS.write_bytes(best.read_bytes())
        print(f"Copied best weights to {YOLO_WEIGHTS}")


if __name__ == "__main__":
    main()
