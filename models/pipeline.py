"""Two-stage clinical vision pipeline: YOLOv8 → IEEE classifier stack → Grad-CAM → refix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config
from explainability.gradcam import generate_gradcam
from refix_simulation.simulator import HardwarePlan, save_refixation


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class InferenceResult:
    predicted_class: str
    anatomical_site: str
    confidence: float
    detector_model: str
    classifier_model: str
    detections: list[Detection] = field(default_factory=list)
    needs_refixation: bool = False
    hardware_plan: HardwarePlan | None = None
    original_rel: str = ""
    yolo_rel: str = ""
    gradcam_rel: str = ""
    refix_rel: str = ""
    side_by_side_rel: str = ""
    demo_mode: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "anatomical_site": self.anatomical_site,
            "confidence": round(float(self.confidence), 4),
            "detector_model": self.detector_model,
            "classifier_model": self.classifier_model,
            "detections": [
                {"label": d.label, "confidence": round(d.confidence, 4), "box": list(d.box)}
                for d in self.detections
            ],
            "needs_refixation": self.needs_refixation,
            "hardware": None
            if self.hardware_plan is None
            else {"hardware": self.hardware_plan.hardware, "note": self.hardware_plan.note},
            "original_url": self.original_rel,
            "yolo_url": self.yolo_rel,
            "gradcam_url": self.gradcam_rel,
            "refix_url": self.refix_rel,
            "side_by_side_url": self.side_by_side_rel,
            "demo_mode": self.demo_mode,
            "notes": self.notes,
        }


class FracturePipeline:
    def __init__(self) -> None:
        Config.ensure_directories()
        self._yolo = None
        self._vgg = None
        self._vgg_rf = None
        self._loaded = False
        self.demo_mode = True

    def _try_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if Config.WEIGHT_FILES["yolov8"].exists():
            try:
                from models.yolov8 import load_yolo

                self._yolo = load_yolo()
                self.demo_mode = False
            except Exception:
                self._yolo = None
        if Config.WEIGHT_FILES["vgg16"].exists():
            try:
                from tensorflow.keras.models import load_model

                self._vgg = load_model(str(Config.WEIGHT_FILES["vgg16"]))
                self.demo_mode = False
            except Exception:
                self._vgg = None
        try:
            import joblib

            if Config.WEIGHT_FILES["vgg16_rf"].exists():
                self._vgg_rf = joblib.load(Config.WEIGHT_FILES["vgg16_rf"])
                self.demo_mode = False
        except Exception:
            pass

    def _read(self, path: Path) -> np.ndarray:
        if path.suffix.lower() in {".dcm", ".dicom"}:
            import pydicom

            arr = pydicom.dcmread(str(path)).pixel_array.astype(np.float32)
            arr = np.clip(arr / (arr.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {path}")
        return image

    def _heuristic(self, image_bgr: np.ndarray) -> tuple[str, str, float]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        density = float(cv2.Canny(gray, 50, 150).mean())
        h, w = gray.shape
        aspect = w / max(h, 1)
        morph = Config.MORPHOLOGY_CLASSES[int(density * 10) % len(Config.MORPHOLOGY_CLASSES)]
        if density > 18:
            morph = "Comminuted Fracture"
        elif density > 12:
            morph = "Spiral Fracture" if aspect < 0.85 else "Oblique Fracture"
        elif density < 6:
            morph = "Hairline Fracture"
        sites = ["Humerus", "Forearm", "Wrist", "Femur", "Knee", "Ankle", "Spine"]
        site = sites[int(aspect * 10) % len(sites)]
        return morph, site, float(np.clip(0.55 + density / 40.0, 0.55, 0.92))

    def _draw_yolo(self, image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        canvas = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (48, 201, 186), 2)
            cv2.putText(canvas, f"{det.label} {det.confidence:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (48, 201, 186), 1, cv2.LINE_AA)
        return canvas

    def analyze(self, image_path: Path, stem: str) -> InferenceResult:
        self._try_load()
        image = self._read(image_path)
        notes: list[str] = []
        detections: list[Detection] = []
        if self._yolo is not None:
            from models.yolov8 import predict_yolo

            for item in predict_yolo(image, model=self._yolo):
                detections.append(Detection(item["label"], item["confidence"], item["box"]))
        predicted, site, conf = self._heuristic(image)
        clf_name = "Heuristic (weights not installed)"
        if self._vgg is not None:
            try:
                from tensorflow.keras.applications.vgg16 import preprocess_input

                rgb = cv2.cvtColor(cv2.resize(image, Config.IMAGE_SIZE), cv2.COLOR_BGR2RGB)
                preds = self._vgg.predict(preprocess_input(np.expand_dims(rgb.astype(np.float32), 0)), verbose=0)[0]
                idx = int(np.argmax(preds))
                predicted = Config.UNIFIED_CLASSES[idx] if idx < len(Config.UNIFIED_CLASSES) else predicted
                conf = float(preds[idx])
                clf_name = "VGG-16 Softmax"
            except Exception:
                pass
        if detections:
            site = detections[0].label
        needs = predicted in Config.SEVERE_CLASSES or site in Config.SEVERE_CLASSES
        out = Config.OUTPUT_FOLDER
        orig, yolo_n, cam_n, refix_n = f"{stem}_original.png", f"{stem}_yolo.png", f"{stem}_gradcam.png", f"{stem}_refix.png"
        cv2.imwrite(str(out / orig), image)
        cv2.imwrite(str(out / yolo_n), self._draw_yolo(image, detections))
        cv2.imwrite(str(out / cam_n), generate_gradcam(image, keras_model=self._vgg))
        hardware = None
        refix_rel = side_rel = ""
        if needs:
            _, hardware, side = save_refixation(image, predicted, out / refix_n, box=detections[0].box if detections else None)
            refix_rel = f"/static/outputs/{refix_n}"
            side_rel = f"/static/outputs/{side.name}"
            notes.append("Severe pattern — pre-op vs post-op refixation demo generated.")
        if self.demo_mode:
            notes.append("Running without project weights. Train via: python main.py train --model all")
        return InferenceResult(
            predicted_class=predicted,
            anatomical_site=site,
            confidence=conf,
            detector_model="YOLOv8" if self._yolo is not None else "YOLOv8 (weights pending)",
            classifier_model=clf_name,
            detections=detections,
            needs_refixation=needs,
            hardware_plan=hardware,
            original_rel=f"/static/outputs/{orig}",
            yolo_rel=f"/static/outputs/{yolo_n}",
            gradcam_rel=f"/static/outputs/{cam_n}",
            refix_rel=refix_rel,
            side_by_side_rel=side_rel,
            demo_mode=self.demo_mode,
            notes=notes,
        )


PIPELINE = FracturePipeline()
