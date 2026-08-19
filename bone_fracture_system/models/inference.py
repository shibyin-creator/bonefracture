"""Unified inference pipeline: YOLOv8 localization + VGG-16 ensembles + Grad-CAM.

If trained weights are missing the pipeline still runs in clinical-demo mode:
OpenCV preprocessing, optional ImageNet-pretrained VGG-16 features, and
saliency heatmaps. Replace weights under ``weights/`` after training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config
from models.gradcam import generate_gradcam
from models.refixation import HardwarePlan, save_refixation


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
            "hardware": None if self.hardware_plan is None else {
                "hardware": self.hardware_plan.hardware,
                "note": self.hardware_plan.note,
            },
            "original_url": self.original_rel,
            "yolo_url": self.yolo_rel,
            "gradcam_url": self.gradcam_rel,
            "refix_url": self.refix_rel,
            "demo_mode": self.demo_mode,
            "notes": self.notes,
        }


class FracturePipeline:
    """Lazy-loads heavy frameworks so the Flask process can boot without GPUs."""

    def __init__(self) -> None:
        Config.ensure_directories()
        self._yolo = None
        self._vgg = None
        self._vgg_rf = None
        self._resnet_svm = None
        self._eff_xgb = None
        self._loaded = False
        self.demo_mode = True

    def _try_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        yolo_path = Config.WEIGHT_FILES["yolov8"]
        if yolo_path.exists():
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(str(yolo_path))
                self.demo_mode = False
            except Exception:
                self._yolo = None

        vgg_path = Config.WEIGHT_FILES["vgg16"]
        if vgg_path.exists():
            try:
                from tensorflow.keras.models import load_model

                self._vgg = load_model(str(vgg_path))
                self.demo_mode = False
            except Exception:
                self._vgg = None

        try:
            import joblib

            rf_path = Config.WEIGHT_FILES["vgg16_rf"]
            if rf_path.exists():
                self._vgg_rf = joblib.load(rf_path)
                self.demo_mode = False
            svm_path = Config.WEIGHT_FILES["resnet_svm"]
            if svm_path.exists():
                self._resnet_svm = joblib.load(svm_path)
            xgb_path = Config.WEIGHT_FILES["efficientnet_xgb"]
            if xgb_path.exists():
                self._eff_xgb = joblib.load(xgb_path)
        except Exception:
            pass

    def _read_image(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in {".dcm", ".dicom"}:
            try:
                import pydicom

                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                arr = np.clip(arr / (arr.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                if arr.ndim == 2:
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                return arr
            except Exception as exc:
                raise RuntimeError(f"Unable to read DICOM: {exc}") from exc
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {path}")
        return image

    def _heuristic_classify(self, image_bgr: np.ndarray) -> tuple[str, str, float]:
        """Deterministic stand-in used until trained weights are installed."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = float(edges.mean())
        h, w = gray.shape
        aspect = w / max(h, 1)
        moments = cv2.HuMoments(cv2.moments(edges)).flatten()
        score = abs(float(moments[0])) * 1000 + density

        morph = Config.MORPHOLOGY_CLASSES[int(score) % len(Config.MORPHOLOGY_CLASSES)]
        if density > 18:
            morph = "Comminuted Fracture"
        elif density > 12:
            morph = "Spiral Fracture" if aspect < 0.85 else "Oblique Fracture"
        elif density < 6:
            morph = "Hairline Fracture"

        if aspect < 0.7:
            site = "Humerus Fracture"
        elif aspect > 1.3:
            site = "Forearm Fracture"
        else:
            site = "Wrist Positive"
        confidence = float(np.clip(0.55 + density / 40.0, 0.55, 0.92))
        return morph, site, confidence

    def _run_yolo(self, image_bgr: np.ndarray) -> list[Detection]:
        if self._yolo is None:
            return []
        results = self._yolo.predict(
            source=image_bgr,
            imgsz=Config.YOLO_IMAGE_SIZE,
            conf=Config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        label=str(names.get(cls_id, cls_id)),
                        confidence=float(box.conf[0]),
                        box=tuple(int(v) for v in xyxy),  # type: ignore[arg-type]
                    )
                )
        return detections

    def _draw_yolo(self, image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        canvas = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (48, 201, 186), 2)
            caption = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                canvas,
                caption,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (48, 201, 186),
                1,
                cv2.LINE_AA,
            )
        if not detections:
            cv2.putText(
                canvas,
                "YOLOv8: no trained detector weights — localization pending",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 200, 210),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _classify(self, image_bgr: np.ndarray) -> tuple[str, float, str]:
        if self._vgg_rf is not None:
            try:
                features = self._extract_vgg_features(image_bgr)
                proba = self._vgg_rf.predict_proba(features)[0]
                idx = int(np.argmax(proba))
                classes = list(getattr(self._vgg_rf, "classes_", Config.UNIFIED_CLASSES))
                return str(classes[idx]), float(proba[idx]), "VGG-16 + Random Forest"
            except Exception:
                pass
        if self._vgg is not None:
            try:
                from tensorflow.keras.applications.vgg16 import preprocess_input

                rgb = cv2.cvtColor(cv2.resize(image_bgr, Config.IMAGE_SIZE), cv2.COLOR_BGR2RGB)
                tensor = preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0))
                preds = self._vgg.predict(tensor, verbose=0)[0]
                idx = int(np.argmax(preds))
                label = Config.UNIFIED_CLASSES[idx] if idx < len(Config.UNIFIED_CLASSES) else str(idx)
                return label, float(preds[idx]), "VGG-16 Softmax"
            except Exception:
                pass
        morph, _, conf = self._heuristic_classify(image_bgr)
        return morph, conf, "Heuristic (weights not installed)"

    def _extract_vgg_features(self, image_bgr: np.ndarray) -> np.ndarray:
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        from tensorflow.keras.models import Model

        rgb = cv2.cvtColor(cv2.resize(image_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0))
        base = VGG16(weights="imagenet", include_top=False, pooling="avg")
        model = Model(inputs=base.input, outputs=base.output)
        return model.predict(tensor, verbose=0)

    def analyze(self, image_path: Path, stem: str) -> InferenceResult:
        self._try_load()
        image = self._read_image(image_path)
        notes: list[str] = []
        if self.demo_mode:
            notes.append(
                "Running without project-specific weights. Train models via "
                "models/train_vgg16_rf.py, models/train_yolov8.py, or Fracture_Detection_Colab.ipynb."
            )

        detections = self._run_yolo(image)
        predicted, confidence, clf_name = self._classify(image)
        anatomical = detections[0].label if detections else self._heuristic_classify(image)[1]
        if detections:
            predicted_from_box = detections[0].label
            if predicted_from_box in Config.ANATOMICAL_CLASSES:
                anatomical = predicted_from_box

        needs = predicted in Config.SEVERE_CLASSES or anatomical in Config.SEVERE_CLASSES
        box = detections[0].box if detections else None

        out_dir = Config.OUTPUT_FOLDER
        yolo_name = f"{stem}_yolo.png"
        cam_name = f"{stem}_gradcam.png"
        refix_name = f"{stem}_refix.png"
        orig_name = f"{stem}_original.png"

        cv2.imwrite(str(out_dir / orig_name), image)
        cv2.imwrite(str(out_dir / yolo_name), self._draw_yolo(image, detections))
        heatmap = generate_gradcam(image, keras_model=self._vgg)
        cv2.imwrite(str(out_dir / cam_name), heatmap)

        hardware = None
        refix_rel = ""
        if needs:
            _, hardware = save_refixation(image, predicted, out_dir / refix_name, box=box)
            refix_rel = f"/static/outputs/{refix_name}"
            notes.append("Severe / displaced pattern detected — surgical refixation demo generated.")

        return InferenceResult(
            predicted_class=predicted,
            anatomical_site=anatomical,
            confidence=confidence,
            detector_model="YOLOv8" if self._yolo is not None else "YOLOv8 (weights pending)",
            classifier_model=clf_name,
            detections=detections,
            needs_refixation=needs,
            hardware_plan=hardware,
            original_rel=f"/static/outputs/{orig_name}",
            yolo_rel=f"/static/outputs/{yolo_name}",
            gradcam_rel=f"/static/outputs/{cam_name}",
            refix_rel=refix_rel,
            demo_mode=self.demo_mode,
            notes=notes,
        )


PIPELINE = FracturePipeline()
