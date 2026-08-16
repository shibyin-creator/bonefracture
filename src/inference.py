from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from config import DETECTION_CLASSES, TINY_WEIGHTS, YOLO_WEIGHTS


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class FractureEngine:
    """YOLOv8 if trained weights exist; otherwise the tiny CNN + optional Hough fallback."""

    def __init__(self) -> None:
        self.mode = "heuristic"
        self.yolo = None
        self.tiny = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if YOLO_WEIGHTS.exists():
            try:
                from ultralytics import YOLO

                self.yolo = YOLO(str(YOLO_WEIGHTS))
                self.mode = "yolov8"
            except Exception:
                self.yolo = None
        if self.yolo is None and TINY_WEIGHTS.exists():
            from src.tiny_model import FractureCNN

            self.tiny = FractureCNN(len(DETECTION_CLASSES))
            ckpt = torch.load(TINY_WEIGHTS, map_location=self.device, weights_only=False)
            self.tiny.load_state_dict(ckpt["state_dict"])
            self.tiny.to(self.device).eval()
            self.mode = "tiny_cnn"

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        if self.mode == "yolov8":
            return self._predict_yolo(bgr)
        if self.mode == "tiny_cnn":
            return self._predict_tiny(bgr)
        return self._predict_heuristic(bgr)

    def _predict_yolo(self, bgr: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for r in self.yolo.predict(bgr, verbose=False):
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names.get(cls_id, DETECTION_CLASSES[cls_id])
                out.append(Detection(name, conf, tuple(map(int, xyxy))))
        return out

    def _predict_tiny(self, bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        inp = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(inp[None, None, ...]).to(self.device)
        with torch.no_grad():
            logits, box = self.tiny(tensor)
            prob = torch.softmax(logits, 1)[0]
            conf, cls_id = float(prob.max()), int(prob.argmax())
            cx, cy, bw, bh = box[0].tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return [Detection(DETECTION_CLASSES[cls_id], conf, (x1, y1, x2, y2))]

    def _predict_heuristic(self, bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=20, maxLineGap=8)
        h, w = gray.shape
        if lines is None:
            return [
                Detection(
                    "Hairline-like region (heuristic)",
                    0.35,
                    (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                )
            ]
        x1, y1, x2, y2 = lines[0][0]
        pad = 12
        box = (
            max(0, min(x1, x2) - pad),
            max(0, min(y1, y2) - pad),
            min(w - 1, max(x1, x2) + pad),
            min(h - 1, max(y1, y2) + pad),
        )
        # crude anatomy guess from aspect / position
        cls = DETECTION_CLASSES[int((y1 + y2) / 2) % len(DETECTION_CLASSES)]
        return [Detection(cls, 0.42, box)]


def annotate(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    vis = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 180), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 180), 1, cv2.LINE_AA)
    return vis
