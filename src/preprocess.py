from __future__ import annotations

import cv2
import numpy as np


def prepare_xray(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """CLAHE + resize used by the improved proposal (not in the base paper)."""
    if bgr.ndim == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = cv2.resize(eq, (size, size))
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
