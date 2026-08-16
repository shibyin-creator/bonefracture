"""Lightweight tests that do not require trained neural weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from models.refixation import simulate_refixation  # noqa: E402


def test_taxonomy_covers_paper_and_abstract():
    assert len(Config.MORPHOLOGY_CLASSES) == 10
    assert "Comminuted Fracture" in Config.MORPHOLOGY_CLASSES
    assert "Wrist Positive" in Config.ANATOMICAL_CLASSES
    assert "Pelvic Fracture" in Config.ANATOMICAL_CLASSES
    assert len(Config.UNIFIED_CLASSES) >= 12
    assert Config.SEVERE_CLASSES & set(Config.MORPHOLOGY_CLASSES)


def test_refixation_overlay_changes_pixels():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = (40, 40, 40)
    image[40:220, 110:150] = (200, 200, 200)
    overlay, plan = simulate_refixation(image, "Comminuted Fracture")
    assert overlay.shape == image.shape
    assert overlay.sum() > image.sum()
    assert "plate" in plan.hardware or "nail" in plan.hardware or "screw" in plan.hardware.lower()


def test_allowed_extensions_include_dicom():
    assert "dcm" in Config.ALLOWED_EXTENSIONS
    assert "png" in Config.ALLOWED_EXTENSIONS


if __name__ == "__main__":
    test_taxonomy_covers_paper_and_abstract()
    test_refixation_overlay_changes_pixels()
    test_allowed_extensions_include_dicom()
    print("ok")
