"""IEEE modular stack smoke tests (no GPU weights required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset_handler.preprocess import make_sample_structure, scale_rgb  # noqa: E402
from explainability.metrics import evaluate_classifier, multiclass_specificity  # noqa: E402
from refix_simulation.simulator import simulate_refixation  # noqa: E402


def test_anatomy_includes_lower_limb_and_spine():
    for name in ("Femur", "Knee", "Ankle", "Spine", "Elbow", "Wrist"):
        assert name in Config.ANATOMICAL_CLASSES
    assert len(Config.MORPHOLOGY_CLASSES) == 10


def test_scale_rgb_is_256x256x3():
    img = np.zeros((180, 220), dtype=np.uint8)
    rgb = scale_rgb(img)
    assert rgb.shape == (256, 256, 3)
    assert rgb.max() <= 1.0


def test_refix_side_by_side_and_gap_vector():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = 40
    image[40:220, 110:150] = 200
    result = simulate_refixation(image, "Comminuted Fracture", box=(100, 60, 160, 200))
    assert result.side_by_side.shape[1] > image.shape[1]
    assert result.post_op.shape == image.shape
    assert result.gap_vector[0] ** 2 + result.gap_vector[1] ** 2 > 0


def test_metric_engine_specificity():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    specs = multiclass_specificity(y_true, y_pred, 3)
    assert specs.shape == (3,)
    report = evaluate_classifier(y_true, y_pred, class_names=["a", "b", "c"], model_name="VGG-16")
    assert "accuracy" in report and "gflops_256" in report


def test_sample_structure_writes_ten_classes():
    root = make_sample_structure(Config.DATASET_DIR / "classification", per_class=1)
    assert (root / "train" / "Avulsion Fracture").exists()
    assert (root / "val" / "Spiral Fracture").exists()


if __name__ == "__main__":
    test_anatomy_includes_lower_limb_and_spine()
    test_scale_rgb_is_256x256x3()
    test_refix_side_by_side_and_gap_vector()
    test_metric_engine_specificity()
    test_sample_structure_writes_ten_classes()
    print("ok")
