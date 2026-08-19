"""Clinical metric engine: precision, recall, F1, specificity, accuracy, GFLOPs, ROC-AUC.

Definitions (per class, then macro-averaged):

    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)          (sensitivity)
    Specificity = TN / (TN + FP)
    F1          = 2PR / (P + R)
    Accuracy    = (TP + TN) / (TP+TN+FP+FN) for binary; overall correct / N for multi-class

mAP50 (detector): mean of per-class AP at IoU = 0.50.

GFLOPs: theoretical multiply-adds × 1e-9. VGG-16 ≈ 15.5 at 224²; scaled by (H/224)².
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import Config

THEORETICAL_GFLOPS = {
    "VGG-16": 15.5,
    "ResNet-50": 4.1,
    "EfficientNetB0": 0.39,
    "YOLOv8n": 8.7,
}


def scale_gflops(base_224: float, height: int = 256) -> float:
    return round(base_224 * (height / 224.0) ** 2, 4)


def multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    specs = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(len(y_true) - tp - fn - fp)
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.asarray(specs, dtype=np.float64)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    model_name: str = "VGG-16",
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    names = class_names or [str(i) for i in labels]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names[: len(labels)], output_dict=True, zero_division=0
    )
    specs = multiclass_specificity(y_true, y_pred, max(labels) + 1 if labels else 1)
    roc = None
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        try:
            roc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc = None
    payload = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(np.mean(specs)) if len(specs) else 0.0,
        "roc_auc_ovr_macro": roc,
        "gflops_224": THEORETICAL_GFLOPS.get(model_name.split("+")[0].strip(), THEORETICAL_GFLOPS["VGG-16"]),
        "gflops_256": scale_gflops(THEORETICAL_GFLOPS.get("VGG-16", 15.5), Config.IMAGE_SIZE[0]),
        "per_class": report,
        "published": Config.PUBLISHED_METRICS.get(model_name),
    }
    Config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    dest = Config.METRICS_DIR / f"{model_name.replace(' ', '_').replace('+', 'plus')}.json"
    dest.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str], dest: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names[:n_classes]):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC")
    ax.legend(fontsize=7, loc="lower right")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
