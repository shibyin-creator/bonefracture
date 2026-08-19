"""Explainability: Grad-CAM heatmaps and the clinical metric engine."""

from explainability.gradcam import generate_gradcam, opencv_saliency_heatmap, save_heatmap
from explainability.metrics import evaluate_classifier, plot_roc, scale_gflops

__all__ = [
    "evaluate_classifier",
    "generate_gradcam",
    "opencv_saliency_heatmap",
    "plot_roc",
    "save_heatmap",
    "scale_gflops",
]
