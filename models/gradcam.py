"""Grad-CAM heatmap engine (Keras GradientTape + OpenCV fallback).

Last convolutional block of VGG-16 is ``block5_conv3``. When TensorFlow is
unavailable the OpenCV saliency heatmap still produces a clinical overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def overlay_heatmap(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_norm = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    resized = cv2.resize(color, (bgr.shape[1], bgr.shape[0]))
    return cv2.addWeighted(bgr, 1.0 - alpha, resized, alpha, 0)


def opencv_saliency_heatmap(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 40, 120)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    high_freq = cv2.absdiff(enhanced, blur)
    cam = 0.45 * edges.astype(np.float32) + 0.55 * high_freq.astype(np.float32)
    cam = cv2.GaussianBlur(cam, (21, 21), 0)
    return overlay_heatmap(image_bgr, cam)


def generate_gradcam(
    image_bgr: np.ndarray,
    keras_model: Any | None = None,
    class_index: int | None = None,
    last_conv_name: str = "block5_conv3",
) -> np.ndarray:
    if keras_model is None:
        return opencv_saliency_heatmap(image_bgr)
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.vgg16 import preprocess_input
    except Exception:
        return opencv_saliency_heatmap(image_bgr)

    img = cv2.resize(image_bgr, (224, 224))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0))
    try:
        conv_layer = keras_model.get_layer(last_conv_name)
        grad_model = tf.keras.models.Model([keras_model.inputs], [conv_layer.output, keras_model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled, conv_outputs[0]), axis=-1)
        cam = tf.nn.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return overlay_heatmap(image_bgr, cam)
    except Exception:
        return opencv_saliency_heatmap(image_bgr)


def save_heatmap(image_bgr: np.ndarray, destination: Path, **kwargs: Any) -> Path:
    overlay = generate_gradcam(image_bgr, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay)
    return destination
