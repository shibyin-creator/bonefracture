"""Flask CAD app: upload X-ray → detect/classify fracture → show overlay."""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    DETECTION_CLASSES,
    MAX_CONTENT_MB,
    MORPHOLOGY_CLASSES,
    RESULT_DIR,
    SECRET_KEY,
    UPLOAD_DIR,
)
from src.inference import FractureEngine, annotate
from src.preprocess import prepare_xray

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ENGINE: FractureEngine | None = None


def get_engine() -> FractureEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = FractureEngine()
    return ENGINE


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    engine = get_engine()
    return render_template(
        "index.html",
        detection_classes=DETECTION_CLASSES,
        morphology_classes=MORPHOLOGY_CLASSES,
        engine_mode=engine.mode,
    )


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Please choose an X-ray image.")
        return redirect(url_for("index"))
    if not allowed(file.filename):
        flash("Allowed types: png, jpg, jpeg, bmp, webp.")
        return redirect(url_for("index"))

    stem = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    raw_path = UPLOAD_DIR / stem
    file.save(raw_path)
    bgr = cv2.imread(str(raw_path))
    if bgr is None:
        flash("Could not read that image.")
        return redirect(url_for("index"))

    prepared = prepare_xray(bgr, 512)
    engine = get_engine()
    detections = engine.predict(prepared)
    overlay = annotate(prepared, detections)
    out_name = f"result_{Path(stem).stem}.jpg"
    out_path = RESULT_DIR / out_name
    cv2.imwrite(str(out_path), overlay)

    return render_template(
        "result.html",
        result_image=url_for("static", filename=f"results/{out_name}"),
        detections=detections,
        engine_mode=engine.mode,
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        detection_classes=DETECTION_CLASSES,
        morphology_classes=MORPHOLOGY_CLASSES,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
