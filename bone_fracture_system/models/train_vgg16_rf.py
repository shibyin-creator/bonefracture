"""Training pipeline for VGG-16 Softmax and VGG-16 + Random Forest.

Faithful to Torne et al., IEEE Access 2025:
  - Input 256 x 256 x 3
  - Frozen VGG-16 convolutional base
  - Flatten + Dense(ReLU) + Dropout + Dense(ReLU) + Softmax
  - Adam lr=5e-4, categorical cross-entropy, 20 epochs, batch 32
  - Hybrid path: VGG-16 feature maps -> RandomForestClassifier

Dataset layout (ImageFolder):
    data/classification/train/<class_name>/*.png
    data/classification/val/<class_name>/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


def build_vgg16_softmax(num_classes: int, image_size: tuple[int, int] = (256, 256)):
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    base = VGG16(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False

    x = Flatten(name="flatten")(base.output)
    x = Dense(512, activation="relu", name="fc1")(x)
    x = Dropout(0.5, name="dropout")(x)
    x = Dense(256, activation="relu", name="fc2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base.input, outputs=outputs, name="vgg16_fracture_softmax")
    model.compile(
        optimizer=Adam(learning_rate=Config.VGG_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _image_generators(train_dir: Path, val_dir: Path, image_size: tuple[int, int], batch_size: int):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    train = train_gen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val = val_gen.flow_from_directory(
        str(val_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val


def extract_vgg_features(directory: Path, image_size: tuple[int, int] = (224, 224)):
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    base = VGG16(weights="imagenet", include_top=False, pooling="avg")
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow = gen.flow_from_directory(
        str(directory),
        target_size=image_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    features = base.predict(flow, verbose=1)
    labels = flow.classes
    class_indices = {v: k for k, v in flow.class_indices.items()}
    return features, labels, class_indices


def train_softmax(train_dir: Path, val_dir: Path, epochs: int, output: Path) -> None:
    train, val = _image_generators(train_dir, val_dir, Config.IMAGE_SIZE, Config.BATCH_SIZE)
    model = build_vgg16_softmax(num_classes=train.num_classes, image_size=Config.IMAGE_SIZE)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    Config.ensure_directories()
    ckpt = Config.LOG_DIR / "vgg16_softmax_best.h5"
    callbacks = [
        ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        TensorBoard(log_dir=str(Config.LOG_DIR / "tensorboard_vgg16")),
    ]
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    hist_path = output.with_suffix(".history.json")
    hist_path.write_text(json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2))
    mapping = {k: int(v) for k, v in train.class_indices.items()}
    (output.parent / "vgg16_class_indices.json").write_text(json.dumps(mapping, indent=2))
    print(f"Saved VGG-16 Softmax -> {output}")


def train_random_forest(train_dir: Path, val_dir: Path, output: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import joblib

    x_train, y_train, idx = extract_vgg_features(train_dir)
    x_val, y_val, _ = extract_vgg_features(val_dir)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    target_names = [idx[i] for i in sorted(idx)]
    print(classification_report(y_val, preds, target_names=target_names, zero_division=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output)
    # Store class names beside the forest for inference
    output.with_suffix(".classes.json").write_text(json.dumps(idx, indent=2))
    print(f"Saved VGG-16 + Random Forest -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGG-16 Softmax and VGG-16 + RF")
    parser.add_argument("--train-dir", type=Path, default=Config.DATASET_DIR / "classification" / "train")
    parser.add_argument("--val-dir", type=Path, default=Config.DATASET_DIR / "classification" / "val")
    parser.add_argument("--epochs", type=int, default=Config.VGG_EPOCHS)
    parser.add_argument("--mode", choices=["softmax", "rf", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_dir.exists():
        raise SystemExit(
            f"Training directory not found: {args.train_dir}\n"
            "Place ImageFolder data under data/classification/train/<class>/"
        )
    Config.ensure_directories()
    if args.mode in {"softmax", "both"}:
        train_softmax(args.train_dir, args.val_dir, args.epochs, Config.WEIGHT_FILES["vgg16"])
    if args.mode in {"rf", "both"}:
        train_random_forest(args.train_dir, args.val_dir, Config.WEIGHT_FILES["vgg16_rf"])


if __name__ == "__main__":
    main()
