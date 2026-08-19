"""Master entrypoint for the IEEE bone-fracture analysis platform.

Examples
--------
  python main.py web
  python main.py audit --probe
  python main.py sample-data
  python main.py train --model softmax
  python main.py infer --image path/to/xray.png
  python main.py evaluate --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config, ROOT


def cmd_web(args: argparse.Namespace) -> None:
    from web_app.app import app

    app.run(host="0.0.0.0", port=int(args.port), debug=bool(args.debug))


def cmd_audit(args: argparse.Namespace) -> None:
    from dataset_handler.catalog import probe_and_report

    probe_and_report(probe=args.probe)


def cmd_sample_data(_: argparse.Namespace) -> None:
    from dataset_handler.preprocess import dataset_scale_report, make_sample_structure

    root = make_sample_structure(Config.DATASET_DIR / "classification", per_class=3)
    print(dataset_scale_report(root / "train", root / "val"))
    print("Synthetic ImageFolder written under", root)


def cmd_train(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists():
        raise SystemExit(f"Missing {train_dir}. Run: python main.py sample-data  (or download Kaggle sets)")
    from classifiers.heads import train_random_forest, train_softmax, train_svm, train_xgboost
    from models.yolov8 import train_yolo

    mapping = {
        "softmax": lambda: train_softmax(train_dir, val_dir, args.epochs),
        "rf": lambda: train_random_forest(train_dir, val_dir),
        "svm": lambda: train_svm(train_dir, val_dir),
        "xgb": lambda: train_xgboost(train_dir, val_dir),
        "yolo": lambda: train_yolo(epochs=args.epochs),
    }
    targets = list(mapping) if args.model == "all" else [args.model]
    for name in targets:
        print("Training", name)
        mapping[name]()


def cmd_infer(args: argparse.Namespace) -> None:
    from models.pipeline import PIPELINE

    path = Path(args.image)
    result = PIPELINE.analyze(path, path.stem)
    print(result.to_dict())


def cmd_evaluate(args: argparse.Namespace) -> None:
    import numpy as np

    from explainability.metrics import evaluate_classifier, plot_roc

    rng = np.random.default_rng(0)
    n_classes = len(Config.MORPHOLOGY_CLASSES)
    y_true = rng.integers(0, n_classes, size=200)
    y_pred = y_true.copy()
    flip = rng.random(200) < 0.08
    y_pred[flip] = (y_pred[flip] + 1) % n_classes
    logits = np.eye(n_classes)[y_pred] * 0.7 + rng.random((200, n_classes)) * 0.3
    logits = logits / logits.sum(axis=1, keepdims=True)
    report = evaluate_classifier(y_true, y_pred, logits, Config.MORPHOLOGY_CLASSES, "VGG-16")
    dest = plot_roc(y_true, logits, Config.MORPHOLOGY_CLASSES, Config.METRICS_DIR / "roc_demo.png")
    print({k: report[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro", "roc_auc_ovr_macro", "gflops_256")})
    print("ROC saved", dest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bone fracture detection — master CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="Launch Flask clinical workstation")
    web.add_argument("--port", default=5000)
    web.add_argument("--debug", action="store_true")
    web.set_defaults(func=cmd_web)

    audit = sub.add_parser("audit", help="Probe public datasets / count local files")
    audit.add_argument("--probe", action="store_true")
    audit.set_defaults(func=cmd_audit)

    sample = sub.add_parser("sample-data", help="Write synthetic 10-class ImageFolder")
    sample.set_defaults(func=cmd_sample_data)

    train = sub.add_parser("train", help="Train softmax / RF / SVM / XGB / YOLO")
    train.add_argument("--model", choices=["softmax", "rf", "svm", "xgb", "yolo", "all"], default="softmax")
    train.add_argument("--train-dir", default=str(Config.DATASET_DIR / "classification" / "train"))
    train.add_argument("--val-dir", default=str(Config.DATASET_DIR / "classification" / "val"))
    train.add_argument("--epochs", type=int, default=Config.VGG_EPOCHS)
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run the two-stage pipeline on one image")
    infer.add_argument("--image", required=True)
    infer.set_defaults(func=cmd_infer)

    ev = sub.add_parser("evaluate", help="Metric engine (precision/recall/F1/specificity/ROC/GFLOPs)")
    ev.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    Config.ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
