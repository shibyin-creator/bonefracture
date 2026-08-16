# Bone Fracture CAD (OsteoScan)

Web CAD for **bone fracture detection and classification** from X-ray images.

Student: **Shibin Antony (811241013)**

## What this project adds

| Original abstract | Base paper (IEEE Access 2025) | This project |
|---|---|---|
| YOLOv8, 3,715 images, 7 body-part classes, mAP50 86%, Flask app | VGG-16 / RF / ResNet+SVM / EfficientNet+XGBoost, **1,129** images, 10 morphology classes, **no boxes** | YOLOv8 detection **+** morphology baselines, Flask UI, **≥ 20,000** image training plan |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Build a 20,000-image labeled corpus (phantoms; merge real data next)
python scripts/build_dataset.py --images 20000

# 2) Optional: train CPU tiny detector so the web app is immediately useful
python scripts/train_tiny.py --epochs 5 --limit 6000

# 3) Optional: full YOLOv8 (needs GPU for 50 epochs)
python scripts/train_yolo.py --epochs 50 --imgsz 256 --batch 16

# 4) Optional: base-paper hybrid classifiers on morphology labels
python scripts/train_baselines.py

# 5) Web app
python app.py
```

Open http://127.0.0.1:5000

## Dataset (≥ 20,000)

Bundled generator writes **20,000** YOLO-labeled 256×256 radiographs under `dataset/`.

For the **clinical** ≥20,000 set used in the report, merge:

- GRAZPEDWRI-DX (~20,327)
- MURA (~40,561)
- FracAtlas (~4,083)
- Roboflow 7-class YOLO set (~3,715; original abstract)
- Kaggle 10-class morphology (~1,129; base paper)

See `docs/PROPOSAL.md` and `python scripts/public_dataset_plan.py`.

## Layout

```
app.py                 Flask entry
config.py
scripts/               dataset, YOLO train, tiny CNN, baselines, eval
src/                   inference, preprocess
templates/ static/     HTML/CSS/JS
dataset/               images, labels, data.yaml (after build)
weights/               fracture_yolov8n.pt or tiny_detector.pt
docs/                  improved abstract + proposal
```
