# How to run (Shibin Antony — 811241013)

## 1. Unzip and install

```bash
unzip OsteoScan_BoneFractureCAD_20k.zip -d OsteoScan
cd OsteoScan
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the zip you received is **code-only**, build the 20,000-image set first:

```bash
python scripts/build_dataset.py --images 20000
python scripts/train_tiny.py --epochs 5 --limit 6000
```

## 2. Start the web app

```bash
python app.py
```

Browser: http://127.0.0.1:5000  
Upload any X-ray (or a file from `dataset/images/test/`).

## 3. Train YOLOv8 (GPU recommended)

```bash
python scripts/train_yolo.py --epochs 50 --imgsz 256 --batch 16
```

Weights copy to `weights/fracture_yolov8n.pt`. Restart Flask; engine mode becomes `yolov8`.

## 4. Base-paper hybrid classifiers (10 morphology classes)

```bash
python scripts/train_baselines.py
```

## 5. Real ≥20,000 clinical images

Download GRAZPEDWRI-DX, MURA, FracAtlas, Roboflow 7-class, Kaggle 10-class (see `docs/PROPOSAL.md`).  
Put YOLO folders in `dataset/external/` then:

```bash
python scripts/merge_external.py
python scripts/train_yolo.py --epochs 50
```
EOF