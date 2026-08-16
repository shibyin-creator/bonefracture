# Medical Image Analysis for Bone Fracture Detection Using Deep Learning

Production Flask application that combines the **YOLOv8** detector from the project abstract with the **IEEE Access 2025** classification stack (VGG-16 Softmax, VGG-16 + Random Forest, ResNet-50 + SVM, EfficientNetB0 + XGBoost), Grad-CAM explainability, and an educational surgical refixation overlay.

## Capabilities

- Real-time localization (bounding boxes) with YOLOv8
- Multi-class fracture typing (10 morphological classes + 8 anatomical sites)
- Hybrid ensembles matching Torne et al., *IEEE Access*, 2025 (VGG-16 / VGG-16+RF reported at **0.95** accuracy; YOLOv8 abstract **mAP50 = 0.86**)
- Grad-CAM heatmaps of suspected fracture regions
- Automatic post-surgical refixation demo for severe / displaced patterns (plates, screws, IM rods)
- Admin authentication, user management, model-weight registry, audit logs
- Clinical dashboard with dark / light mode and side-by-side original · YOLO · Grad-CAM · refixation views
- Training on local disks, Jupyter, or Google Colab (`Fracture_Detection_Colab.ipynb`), scaled toward 20k+ radiographs

## Quick start

```bash
cd bone_fracture_system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` and sign in with the bootstrap administrator:

- Username: `admin`
- Password: `Admin@123`

Change `SECRET_KEY` and `ADMIN_PASSWORD` in the environment before any clinical deployment.

Without trained weights the API still runs in **demo mode** (OpenCV preprocessing, saliency heatmap, heuristic class prior). Drop artifacts into `weights/` after training:

| File | Model |
| --- | --- |
| `vgg16_softmax.h5` | VGG-16 + Softmax |
| `vgg16_random_forest.joblib` | VGG-16 features + Random Forest |
| `resnet50_svm.joblib` | ResNet-50 + linear SVM |
| `efficientnetb0_xgboost.joblib` | EfficientNetB0 + XGBoost |
| `yolov8_fracture.pt` | YOLOv8 detector |

## Training

ImageFolder (classification):

```text
data/classification/train/<class>/*.png
data/classification/val/<class>/*.png
```

```bash
python models/train_vgg16_rf.py --mode both
python models/train_ensembles.py --mode both
```

YOLO detection (Ultralytics):

```text
data/yolo/images/{train,val}
data/yolo/labels/{train,val}
```

```bash
python models/train_yolov8.py --model yolov8n.pt --epochs 80
```

Google Colab / Jupyter: open `Fracture_Detection_Colab.ipynb`.

## Fracture classes

**Morphology (paper):** Avulsion, Comminuted, Fracture-Dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral.

**Anatomy (abstract + pelvis):** Elbow Positive, Fingers Positive, Forearm Fracture, Humerus, Humerus Fracture, Shoulder Fracture, Wrist Positive, Pelvic Fracture.

Severe classes automatically open the refixation module: comminuted, dislocation, spiral, impacted, pathological, humerus / forearm / shoulder / pelvic fractures.

## Safety notice

This software is a research and education system. It is **not** a medical device, **not** FDA/CE cleared, and must not be used as the sole basis for diagnosis or operative planning. All outputs require review by a licensed clinician.

## Project layout

```text
bone_fracture_system/
├── app.py
├── config.py
├── db.py
├── models/
│   ├── inference.py
│   ├── gradcam.py
│   ├── refixation.py
│   ├── train_vgg16_rf.py
│   ├── train_yolov8.py
│   └── train_ensembles.py
├── templates/
├── static/
└── Fracture_Detection_Colab.ipynb
```
