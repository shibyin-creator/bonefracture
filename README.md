# Medical Image Analysis for Bone Fracture Detection Using Deep Learning

IEEE Access 2025 classification stack (VGG-16 Softmax, VGG-16 + Random Forest, ResNet-50 + SVM, EfficientNetB0 + XGBoost) plus YOLOv8 localization (abstract **mAP50 = 0.86**), Grad-CAM, orthopedic refixation simulation, and a Flask clinical workstation.

## Repository layout

```text
├── data/
│   ├── audit_datasets.py        # HTTP prober, Kaggle/Figshare downloader
│   ├── DATASETS.md              # 1,129 10-class set + GRAZPEDWRI-DX 20,327
│   ├── classification/          # ImageFolder train/val · 256×256×3
│   └── yolo/                    # images/, labels/, data.yaml · 640²
├── models/
│   ├── train_vgg16_rf.py        # VGG-16 Softmax & VGG-16 + Random Forest
│   ├── train_ensembles.py       # ResNet-50 + SVM, EfficientNetB0 + XGBoost
│   ├── train_yolov8.py          # YOLOv8 fine-tune (yaml + loop)
│   └── gradcam.py               # Grad-CAM (Keras / OpenCV)
├── classifiers/                 # Hybrid heads + joblib wrappers
├── refix_simulation/
│   └── orthopedics_refix.py     # Alignment vectors + plate/screw/IM overlay
├── web_app/                     # Flask · index.html, login.html, dashboard.html
├── notebooks/
│   └── fracture_detection_pipeline.ipynb
├── weights/
├── config.py
├── requirements.txt
└── main.py
```

**Research & Diagnostic Support Tool Only — Not an Authorized Medical Device.**

### Categorical cross-entropy (paper Eq. 1)

\[
\mathrm{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\log(p_{i,c})
\]

### mAP50 (YOLOv8)

\[
\mathrm{mAP}_{50} = \frac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_{c}\quad(\mathrm{IoU}=0.50)
\]

## Quick start (dataset included)

```bash
pip install -r requirements.txt
python data/install_offline_dataset.py   # unpacks 4,083 FracAtlas X-rays already in the repo (no Kaggle, no extra download)
python main.py web --port 5000
```

GitHub cannot store the 15 GB GRAZPEDWRI-DX image pack (100 MB file cap). FracAtlas **is** in this repo as `data/offline_bundles/FracAtlas.zip.part00` … `part04` (~323 MB). After clone, the install script joins the parts locally. Details: [`data/DATASETS.md`](data/DATASETS.md).

Without unpacking FracAtlas you can still run a tiny demo:

```bash
python main.py sample-data          # tiny synthetic 10-class folders so the pipeline runs
python main.py evaluate             # metric engine demo (ROC + GFLOPs JSON)
python main.py web --port 5000
```

Sign-in:

| Role | Username | Password |
| --- | --- | --- |
| Admin | `admin` | `Admin@123` |
| Clinician | `clinician` | `Clinic@123` |

Admin panel (retrain command, users, audit, weights) is **admin-only**. The clinical portal uploads X-rays and shows YOLO / Grad-CAM / pre-op vs post-op refixation.

## Datasets (>20,000 scale)

Streaming generators support **>20k train and >20k val**. The IEEE 10-class Kaggle set is 1,129 images; 20k scale is [GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193) (20,327 wrist radiographs) plus YOLO + FracAtlas.

```bash
python main.py audit --probe
# Kaggle token in ~/.kaggle/kaggle.json then:
python bone_fracture_system/data/audit_datasets.py --download bone_break_classification
python bone_fracture_system/data/audit_datasets.py --download bonefracture_yolo8
```

Details: `bone_fracture_system/data/DATASETS.md`. Dashboard: `/datasets`.

Anatomical sites: Elbow, Fingers, Forearm, Humerus, Shoulder, Wrist, **Femur, Knee, Ankle, Spine**. Morphology: Avulsion, Comminuted, Dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral.

## Training

```bash
python main.py train --model softmax
python main.py train --model rf
python main.py train --model svm
python main.py train --model xgb
python main.py train --model yolo
python main.py infer --image path/to/xray.png
```

Colab: `notebooks/Bone_Fracture_Detection_Master.ipynb`.

## Presentation pack

| Document | Use |
| --- | --- |
| [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md) | 10–12 min talk + live demo script |
| [`docs/BONE_COVERAGE_ATLAS.md`](docs/BONE_COVERAGE_ATLAS.md) | **How many bones/fractures we can test** |
| [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md) | Printable handout |
| [`docs/Q_AND_A.md`](docs/Q_AND_A.md) | Examiner questions |
| [`docs/Bone_Coverage_Presentation.pptx`](docs/Bone_Coverage_Presentation.pptx) | 8 slides on body coverage |
| Academic 14-slide deck | `bone_fracture_system/docs/Bone_Fracture_Detection_Presentation.pptx` |

**Coverage in one line:** not all 206 bones — **10 fracture patterns** on **11 regions** (~42 named bones). Live page after login: `/coverage`.

Research/education software. **Not** a medical device. Outputs require licensed clinical review.
