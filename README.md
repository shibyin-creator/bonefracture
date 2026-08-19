# Medical Image Analysis for Bone Fracture Detection Using Deep Learning

IEEE Access 2025 classification stack (VGG-16 Softmax, VGG-16 + Random Forest, ResNet-50 + SVM, EfficientNetB0 + XGBoost) plus YOLOv8 localization (abstract **mAP50 = 0.86**), Grad-CAM, orthopedic refixation simulation, and a Flask clinical workstation.

## Repository layout

```text
├── dataset_handler/      # 256×256×3 scaling, augmentation, balancing, >20k streaming, catalog audit
├── models/               # YOLOv8, VGG-16, ResNet-50, EfficientNetB0 + two-stage pipeline
├── classifiers/          # Dense-Softmax, Random Forest, SVM, XGBoost heads
├── explainability/       # Grad-CAM + precision/recall/F1/specificity/accuracy/GFLOPs/ROC
├── refix_simulation/     # Gap realignment vectors + plate/screw/IM-rod overlay + pre/post panel
├── web_app/              # Flask + HTML5/CSS3 medical dashboard (admin vs clinician RBAC)
├── notebooks/            # Bone_Fracture_Detection_Master.ipynb (Colab / Jupyter)
├── config.py
├── requirements.txt
└── main.py
```

### Categorical cross-entropy (paper Eq. 1)

\[
\mathrm{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\log(p_{i,c})
\]

### mAP50 (YOLOv8)

\[
\mathrm{mAP}_{50} = \frac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_{c}\quad(\mathrm{IoU}=0.50)
\]

## Quick start

```bash
pip install -r requirements.txt
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
