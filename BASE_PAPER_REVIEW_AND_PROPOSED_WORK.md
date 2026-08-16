# Bone Fracture X-Ray Classification

## Base Paper Review and New Project Proposal

**Base paper:** Spoorthy Torne et al., “VGG-16, VGG-16 With Random Forest, Resnet50 With SVM, and EfficientNetB0 With XGBoost—Enhancing Bone Fracture Classification in X-Ray Using Deep Learning Models,” *IEEE Access*, vol. 13, pp. 25568–25577, 2025.  
DOI: [10.1109/ACCESS.2025.3534818](https://doi.org/10.1109/ACCESS.2025.3534818)

**Dataset cited in the paper:** [Bone Break Classification Image Dataset (Kaggle)](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset)

---

## 1. Paper identity (full bibliographic details)

| Item | Detail |
|---|---|
| Type | Open-access journal article (IEEE Access) |
| Received / accepted / published | 8 Jan 2025 / 21 Jan 2025 / 27 Jan 2025 (current version 10 Feb 2025) |
| Corresponding author | Dasharathraj K. Shetty (Manipal Institute of Technology) |
| Affiliations | RV University; MIT Manipal; KMC Manipal (Orthopedics, Emergency Medicine, Radiodiagnosis); Sikkim Manipal Institute of Technology |
| License | CC BY 4.0 |
| Keywords | Bone fracture, ensemble learning, fracture classification, medical imaging diagnostics, Random Forest, ResNet-50, VGG-16 |

---

## 2. What the paper actually does (plain summary)

The paper is a **10-class X-ray fracture-type classifier**, not a fracture detector and not an anatomical-site classifier.

It compares four pipelines:

1. **VGG-16 (transfer learning)** — frozen ImageNet backbone + custom dense head (flatten, two ReLU dense layers, dropout, SoftMax).
2. **VGG-16 + Random Forest** — VGG-16 used as a feature extractor; flattened feature vectors go to a Random Forest.
3. **ResNet-50 + linear SVM** — ResNet-50 features → SVM.
4. **EfficientNetB0 + XGBoost** — EfficientNetB0 features → XGBoost.

Reported headline numbers:

| Model | Approx. overall accuracy / macro-F1 |
|---|---|
| VGG-16 | **0.95** |
| VGG-16 + Random Forest | **0.95** |
| ResNet-50 + SVM | **0.93** |
| EfficientNetB0 + XGBoost | **0.41–0.43** |

The authors conclude that **VGG-16 based models are best** for this dataset and that EfficientNetB0 + XGBoost is a poor fit.

---

## 3. Clinical / problem background (as stated)

- Fractures come from trauma (falls, RTA, sports) or weakened bone (osteoporosis).
- Manual reading by radiologists is time-consuming and error-prone.
- Prior DL work (e.g. femoral-neck fracture CNNs ~83% detection) shows promise, but **which backbone + which classical head is best for multi-type fracture classification is still under-explored**.
- **Hypothesis:** transfer-learning CNNs combined with ensemble/classical classifiers will beat standalone CNNs.

---

## 4. Dataset details from the paper

| Item | Paper claim |
|---|---|
| Source | Kaggle bone-break classification set |
| Total images | **1,129** |
| Classes | **10** fracture morphologies (see below) |
| Train / test | **989 / 140** |
| Train / val split | **80 / 20** of the training portion |
| Input size | **256 × 256 × 3** |
| Image type | Grayscale X-rays stored / fed as 3-channel |
| Balance | **Imbalanced** (comminuted and fracture-dislocation over-represented; longitudinal and oblique under-represented) |

**Ten classes**

1. Avulsion  
2. Comminuted  
3. Fracture-dislocation  
4. Greenstick  
5. Hairline  
6. Impacted  
7. Longitudinal  
8. Oblique  
9. Pathological  
10. Spiral  

**Important extra observation from the paper:** some images contain laterality markers (e.g. letter “D” for dextro). That is a known shortcut risk: a model can learn the marker instead of the fracture.

---

## 5. Methodology (reproducible checklist)

### 5.1 Conceptual pipeline

```
X-ray image (256×256×3)
        │
        ▼
Pretrained CNN without top classifier
(VGG-16 / ResNet-50 / EfficientNetB0)
        │
        ▼
Feature maps → flatten / global pool → feature vector
        │
        ├── VGG-16: custom dense + SoftMax (end-to-end transfer learning)
        ├── VGG-16 features → Random Forest
        ├── ResNet-50 features → linear SVM
        └── EfficientNetB0 features → XGBoost
        │
        ▼
Precision, Recall, F1, Accuracy, ROC-AUC, Grad-CAM
```

### 5.2 VGG-16 transfer-learning training setup

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | **0.0005** |
| Loss | Categorical cross-entropy |
| Epochs | **20** |
| Batch size | **32** |
| Output | 10-way SoftMax |
| Extra | ModelCheckpoint + TensorBoard |
| Backbone | Frozen ImageNet weights; new head trained |

Categorical cross-entropy used:

\[
\mathrm{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\log(p_{i,c})
\]

### 5.3 Hybrid “ensemble” heads

- **Random Forest:** many decision trees on VGG features (reduces overfitting vs a single tree).
- **SVM:** linear kernel on high-dimensional ResNet features.
- **XGBoost:** gradient boosting on EfficientNet pooled features.

Note: these are **hybrid CNN + classical ML** models. They are not a true ensemble of several deep models (no stacking/voting of VGG + ResNet + EfficientNet).

### 5.4 Evaluation metrics

Accuracy, precision, recall (sensitivity), F1, support, macro average, weighted average, specificity, ROC-AUC per class, Grad-CAM heatmaps, plus a compute table (parameters and GFLOPs). VGG-16 is reported as **40.11 GFLOPs** and is called more expensive but more accurate than ResNet-50 and EfficientNetB0.

---

## 6. Results (what they report)

### 6.1 Classification

- VGG-16 and VGG-16+RF: **~95%** accuracy; high precision/recall/F1 across classes; ROC AUC near **1.00**.
- They claim **100% sensitivity and specificity** for avulsion, dislocation, and hairline on their test split.
- ResNet-50+SVM: **~93%**, still strong AUC.
- EfficientNetB0+XGBoost: **~41%**; impacted-fracture recall as low as **6%**. Authors blame:
  - EfficientNet not suiting small grayscale images
  - scaling-focused architecture missing fine fracture texture
  - XGBoost not matching convolutional features well

### 6.2 Explainability

Grad-CAM overlays for all 10 types (spiral, pathological, oblique, longitudinal, impacted, hairline, greenstick, fracture-dislocation, comminuted, avulsion). They argue the model attends to clinically relevant bone regions.

### 6.3 Real-world application claims

- Plug into hospital PACS / imaging workflow.
- Instant fracture-type label for emergency care.
- Need GPU/cloud, HIPAA-style privacy, continuous retraining.
- They cite a survey: ~73.68% of practitioners think AI reduces diagnostic error; ~65.55% think it improves accuracy.

---

## 7. Paper limitations and research gaps (critical review)

These gaps are the **justification for a new project**. Use them in your problem statement.

| Gap | Why it matters |
|---|---|
| **Very small dataset (1,129 images, 140 test images)** | 95% accuracy on 140 images is statistically fragile. One or two mislabels swing metrics. Near-duplicates in Kaggle sets often inflate accuracy. |
| **Class imbalance not treated** | No class weights, focal loss, oversampling, or stratified reporting of confidence intervals. Rare classes (oblique, longitudinal) are clinically important. |
| **No “normal / no-fracture” class** | Real CAD systems must first say *fracture vs no fracture*, then type. This paper assumes every image is already a fracture. |
| **Classification only, no localization** | Radiologists need *where* the fracture is (bounding box / mask), not only a global label. |
| **No anatomical site label** | “Hairline” on a wrist vs femur is a different clinical problem. Morphology-only labels are incomplete. |
| **Suspicious EfficientNet collapse (41%)** | A modern backbone should not fail this badly unless features were extracted wrongly (no fine-tune, wrong pooling, unscaled features, leakage, or buggy pipeline). Needs a controlled re-run. |
| **Hybrid models called “ensemble”** | True ensembles (soft voting, stacking, snapshot ensembling) were not tested. |
| **Little preprocessing detail** | No CLAHE, bone-windowing, marker removal, laterality stripping, or documented augmentation policy. Marker “D” is a shortcut risk. |
| **No external validation** | Single Kaggle source. No MURA, FracAtlas, GRAZPEDWRI-DX, or hospital hold-out. |
| **No radiologist comparison / inter-rater study** | Cannot claim clinical reliability. |
| **No uncertainty / calibration** | SoftMax 0.99 on a wrong class is dangerous in ER. |
| **No statistical tests** | No McNemar, bootstrap CI, or paired significance between 0.95 and 0.93. |
| **Related-work mismatch** | Several citations are off-topic (batik, rice disease, facial emotion, DVT, inundation mapping). Literature review of *fracture* CAD is thin. |
| **Compute vs accuracy** | VGG-16 is heavy (40+ GFLOPs). Clinical edge devices need lighter models (MobileNet, EfficientNet properly trained, TinyViT). |
| **Ethics / consent** | Public Kaggle images; no demographic breakdown (age, pediatric vs adult, device vendor). Greenstick is pediatric-specific — mixing ages without stratification is a confounder. |

**Bottom line:** the paper is a useful **baseline comparison of CNN feature extractors + classical heads** on a public 10-class fracture set. It is **not** a clinically validated CAD system. A new project should keep the same clinical goal but close the gaps above.

---

## 8. Extra technical details you can cite in your report

### 8.1 Why each backbone was chosen (theory you should write)

- **VGG-16:** 13 conv + 3 FC historically; 3×3 filters stacked for a large receptive field. Good at texture. Heavy and no residual connections. ImageNet-pretrained low-level edges transfer well to X-ray edges.
- **ResNet-50:** skip connections ease vanishing gradients; 50 layers; often better on larger, more diverse data. Paper used it only as a frozen feature extractor + SVM.
- **EfficientNetB0:** compound scaling of depth/width/resolution; few parameters; needs careful input resolution, augmentation, and fine-tuning. Paper’s poor result is likely an **implementation / training** issue, not proof that EfficientNet is unfit for X-rays.
- **Random Forest:** bagging, Gini/entropy splits, handles noisy features, less sensitive to feature scale.
- **SVM (linear):** max-margin in high dimension; works when CNN features are linearly separable.
- **XGBoost:** sequential residual trees; needs feature scaling and careful `max_depth` / `learning_rate` / class weights.

### 8.2 Medical definitions of the 10 classes (write these in Chapter 2)

| Type | Clinical meaning | Extra note |
|---|---|---|
| Avulsion | Ligament/tendon pulls a bone fragment | Often at apophyses; easy to miss |
| Comminuted | Bone broken into ≥3 pieces | High-energy trauma |
| Fracture-dislocation | Fracture + joint displacement | Needs urgent reduction |
| Greenstick | Incomplete, pediatric, cortex bent | Age-specific |
| Hairline | Thin crack, often occult | High miss rate on plain film |
| Impacted | Fragments driven into each other | Alignment may look “normal” |
| Longitudinal | Crack along long axis of bone | Rare in many datasets |
| Oblique | Diagonal to long axis | Distinguish from spiral |
| Pathological | Through diseased bone (tumor, osteoporosis, infection) | Needs workup of underlying disease |
| Spiral | Rotational force, helical fracture line | Classic twisting injury |

### 8.3 Standard extra metrics for medical imaging (paper used some, missed others)

Include in the new project:

- Sensitivity / specificity / PPV / NPV  
- ROC-AUC and **PR-AUC** (better under imbalance)  
- Cohen’s kappa vs radiologist  
- Expected Calibration Error (ECE)  
- Inference time (ms/image) and model size (MB)  
- 95% confidence intervals  

### 8.4 Related public datasets (for extension, not used in the base paper)

| Dataset | What it adds |
|---|---|
| **MURA** (Stanford) | Upper-extremity X-rays, abnormal vs normal, multi-view |
| **FracAtlas** | Fracture detection with bounding boxes |
| **GRAZPEDWRI-DX** | Pediatric wrist trauma, fine-grained labels |
| **RSNA pediatric bone age / fracture challenges** | Larger, better annotated |
| Hospital PACS export | True external test (best if available) |

---

## 9. Proposed new project (recommended title and scope)

### Recommended title

**Explainable Multi-Task Deep Learning for Bone Fracture Detection, Localization, and Morphological Classification on X-Ray Images**

### One-line problem statement

Build a clinically oriented CAD pipeline that (1) detects whether a fracture is present, (2) localizes it, (3) classifies morphological type, and (4) explains the decision — using stronger training practice than the IEEE Access 2025 baseline.

### Objectives

1. Reproduce the base paper’s four models as **baselines** on the same Kaggle set (fair re-implementation).
2. Fix training issues: stratified splits, class imbalance, augmentation, laterality-marker suppression, CLAHE.
3. Replace or add modern backbones: **ConvNeXt-Tiny / EfficientNetV2-S / Swin-T / ViT-B16**.
4. Add a **two-stage clinical pipeline**:
   - Stage A: Fracture vs Normal (and optional body-part classifier).
   - Stage B: 10-class morphology (only if fracture-positive).
5. Add **localization**: YOLOv8 / RT-DETR / Faster R-CNN or weakly supervised CAM + bounding-box datasets (FracAtlas).
6. True **ensemble**: weighted soft-voting or stacking of top-3 CNNs (not only CNN+RF).
7. **Explainability + uncertainty**: Grad-CAM++, Score-CAM, confidence threshold, “refer to radiologist” when entropy is high.
8. Optional web demo: upload X-ray → overlay + class probabilities.

### Novelty vs the base paper (write this in “Contribution”)

| Base paper | Proposed work |
|---|---|
| 10-class only | Detection + classification (+ localization if boxes exist) |
| Frozen CNN + classical ML | Fine-tuned CNNs/transformers + optional hybrid head |
| No imbalance handling | Focal loss, class weights, balanced sampling |
| One Kaggle source | Same set + at least one external test set |
| Grad-CAM only | Grad-CAM++ / Score-CAM + failure-case analysis |
| Accuracy-centric | Calibration, PR-AUC, CI, inference cost |
| VGG declared winner | Controlled study of why EfficientNet failed; likely it can be competitive if trained properly |

---

## 10. Proposed system architecture (new)

```
                    ┌──────────────────────────┐
                    │  Preprocessing            │
                    │  - CLAHE / normalize      │
                    │  - Resize 224/256/384     │
                    │  - Marker / text inpaint  │
                    │  - Augment (train only)   │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     Backbone A            Backbone B          Detector
     (ConvNeXt /           (EfficientNetV2     (YOLOv8-n)
      Swin-T)               or ResNet50)
              │                  │                  │
              └────────┬─────────┘                  │
                       ▼                            ▼
              Shared representation          Bounding box
                       │                     on fracture site
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   Head 1:        Head 2:        Head 3:
   Fracture?      Morphology     Body part
   (2-class)      (10-class)     (optional)
         │             │              │
         └─────────────┴──────────────┘
                       ▼
         Calibration + reject option
                       ▼
         Grad-CAM overlay + report PDF/UI
```

### Preprocessing extras (do these; the paper did not)

- Histogram equalization / **CLAHE** on bone windows.  
- Convert true grayscale to 3-channel by stacking, or train a 1-channel first conv.  
- Random rotation ±15°, shift, scale, contrast, mild Gaussian noise (X-ray realistic).  
- **Do not** use aggressive color jitter or cutout that erases the fracture line.  
- Detect and mask burned-in letters (OCR or simple morphological filter) so the model cannot cheat on “D/L/R”.  
- Stratified 70/15/15 or 5-fold **patient-level** split if IDs exist; otherwise hash filenames and check near-duplicates (pHash).

### Training extras

- Fine-tune last blocks, not frozen-only.  
- Cosine LR decay, label smoothing 0.05.  
- Focal loss or class-weighted CCE.  
- Early stopping on val PR-AUC, not only accuracy.  
- MixUp **off** or very weak (can blur hairline cracks).  
- Seed reporting (42 / 0 / 123) for three runs.

### Proposed model zoo (keep it implementable)

**Must-have (core project)**

1. Reproduced VGG-16 (paper baseline)  
2. Reproduced VGG-16 + RF  
3. Fine-tuned EfficientNetB0 (to test the paper’s failure claim)  
4. Fine-tuned ResNet-50  
5. ConvNeXt-Tiny or MobileNetV3 (efficient clinical candidate)

**Should-have**

6. Soft-voting ensemble of top 3  
7. YOLOv8-nano on FracAtlas (detection demo)  
8. Streamlit/Flask UI  

**Could-have (if time)**

9. Swin Transformer  
10. Federated / differential-privacy paragraph only (theory)  
11. On-device TFLite export  

---

## 11. Proposed evaluation plan (extra, examiners like this)

### Experiments

| ID | Experiment | Purpose |
|---|---|---|
| E0 | Reproduce paper 4 models | Fair baseline |
| E1 | Same backbones + fine-tune + augmentation | Show paper under-trained EfficientNet |
| E2 | Class weights / focal loss | Imbalance |
| E3 | Marker-masked vs raw images | Shortcut learning test |
| E4 | Modern backbone vs VGG-16 | Accuracy vs GFLOPs |
| E5 | Ensemble vs best single | True ensemble gain |
| E6 | External dataset zero-shot / fine-tune | Generalization |
| E7 | Grad-CAM qualitative + pointing game if boxes | Trust |
| E8 | Calibration (temperature scaling) | Clinical safety |

### Target (realistic, not 99%)

On a **clean, de-duplicated, stratified** test set, aim for:

- Fracture vs normal: sensitivity **≥ 0.90**, specificity **≥ 0.85** (if a normal class is added).  
- 10-class morphology: macro-F1 **better than reproduced baseline**, with CI reported.  
- Do **not** promise 95% if your split is stricter than the paper.

---

## 12. Proposed modules for implementation (student project map)

| Module | Contents |
|---|---|
| `data/` | Download scripts, class folders, duplicate filter, stratified split JSON |
| `preprocess/` | CLAHE, resize, marker mask |
| `models/` | VGG16, ResNet50, EfficientNet, ConvNeXt, RF/SVM/XGB heads |
| `train/` | PyTorch or TensorFlow trainer, configs (YAML) |
| `eval/` | Classification report, ROC, PR, confusion matrix, bootstrap CI |
| `explain/` | Grad-CAM++ overlays |
| `detect/` | Optional YOLO |
| `app/` | Streamlit: upload, predict, heatmap |
| `notebooks/` | EDA of class counts, sample images, failure cases |
| `docs/` | This file + IEEE-style project report chapters |

**Suggested stack:** Python 3.10+, PyTorch 2.x (or TensorFlow/Keras to stay close to the paper), scikit-learn, XGBoost, Albumentations, OpenCV, Streamlit, Grad-CAM library.

---

## 13. Proposed report / thesis chapter outline

1. Introduction — burden of fractures, miss rate of occult fractures, need for CAD.  
2. Literature survey — CNNs for fracture (femur, wrist, ribs), transformers in X-ray, detection datasets, **and a dedicated critique of Torne et al. 2025**.  
3. Dataset and ethics — Kaggle + extra set; limitations of public data.  
4. Methodology — preprocessing, models, losses, training, explainability.  
5. Experiments and results — tables vs baseline paper.  
6. Discussion — why VGG won in the paper; whether that holds after proper training; clinical workflow.  
7. Conclusion and future work — multi-view X-rays, CT/MRI, prospective hospital trial.

---

## 14. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| 95% not reproducible | Treat paper numbers as an upper bound; report your numbers honestly |
| Label noise on Kaggle | Spot-check 10% with a clinician if possible; document disagreements |
| GPU limits | Use EfficientNetB0 / MobileNet / Colab T4; freeze early layers |
| No bounding boxes on Kaggle set | Use FracAtlas only for detection chapter |
| Class confusion (oblique vs spiral) | Show confusion matrix and clinical explanation |

---

## 15. What to say in a viva (short answers)

- **Base paper contribution:** compared four hybrid DL pipelines on 10 fracture types; VGG-16 ± RF reached ~95% on a 1,129-image Kaggle set.  
- **Our contribution:** stricter protocol, imbalance handling, modern backbones, detection/localization, explainability, and external testing.  
- **Why not copy 95%:** small test set, possible leakage, no normals, no localization.  
- **Why EfficientNet failed there:** likely frozen features + boosting on unscaled maps, not a fundamental limit.  
- **Clinical caveat:** research prototype, not a medical device; needs prospective validation and regulatory path (SaMD).

---

## 16. References to keep besides the base paper

Use these as extra literature (standard, on-topic):

- Simonyan & Zisserman, VGG (arXiv:1409.1556)  
- He et al., ResNet (CVPR 2016)  
- Tan & Le, EfficientNet (ICML 2019)  
- Rajpurkar et al., MURA dataset  
- Papers on FracAtlas / GRAZPEDWRI-DX  
- Selvaraju et al., Grad-CAM (ICCV 2017)  
- Beyaz et al., femoral neck fracture DL (cited in the base paper)  

---

## 17. Deliverables checklist for the new project

- [ ] Reproduced baseline metrics table (4 paper models)  
- [ ] Improved training metrics table  
- [ ] Confusion matrices and ROC/PR plots  
- [ ] Grad-CAM gallery of successes **and** failures  
- [ ] Optional detection screenshots  
- [ ] Optional Streamlit demo  
- [ ] Honest limitations section  

This document is the extra detail layer on top of the IEEE Access paper: full extraction of their method, a critical gap analysis, and a complete proposed system that is suitable as an academic project upgrade rather than a copy of the baseline.
