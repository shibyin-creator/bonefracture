# Improved project proposal

**Student:** Shibin Antony (811241013)  
**Base paper:** Torne et al., *IEEE Access*, 2025, DOI 10.1109/ACCESS.2025.3534818  
**Original abstract:** YOLOv8 Flask CAD, 3,715 images, 7 classes, mAP50 = 86%

---

## 1. Problem statement

Missed fractures on emergency radiographs remain common (occult hairline, pediatric greenstick, overlapping anatomy). Manual reading is slow in centres with few radiologists. Existing student systems either **classify a whole image** (base paper) or **detect on fewer than 4,000 images** (original abstract). Neither is enough for a robust CAD claim.

**Goal:** a Flask CAD that (1) finds the fracture box, (2) names the anatomical region, (3) names morphology, on a **≥ 20,000-image** training plan.

---

## 2. Gaps in the base paper (why we change it)

| Base paper | Limitation | Improvement |
|---|---|---|
| 1,129 images, 140 test | Overfits; 95% is fragile | ≥ 20,000 curated images + external test |
| Classification only | No “where” | YOLOv8 bounding boxes |
| No normal class | Assumes every film is a fracture | MURA abnormal/normal pre-filter |
| Frozen CNN + RF/SVM/XGB | EfficientNet collapsed to 41% | Fine-tuned YOLO + reproduced hybrids as baselines |
| No CLAHE / marker cleaning | Shortcut learning (“D” laterality) | CLAHE + marker suppression |
| No web tool | Not usable in a lab demo | Flask + HTML/CSS/JS |

## 3. Gaps in the original abstract

| Original abstract | Improvement |
|---|---|
| 3,316 + 399 = **3,715** images | **≥ 20,000** (target 25,000 curated) |
| Detection classes only | Detection **and** 10 morphology classes from the paper |
| mAP50 86% on a small val set | Same metric on a large stratified test + 95% CI |
| No link to IEEE paper models | `train_baselines.py` reproduces RF/SVM/XGBoost |

---

## 4. Dataset plan (≥ 20,000)

### 4.1 Public clinical sources (report / thesis numbers)

| Source | Approx. images | What we use |
|---|---|---|
| GRAZPEDWRI-DX | 20,327 | Pediatric wrist trauma (already ≥ 20k on its own) |
| MURA | 40,561 | Elbow, finger, forearm, humerus, shoulder, wrist; abnormal vs normal |
| FracAtlas | 4,083 | Expert bounding boxes |
| Roboflow 7-class YOLO | 3,715 | Original abstract classes |
| Kaggle bone-break 10-class | 1,129 | Base-paper morphology |

**Unified detection names (7):** Elbow Positive, Fingers Positive, Forearm Fracture, Humerus, Humerus Fracture, Shoulder Fracture, Wrist Positive.

**Unified morphology names (10):** Avulsion, Comminuted, Fracture-Dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral.

**Curation rule:** de-duplicate (pHash), drop unreadable films, stratify by class and source, patient-level split when IDs exist. After merge and QC the working set is **≥ 20,000** (recommended freeze: **25,000**). MURA contributes extra **normal** studies so the detector is not trained only on broken bones.

### 4.2 Bundled software corpus (this zip)

`python scripts/build_dataset.py --images 20000` writes **20,000** 256×256 three-channel grayscale phantoms with YOLO `.txt` labels and a morphology column in `dataset/metadata.csv`.

Use this to prove the code path. For the dissertation results chapter, **replace/merge** with the public sets (`scripts/merge_external.py`).

### 4.3 Split

- Train 80% / val 10% / test 10%  
- Stratified by detection class  
- No test image used in augmentation

### 4.4 Preprocessing (extra vs both prior documents)

- CLAHE (clip 2.0, 8×8 tiles)  
- Resize 256 (YOLO) / 224 (CNN baselines)  
- Suppress burned-in L/R/D markers  
- Augment: rotation ±12°, scale 0.9–1.1, mild Gaussian noise, contrast  
- **Do not** use MixUp/CutMix on hairline cracks

---

## 5. Proposed system

```
X-ray upload (Flask)
    → CLAHE preprocess
    → YOLOv8n/s detector (7 anatomical classes + box)
    → optional morphology classifier (VGG-16 / RF / SVM / XGBoost)
    → overlay + confidence
    → if max prob < τ, message: refer to radiologist
```

### Models

1. **Primary:** YOLOv8n then YOLOv8s (mAP, latency).  
2. **Baselines from paper:** VGG-16 transfer, VGG-16+RF, ResNet-50+SVM, EfficientNet-style features + XGBoost on morphology labels.  
3. **Tiny CNN:** CPU fallback shipped for demos (`scripts/train_tiny.py`).

### Training (YOLO)

- 50–100 epochs, AdamW, cosine LR, mosaic **late-off**, batch 16, image size 256–640  
- Class weights if imbalance remains  
- Early stop on val mAP50-95  

### Metrics

Detection: mAP@0.50, mAP@0.50:0.95, precision, recall, FPS.  
Classification: accuracy, macro-F1, PR-AUC, specificity, ECE.  
Report **bootstrap 95% CI**. Do not quote the paper’s 95% unless reproduced on the new split.

---

## 6. Web application (from the abstract, kept)

- Python Flask API  
- HTML/CSS/JS upload UI  
- Bounding-box overlay and class list  
- About page citing paper + improved dataset  

---

## 7. Work plan (technical, not calendar)

1. Build 20k phantom set and freeze code.  
2. Obtain public datasets; unify YAML class map.  
3. Train YOLOv8; ablate image size and augmentation.  
4. Reproduce paper hybrids on morphology subset.  
5. External test (hold out one source, e.g. FracAtlas).  
6. Flask demo + failure-case Grad-CAM/YOLO attention.  
7. Write limitations (not a medical device; phantom ≠ clinical until real data is merged).

---

## 8. Expected outcomes

- Working CAD zip with train/infer/app scripts.  
- Detection mAP50 competitive with the 86% abstract **on a much larger test set**.  
- Evidence that EfficientNet-style models need **fine-tuning**, not frozen features + XGBoost alone (lesson from the base paper).  
- Honest comparison table: Abstract 3.7k vs Paper 1.1k vs **this work ≥ 20k**.

---

## 9. Ethics

Public datasets only; no identifiable hospital dump in the zip. HIPAA/DPDP discussion in the report. Prototype for education and screening research, not diagnosis.
