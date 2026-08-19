# Bone and fracture coverage — what this project can test

**Short answer for a viva:** this system does **not** diagnose all **206** adult bones. It tests **10 morphological fracture types** on radiographs from **11 skeletal regions**, which together include about **42 named bones / bone groups** that generate most trauma X-rays (limbs, pelvis, spine).

| Figure you can quote | Value |
| --- | ---: |
| Adult human skeleton | 206 bones |
| Morphological classes the classifier names | **10** |
| Body regions in the detector taxonomy | **11** |
| Named bones / bone groups inside those regions | **≈ 42** |
| Theoretical region × morphology combinations | 11 × 10 = **110** |
| Combinations with a dedicated public benchmark split today | Upper-limb YOLO **7** sites + mixed 10-class morphology set (**1,129** images) |
| 20k-scale public radiographs in-repo (wrist) | **20,327** (GRAZPEDWRI-DX manifest) |

---

## 1. Two different questions (do not mix them)

1. **“What pattern is the break?”** — morphology (Avulsion, Spiral, …). IEEE Access 2025, 10 classes.
2. **“Where on the skeleton is it?”** — anatomy (Wrist, Humerus, Femur, …). YOLOv8 boxes + region labels.

A complete clinical sentence is: *“Comminuted fracture of the humeral shaft”* = morphology **and** site.

---

## 2. Eleven regions and the bones inside them

### A. Benchmarked in the project abstract (YOLOv8, mAP50 = 86%)

These seven labels were trained/validated on **3,316 / 399** images.

| Region | Bones you can point to on an X-ray | App labels |
| --- | --- | --- |
| Shoulder | Clavicle, scapula, proximal humerus | Shoulder / Shoulder Fracture |
| Humerus | Humeral shaft, distal humerus | Humerus / Humerus Fracture |
| Elbow | Distal humerus, radial head/neck, olecranon, coronoid | Elbow / Elbow Positive |
| Forearm | Radius shaft, ulna shaft | Forearm / Forearm Fracture |
| Wrist | Distal radius & ulna, scaphoid and other carpals | Wrist / Wrist Positive |
| Fingers | 5 metacarpals + 14 phalanges | Fingers / Fingers Positive |

### B. In the software taxonomy for whole-body scale-up

| Region | Bones | Notes for examiners |
| --- | --- | --- |
| Pelvis | Ilium, ischium, pubis, sacrum, acetabulum | Label `Pelvic Fracture`. Needs extra boxes for a strong detector. |
| Femur | Neck, intertrochanteric region, shaft, distal femur | Label `Femur`. |
| Knee | Distal femur, patella, tibial plateau, proximal fibula | Label `Knee`. |
| Ankle | Distal tibia & fibula (malleoli), talus | Label `Ankle`. |
| Spine | C-spine, T-spine, L-spine **as groups** (not 24 separate IDs) | Label `Spine`. |

### C. Not claimed as dedicated test classes

Skull/face, ribs, sternum, and most bones of the **foot** (unless you add labels and images). Individual vertebrae and individual carpals are **grouped**, not 24 + 8 separate outputs.

---

## 3. Ten fracture *patterns* (morphology)

These are **how the cortex failed**, not a bone name. The same pattern can appear in many bones.

| # | Class | Plain-language meaning | Bones where it is often seen |
| --- | --- | --- | --- |
| 1 | Avulsion | Tendon/ligament pulls a fragment off | Malleoli, phalanges, pelvis, tibial tubercle |
| 2 | Comminuted | ≥3 fragments | Femur, tibia, humerus, high-energy injuries |
| 3 | Fracture-Dislocation | Break + joint displacement | Ankle, elbow, shoulder, wrist, hip |
| 4 | Greenstick | Incomplete paediatric break | Radius/ulna, clavicle |
| 5 | Hairline | Thin crack, easy to miss | Tibia, metatarsal, scaphoid |
| 6 | Impacted | Fragments driven together | Surgical neck of humerus, femoral neck, distal radius |
| 7 | Longitudinal | Crack along the shaft | Long bones, metacarpals |
| 8 | Oblique | Diagonal shaft break | Tibia, femur, humerus, forearm |
| 9 | Pathological | Break through diseased bone | Vertebra, proximal femur, humerus |
| 10 | Spiral | Twist / helical line | Tibia, humerus, femur |

Paper image counts (Kaggle 10-class set, **1,129** radiographs): Fracture-Dislocation 156, Comminuted 148, Pathological 134, Avulsion 123, Greenstick 122, Hairline 111, Spiral 86, Oblique 85, Impacted 84, Longitudinal 80.

---

## 4. What you can actually *demo today*

| Test | What to upload | What the app should show |
| --- | --- | --- |
| Morphology demo | Any of the 10-class Kaggle X-rays | Softmax / RF class + Grad-CAM |
| Upper-limb localization | Elbow / wrist / finger / forearm / humerus / shoulder films | YOLOv8 box (after `yolov8_fracture.pt` is trained) |
| 20k-scale story | GRAZPEDWRI-DX wrist list (CSV already in repo = **20,327** rows) | `/datasets` page: LIVE + count match |
| Surgical story | Comminuted / spiral / dislocation | Pre-op vs post-op plate/rod overlay |

Without weight files the UI still runs in **demo mode** (saliency + heuristic). Say that openly.

---

## 5. Suggested spoken line

> “The adult skeleton has 206 bones. This project is built to screen **the trauma series that dominate emergency radiology**: shoulder through fingers, plus pelvis, femur, knee, ankle and spine — **10 break patterns** on those films. We do not output a separate class for every rib or every carpal bone. Published numbers we cite are **86% mAP50** on seven upper-limb detection classes and **95%** on the ten-class VGG-16 morphology models.”

Machine-readable copy of this atlas: [`docs/bone_coverage.json`](bone_coverage.json).
