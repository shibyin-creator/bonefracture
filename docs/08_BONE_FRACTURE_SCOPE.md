# Bone and fracture scope (read this before the viva)

## Direct answer

**This project cannot test any bone fracture.**

| Question | Answer |
| --- | --- |
| How many bones can we screen? | **0** |
| How many fracture types can we classify? | **0** |
| What imaging is supported? | **None** (no X-ray, CT, MRI, DEXA) |
| What does the software actually take? | Blood/clinical table + typed ECG intervals |

The implemented title is **AI-Powered Cardiac Risk Assessment and Heart Failure Prediction** (IEEE SPIN 2024 heart-failure *survival* classification). There is no skeletal detector in `models.py`, no DICOM pipeline, and no fracture labels in the dataset.

If the GitHub repository is named `bonefracture`, treat that as an **unrelated repository name**. Do not tell the panel that this app “detects fractures.”

---

## Why the question still comes up

Examiners may mix **cardiology** (this codebase) with **orthopedic imaging** (a different ML problem: bounding boxes / segmentation / classification on radiographs). Those problems do not share features, models, or ground truth with `DEATH_EVENT`.

---

## Human skeleton facts (general anatomy — *not implemented*)

Use this only if asked “how many bones are in the body?” as general knowledge.

Adult humans typically have **206 bones**. Infants have more (~270); some bones fuse with age.

### Axial skeleton — 80 bones

| Region | Bones (count) | Examples |
| --- | --- | --- |
| Skull (cranium + face) | 22 | Frontal, parietals, occipital, mandible, maxillae, zygomatic, … |
| Auditory ossicles | 6 | Malleus, incus, stapes (×2) |
| Hyoid | 1 | Hyoid |
| Vertebral column | 26 | 7 cervical, 12 thoracic, 5 lumbar, sacrum, coccyx |
| Thoracic cage | 25 | Sternum (1) + 24 ribs (12 pairs) |

### Appendicular skeleton — 126 bones

| Region | Count | Major bones |
| --- | --- | --- |
| Pectoral girdle | 4 | Clavicle, scapula (×2) |
| Upper limbs | 60 | Humerus, radius, ulna, 8 carpals, 5 metacarpals, 14 phalanges (×2) |
| Pelvic girdle | 2 | Hip bones (ilium/ischium/pubis fused) |
| Lower limbs | 60 | Femur, patella, tibia, fibula, 7 tarsals, 5 metatarsals, 14 phalanges (×2) |

---

## If this *were* a fracture-detection product (it is not)

A radiology AI would still **not** “test 206 bones independently” in one click. Clinical X-ray workflows image **regions**. Typical research datasets cover on the order of **7–20 study types**, not 206 classes.

### Common radiographic regions in fracture ML (reference only)

| # | Body region | Bones usually visible | Typical views |
| --- | --- | --- | --- |
| 1 | Skull / face | Cranial vault, facial bones | AP, lateral |
| 2 | Cervical spine | C1–C7 | Lateral, AP, odontoid |
| 3 | Thoracic spine | T1–T12 | AP, lateral |
| 4 | Lumbar spine | L1–L5 | AP, lateral |
| 5 | Ribs / chest | Ribs 1–12, sternum | Chest PA/AP, rib series |
| 6 | Clavicle | Clavicle | AP |
| 7 | Scapula / shoulder | Scapula, proximal humerus, glenoid | AP, lateral/Y, axillary |
| 8 | Humerus | Humerus | AP, lateral |
| 9 | Elbow | Distal humerus, proximal radius/ulna | AP, lateral |
| 10 | Forearm | Radius, ulna | AP, lateral |
| 11 | Wrist | Distal radius/ulna, carpals | PA, lateral, oblique |
| 12 | Hand | Metacarpals, phalanges | PA, oblique, lateral |
| 13 | Fingers / thumb | Phalanges | Dedicated digit views |
| 14 | Pelvis / hip | Innominate bone, proximal femur | AP pelvis, hip lateral |
| 15 | Femur | Femoral shaft | AP, lateral |
| 16 | Knee | Distal femur, proximal tibia/fibula, patella | AP, lateral, sunrise |
| 17 | Tibia / fibula | Leg | AP, lateral |
| 18 | Ankle | Distal tibia/fibula, talus | AP, mortise, lateral |
| 19 | Foot | Tarsals, metatarsals, phalanges | AP, oblique, lateral |
| 20 | Toes | Phalanges | Dedicated views |

**Count you could quote for a *hypothetical* orthopedic AI:** about **20 body regions** (table above), or **7 upper-extremity study types** if you copied the public **MURA** dataset (elbow, finger, forearm, hand, humerus, shoulder, wrist). Neither is available in *this* repository.

### Fracture *patterns* (orthopedics — not in our labels)

These are clinical descriptors, not software outputs here:

- Closed vs open (compound)  
- Simple vs comminuted vs segmental  
- Transverse, oblique, spiral  
- Greenstick / torus (pediatric)  
- Intra-articular vs extra-articular  
- Displaced vs undisplaced  
- Pathologic, stress / insufficiency  
- AO/OTA alphanumeric classification (site + pattern)

A serious fracture system would also need laterality (left/right), bone name, and often a bounding box. **None of that exists in this codebase.**

---

## What *this* project *can* “test” (cardiac, not bone)

| Item | Count |
| --- | --- |
| Heart-failure clinical attributes (SPIN Table I) | **12** |
| Target | **1** (`DEATH_EVENT`) |
| ECG interval features | **9** |
| ECG screening flags | **5** |
| Classifiers (including stacking) | **8** |
| Synthetic patients generated | **> 20,000** |
| Bones / fractures | **0** |

---

## One-sentence line for the panel

“There are 206 bones in the adult skeleton and dozens of fracture patterns in orthopedic radiology, but this software implements none of them; it predicts heart-failure follow-up mortality from labs and ECG intervals, following IEEE SPIN 2024.”
