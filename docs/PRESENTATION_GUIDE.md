# Presentation pack (viva / review)

Use this with the PPTX, the live Flask app, and `BONE_COVERAGE_ATLAS.md`.

## What to open on demo day

| Item | Path / command |
| --- | --- |
| Slide deck (14 academic slides) | `bone_fracture_system/docs/Bone_Fracture_Detection_Presentation.pptx` |
| Coverage slides (bones & how many we test) | `docs/Bone_Coverage_Presentation.pptx` |
| Atlas (print or PDF from Markdown) | `docs/BONE_COVERAGE_ATLAS.md` |
| One-page handout | `docs/ONE_PAGER.md` |
| Q&A | `docs/Q_AND_A.md` |
| Live app | `python main.py web --port 5000` |
| Dataset LIVE check | Sign in → **Datasets** |
| Bone coverage page | Sign in → **Body coverage** (`/coverage`) |
| Colab story | `notebooks/Bone_Fracture_Detection_Master.ipynb` |

Logins: **admin / Admin@123** (governance) and **clinician / Clinic@123** (upload / Grad-CAM / refixation).

---

## 10–12 minute talk track

| Min | Slide / screen | What to say |
| --- | --- | --- |
| 0–1 | Title | Name, enrollment **811241013**, project title, “decision support — not a device.” |
| 1–2 | Abstract | YOLO finds **where**; VGG-16 family names **what type**; Grad-CAM shows **why**; severe cases get a **fixation sketch**. |
| 2–3 | Problem | Missed hairline injuries, slow reporting, no explanation, no teaching overlay. |
| 3–4 | Coverage | **Not 206 bones.** **10 patterns × 11 regions ≈ 42 named bones.** Upper limb is the published YOLO set. |
| 4–5 | Architecture | Stage 1 YOLOv8 (mAP50 86%). Stage 2 VGG-16 / VGG-16+RF 95%, ResNet+SVM 93%. |
| 5–6 | Dataset | 1,129 morphology; 3,316/399 YOLO; 20,327 GRAZ wrist rows; FracAtlas 4,083. |
| 6–8 | Live demo | Upload X-ray → boxes → heatmap → if comminuted/spiral, pre/post hardware. |
| 8–9 | Admin | Users, audit, weight registry — hospital-style governance. |
| 9–11 | Limits & future | Foot/skull/ribs out of class list; need labelled femur/spine for those detectors; reader study. |
| 11–12 | Close | Recap 95% / 86% / 10 types / 11 regions. Questions. |

Insert your **guide’s name** on slide 1 of the academic PPTX.

---

## Live demo script (3 minutes)

1. Login as clinician.  
2. Open **Body coverage** — show 11 regions and “not 206 bones.”  
3. Open **Datasets** — four sources LIVE, GRAZ **20,327** match.  
4. **Clinical portal** — upload one PNG. Narrate YOLO, class, Grad-CAM.  
5. If refixation appears, open the demo: “education only, not implant planning.”  
6. Login as admin — users + audit log.

If weights are missing, say: “Demo mode until `weights/yolov8_fracture.pt` and `vgg16_softmax.h5` are trained.”

---

## Sentences that lose marks (avoid)

- “We detect all 206 bones.”  
- “We have 20,000 images of every fracture type.” (20k is mainly **wrist** GRAZ + other public sets.)  
- “This replaces the radiologist.”  
- “The plate overlay is a surgical plan.”
