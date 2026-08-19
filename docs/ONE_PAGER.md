# One-page project handout

**Medical Image Analysis for Bone Fracture Detection Using Deep Learning**  
SHIBIN ANTONY · Enrollment No. 811241013

## What it is
A Flask clinical workstation: **YOLOv8** localizes a fracture, **VGG-16 / VGG-16+Random Forest** (IEEE Access 2025) name the **10 break patterns**, **Grad-CAM** highlights the region, and severe patterns get an **educational** plate/rod overlay.

## Bones we can test
Not all **206** bones. **11 regions** (shoulder, humerus, elbow, forearm, wrist, fingers, pelvis, femur, knee, ankle, spine) covering **~42 named bones/groups**. Published detector classes are **7 upper-limb sites**. Morphology: **10 types** (avulsion → spiral).

## Numbers to remember
| Item | Figure |
| --- | --- |
| VGG-16 / VGG-16+RF accuracy (paper) | **95%** |
| ResNet-50+SVM | **93%** |
| YOLOv8 mAP50 (abstract) | **86%** |
| Morphology images (paper set) | **1,129** |
| YOLO train / val | **3,316 / 399** |
| GRAZPEDWRI-DX wrist studies (CSV in repo) | **20,327** |

## How to run
`pip install -r requirements.txt` then `python main.py web --port 5000`  
Admin `admin` / `Admin@123` · Clinician `clinician` / `Clinic@123`

## Safety
Research/teaching software. Not FDA/CE cleared. Every output needs a licensed clinician.
