# Datasets — how to get, show, and cross-check

Images are **not** stored in git (the 20k wrist set alone is ~15 GB). This folder keeps the **catalog**, a **20,327-row GRAZPEDWRI-DX manifest**, and an auditor that proves each public source is still online.

## Honest scale

| Source | Role | Expected images | Status to check |
| --- | --- | --- | --- |
| [Bone Break Classification](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset) | IEEE 10 morphology classes | **1,129** (989 train / 140 test) | Kaggle live |
| [Bone Fracture Detection YOLO](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) | Abstract YOLOv8 boxes | **3,316 train / 399 val** (~3.7k–4.1k by version) | Kaggle live |
| [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) | Extra multi-region X-rays + YOLO/COCO | **4,083** | Figshare live (~323 MB zip) |
| [GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193) | 20k-scale wrist trauma | **20,327** | Figshare + Kaggle; CSV manifest in `manifests/` |

**20,000+ does not mean 20k images per fracture type.** The paper set is 1,129. Twenty-thousand scale is GRAZPEDWRI-DX (20,327 pediatric wrist radiographs) plus the other public sets (~28k combined). Morphology labels exist only on the 10-class Kaggle set unless you relabel.

## 1. Cross-check (no GPU, no 15 GB download)

```bash
cd bone_fracture_system
python data/audit_datasets.py --probe
```

This HTTP-checks Kaggle and Figshare, counts any local images, and verifies the GRAZ CSV has **exactly 20,327 rows**. Open the report:

- JSON: `data/inventory_report.json`
- Dashboard (after `python app.py`): **Datasets** in the sidebar → `/datasets`

## 2. Download images

Create a Kaggle API token: [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New Token** → save as `~/.kaggle/kaggle.json` (`chmod 600`).

```bash
pip install kaggle
python data/audit_datasets.py --download bone_break_classification
python data/audit_datasets.py --download bonefracture_yolo8
python data/audit_datasets.py --download fracatlas          # Figshare, no Kaggle token
# optional, ~16 GB:
python data/audit_datasets.py --download grazpedwri_dx
```

Then copy ImageFolder / YOLO files into:

```text
data/classification/train/<Class Name>/*.png
data/classification/val/<Class Name>/*.png
data/yolo/images/{train,val}
data/yolo/labels/{train,val}
```

Folder names should match `config.py` (for example `Avulsion Fracture`). The auditor maps common aliases such as `fracture dislocation` → `Fracture-Dislocation`.

## 3. Show it in Colab

Open `Fracture_Detection_Colab.ipynb` and run the **Dataset download & audit** cells. On Colab, upload `kaggle.json` when prompted, or mount Drive if the zips already live there.

## 4. Paper class counts (Table 1 style)

| Class | Images |
| --- | ---: |
| Fracture-Dislocation | 156 |
| Comminuted Fracture | 148 |
| Pathological Fracture | 134 |
| Avulsion Fracture | 123 |
| Greenstick Fracture | 122 |
| Hairline Fracture | 111 |
| Spiral Fracture | 86 |
| Oblique Fracture | 85 |
| Impacted Fracture | 84 |
| Longitudinal Fracture | 80 |
| **Total** | **1,129** |
