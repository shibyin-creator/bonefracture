# AI-Powered Cardiac Risk Assessment and Heart Failure Prediction

Production-ready research software for **heart-failure survival classification** and **multi-modal cardiac risk screening**. The implementation extends the IEEE SPIN 2024 baseline paper:

> P. K. Sandilya, D. Pal, S. R. Dahiya, and S. K. Dana, “Prognostic Modeling for Heart Failure Survival: A Classification Approach,” *Proc. 11th Int. Conf. Signal Processing and Integrated Networks (SPIN)*, 2024, doi: 10.1109/SPIN60856.2024.10512260.

and the submitted project abstract *Heart Failure Prediction using Machine Learning* (Stacking Classifier + XGBoost, web deployment).

## What this repository adds beyond the base paper

| SPIN 2024 baseline | This project |
| --- | --- |
| 299 UCI Heart Failure Clinical Records | Programmatic synthetic cohort **> 20,000** patients (Gaussian copula + SMOTE) |
| 12 clinical attributes + `DEATH_EVENT` | Core 12 attributes **plus ECG interval features** (PR, QRS, QT/QTc, ST, RR, P, T) |
| NB, KNN, DT, SVM, LR, RF, XGBoost | Same suite **plus Stacking Classifier** (meta: Logistic Regression; bases: XGBoost, RF, SVM) |
| GridSearchCV on accuracy | Multi-metric GridSearchCV (**accuracy, F1, MCC, ROC-AUC**) |
| Streamlit UI | Medical-themed dashboard, ECG visualizer, **secure admin portal** (`/admin`) |

Feature-importance analysis is designed so **ejection fraction, serum creatinine, follow-up time, and age** remain dominant, consistent with SPIN 2024 Fig. 2 and Chicco & Jurman (2020).

## Repository layout

```
config.py              # Paths, grids, IEEE schema, UI theme
auth.py                # AdminAuthManager (SQLite + PBKDF2-HMAC)
data_loader.py         # DatasetHandler (ingest, validate, synthesize)
preprocessing.py       # DataPreprocessor (impute, StandardScaler, SMOTE, ECG)
models.py              # AdvancedHeartModels (baselines + stacking)
tuning.py              # HyperparameterTuner (GridSearchCV)
pipeline.py            # End-to-end training orchestration + ECG waveform
ui.py / app.py         # Streamlit clinical dashboard
pages/admin.py         # Dedicated /admin view
main_notebook.ipynb    # Colab / Jupyter end-to-end pipeline
tests/test_modules.py  # Smoke tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train models

```bash
python - <<'PY'
from pipeline import CardiacAssessmentPipeline
CardiacAssessmentPipeline(fast_tuning=True).run()
PY
```

The first run writes `data/heart_failure_multimodal_20k.csv` (>20,000 rows) and `artifacts/cardiac_risk_bundle.joblib`.

### Web application

```bash
streamlit run app.py
```

- Clinical workstation: sidebar **Clinical assessment**
- Admin portal: sidebar **Admin portal**, or Streamlit page **admin** (`/admin`)
- Query alias: `?view=admin`

Default administrator credentials (change after first login):

- Username: `admin`
- Password: `CardiacAdmin@2024`

Admin capabilities: upload datasets, generate the synthetic cohort, inspect execution logs, retrain models, and review IEEE-style metrics.

### Notebook (Jupyter / Colab)

Open `main_notebook.ipynb`. In Colab, upload this repository (or clone it) and run all cells. The first code cell installs dependencies.

## Evaluation metrics

Hold-out metrics match and extend SPIN 2024 Tables II–IV:

Accuracy, Precision, Recall, F1-score, MCC, PR-AUC, ROC-AUC, plus confusion matrices.

## Academic disclaimer

This software is a **university research prototype**. It is not a medical device, does not provide diagnosis, and must not be used as a substitute for licensed clinical judgment. Synthetic ECG traces are pedagogical visualizations derived from interval parameters, not recorded electrograms.

## Presentation documentation

All viva/demo materials are in [`docs/`](docs/README.md):

- Browser slides: open [`docs/presentation.html`](docs/presentation.html) (arrow keys)
- Speaker notes, IEEE report, user/admin manual, architecture, demo script, poster
- Bone/fracture question: **this system tests 0 bones** — see [`docs/08_BONE_FRACTURE_SCOPE.md`](docs/08_BONE_FRACTURE_SCOPE.md)

## License / course use

Prepared for academic submission. Cite the SPIN 2024 paper and the UCI Heart Failure Clinical Records dataset (Ahmad et al., 2017; Chicco & Jurman, 2020).
