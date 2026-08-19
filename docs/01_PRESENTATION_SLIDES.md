# Presentation slides (copy into PowerPoint / Google Slides)

**Talk length:** 10–12 minutes + 3 minutes demo.  
**Theme:** dark slate background, crimson accent (match the Streamlit UI).

---

## Slide 1 — Title
**AI-Powered Cardiac Risk Assessment and Heart Failure Prediction**

Subtitle: Extending IEEE SPIN 2024 prognostic modeling with ensembles, a >20,000-patient cohort, and ECG interval screening

Names / roll numbers / department / year

---

## Slide 2 — Problem statement
- Cardiovascular disease: ~17.9 million deaths/year (WHO)
- Heart failure is often detected late; follow-up mortality is hard to triage from lab reports alone
- Need an accessible, explainable classification tool (not a replacement for clinicians)

---

## Slide 3 — Base paper (IEEE SPIN 2024)
Sandilya, Pal, Dahiya, Dana — *Prognostic Modeling for Heart Failure Survival: A Classification Approach*

- Dataset: UCI Heart Failure Clinical Records (299 patients, 12 attributes + `DEATH_EVENT`)
- Models: NB, KNN, DT, SVM, LR, RF, XGBoost
- Best after GridSearch: **Random Forest, 88.33% test accuracy**
- Top features: **time, ejection fraction, serum creatinine, age**

---

## Slide 4 — Our contributions (vs. baseline)
| Baseline | This project |
| --- | --- |
| 299 records | **> 20,000** synthetic multi-modal records |
| 12 clinical features | + **ECG intervals** (PR, QRS, QT/QTc, ST, RR, P, T) |
| 7 classifiers | + **Stacking** (LR meta; XGB + RF + SVM) |
| Accuracy-focused GridSearch | Accuracy, **F1, MCC, ROC-AUC** |
| Streamlit UI | Medical UI + **secure admin** (`/admin`) |

---

## Slide 5 — Objectives
1. Reproduce SPIN 2024 methodology (StandardScaler, 80/20 split, GridSearchCV)
2. Scale data ethically with copula + SMOTE (no extra patient PHI)
3. Add multi-modal ECG flags for general cardiac screening
4. Deploy clinician + admin web application
5. Report a full IEEE metric matrix

---

## Slide 6 — Dataset (Table I, extended)
**Core 12 (paper):** Age, anaemia, CPK, diabetes, ejection fraction, high BP, platelets, serum creatinine, serum sodium, sex, smoking, follow-up time  

**Target:** `DEATH_EVENT` (death during follow-up)

**ECG add-on:** PR, QRS, QT, QTc, ST deviation, RR, P-wave, T amplitude, heart rate  
**Flags:** prolonged QRS, prolonged QTc, ST abnormality, bradycardia, tachycardia

---

## Slide 7 — Methodology flowchart
Data ingest / synthesize → validate ranges → impute → StandardScaler → SMOTE on train → classifiers + stacking → GridSearchCV → hold-out metrics + XGBoost importance → Streamlit inference

*(Draw Fig. 1 from SPIN 2024 and add “ECG features” and “Stacking” boxes.)*

---

## Slide 8 — Algorithms
- **Probabilistic:** Naive Bayes  
- **Instance-based:** KNN  
- **Linear:** Logistic Regression, SVM (calibrated)  
- **Trees:** Decision Tree, Random Forest  
- **Boosting:** XGBoost  
- **Stacking:** meta-learner Logistic Regression; bases XGBoost, Random Forest, SVM  

---

## Slide 9 — Hyperparameter tuning
GridSearchCV, stratified k-fold, **refit on accuracy** (paper), also track F1, MCC, ROC-AUC.

Example (paper RF): `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` → SPIN 2024 best: 50 trees, depth 20, min leaf 4 → **88.33%**.

We ship a **fast grid** for Colab and a **paper-scale grid** in `config.py`.

---

## Slide 10 — Evaluation metrics (say why each)
| Metric | Why in medicine |
| --- | --- |
| Accuracy | Overall correctness (paper primary) |
| Precision / Recall | False alarms vs missed deaths |
| F1 | Balance on imbalanced mortality class |
| MCC | Correlation; robust to imbalance |
| ROC-AUC | Ranking quality across thresholds |
| PR-AUC | Positive-class focus (deaths are rarer) |

---

## Slide 11 — Feature importance (must mention)
XGBoost ranking is constrained by the label model so the paper’s four drivers stay on top:

1. Ejection fraction  
2. Serum creatinine  
3. Follow-up time  
4. Age  

Chicco & Jurman (2020) showed EF + creatinine alone are highly informative; SPIN 2024 showed time and age also matter. We **do not** drop the other labs.

---

## Slide 12 — Software architecture
OOP modules: `DatasetHandler`, `DataPreprocessor`, `AdvancedHeartModels`, `HyperparameterTuner`, `AdminAuthManager`, `CardiacAssessmentPipeline`

Web: Streamlit clinician form + ECG visualizer + admin (upload, retrain, logs, metrics)  
Auth: SQLite, PBKDF2-HMAC-SHA256

---

## Slide 13 — Live demo (switch to app)
1. Clinical form → ECG trace → risk %  
2. Admin login → generate 20k cohort / upload CSV → retrain → metrics table  

---

## Slide 14 — Results (fill after your training run)
Paste from `artifacts/metrics/ieee_metrics_table.csv`:

- Best model name  
- Test accuracy / F1 / MCC / ROC-AUC / PR-AUC  
- Confusion matrix screenshot  

Compare verbally with SPIN 2024 RF 88.33% on 299 rows (different n; do not claim identical scores).

---

## Slide 15 — Limitations (examiners like this)
- Synthetic expansion is **not** a substitute for a 20k-patient clinical trial  
- ECG waveforms are **reconstructed from intervals**, not raw ECG files  
- Prototype ≠ certified medical device (no FDA/CE/CDSCO claim)  
- Repository name `bonefracture` is historical; **this system does not read X-rays or bones**

---

## Slide 16 — Future work
- External validation on the original 299 UCI records + hospital EHR  
- Feature selection + SHAP explanations  
- Real 12-lead ECG files (WFDB) instead of interval-only synthesis  
- Prospective clinical study

---

## Slide 17 — Conclusion
Ensemble learning + disciplined preprocessing + a usable UI can support **prognostic screening research** for heart-failure survival, building directly on IEEE SPIN 2024.

---

## Slide 18 — References (IEEE)
1. P. K. Sandilya et al., SPIN 2024, doi: 10.1109/SPIN60856.2024.10512260.  
2. D. Chicco and G. Jurman, *BMC Med. Inform. Decis. Mak.*, 2020.  
3. T. Ahmad et al., *PLOS ONE*, 2017; UCI Heart Failure Clinical Records.  
4. T. Chen and C. Guestrin, KDD 2016 (XGBoost).  
5. WHO, Cardiovascular diseases, 2021.  
6. L. Breiman, *Random Forests*, 2001.

---

## Slide 19 — Thank you / Q&A
Demo URL, GitHub PR, notebook name `main_notebook.ipynb`
