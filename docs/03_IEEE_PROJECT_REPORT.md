# IEEE-style project report

**Title:** AI-Powered Cardiac Risk Assessment and Heart Failure Prediction  
**Type:** Undergraduate / postgraduate software project report  
**Base publication:** P. K. Sandilya, D. Pal, S. R. Dahiya, and S. K. Dana, “Prognostic Modeling for Heart Failure Survival: A Classification Approach,” *Proc. 11th Int. Conf. Signal Processing and Integrated Networks (SPIN)*, 2024, pp. 415–420, doi: 10.1109/SPIN60856.2024.10512260.

## Abstract

Heart failure requires timely prognostic assessment. This work implements a classification system for follow-up mortality (`DEATH_EVENT`) using the twelve clinical attributes of the UCI Heart Failure Clinical Records dataset, as studied in IEEE SPIN 2024. The software reproduces StandardScaler preprocessing, an 80/20 stratified split, seven baseline learners, and GridSearchCV. It extends the paper with (i) programmatic expansion to more than 20,000 synthetic patients via a Gaussian copula and SMOTE, (ii) electrocardiographic interval features and conduction flags for multi-modal screening, (iii) a stacking classifier whose meta-learner is logistic regression and whose base learners are XGBoost, random forest, and SVM, and (iv) a Streamlit application with a PBKDF2-authenticated administrator portal. Evaluation reports accuracy, precision, recall, F1-score, MCC, PR-AUC, and ROC-AUC. Feature importance is checked against the paper’s ranking of ejection fraction, serum creatinine, follow-up time, and age. The system is a research prototype, not a medical device.

**Index terms:** Heart failure, machine learning, classification, stacking, XGBoost, ECG intervals, Streamlit.

## I. Introduction

WHO estimates nearly 17.9 million cardiovascular deaths annually [1]. Heart failure strains health systems and is often recognized too late [2]. SPIN 2024 argued that accessible web tools can support survival prediction from laboratory and demographic attributes [3]. The UCI corpus contains 299 adults (194 male, 105 female, ages 40–95) collected at the Faisalabad Institute of Cardiology [4], [5].

Prior work showed that ejection fraction and serum creatinine are especially informative [5]. SPIN 2024 additionally ranked follow-up time and age among the top four XGBoost features and reported 88.33% test accuracy with a tuned random forest [3].

This project keeps that scientific framing and delivers a complete, modular Python codebase and notebook suitable for university demonstration.

## II. Related work

| Study | Setting | Notable result |
| --- | --- | --- |
| Chicco & Jurman [5] | 299 UCI records | EF + creatinine; published metric suite |
| Oladimeji & Oladimeji [6] | Classification baselines | Comparative accuracy / F1 |
| Almazroi [7] | Survival prediction | DT / SVM / LR comparison |
| Sandilya et al. [3] | SPIN 2024 | RF 88.33% after GridSearch |
| Submitted abstract (Dipin V) | 5,000-row expansion, stacking + XGB | Web deployment narrative |

We implement both the SPIN 2024 model list and the abstract’s stacking / XGBoost emphasis.

## III. Dataset

### A. Core attributes (SPIN 2024 Table I)

| Attribute | Meaning | Range (paper) |
| --- | --- | --- |
| Age | Years | 40–95 |
| Anaemia | RBC deficiency | {0,1} |
| High blood pressure | Hypertension | {0,1} |
| Diabetes | Diabetic status | {0,1} |
| CPK | Creatine phosphokinase | 23–7861 mcg/L (UCI; paper table lists a lower cap) |
| Sex | Female 0 / male 1 | {0,1} |
| Ejection fraction | % blood ejected per beat | 14–80 % |
| Serum creatinine | Renal marker | 0.50–9.40 mg/dL |
| Platelets | Count | 25.01–850 kiloplatelets/mL (UCI uses raw counts) |
| Time | Follow-up days | 4–285 |
| Smoking | Smoker | {0,1} |
| Serum sodium | Electrolyte | 114–148 mEq/L |
| DEATH_EVENT | Death in follow-up | {0,1} |

### B. Synthetic expansion (>20,000)

A physiologically constrained seed (moments similar to published UCI statistics) is expanded with a Gaussian copula (CTGAN-style joint sampling without the heavy SDV stack) and SMOTE-style interpolation. Labels use a logistic risk score whose largest weights are **low EF, high creatinine, short follow-up time, and older age**, so importance plots remain literature-consistent.

### C. ECG interval layer

Columns: PR, QRS, QT, QTc, ST deviation, RR, P-wave duration, T-wave amplitude, heart rate. Binary flags: prolonged QRS (≥120 ms), prolonged QTc (≥460 ms), |ST| ≥ 1 mm, bradycardia (<60 bpm), tachycardia (>100 bpm). Waveforms in the UI are **synthesized** from these numbers for teaching, not recorded ECG.

## IV. Methodology

1. **Clean:** duplicates, type coercion, clinical range clipping (Table I).  
2. **Impute:** median (continuous), mode (binary).  
3. **StandardScaler:** zero mean, unit variance [3, Sec. IV].  
4. **Split:** 80% train / 20% test, stratified [3].  
5. **SMOTE:** train fold only.  
6. **Fit:** seven SPIN classifiers + stacking.  
7. **Tune:** GridSearchCV, multi-metric, refit accuracy.  
8. **Evaluate:** IEEE metric matrix + confusion matrix + ROC/PR.  
9. **Explain:** XGBoost feature importance.  
10. **Serve:** Streamlit clinician + admin.

## V. Algorithms (brief, for the report body)

- Logistic regression: linear log-odds classifier (paper 83.33% before tuning) [3].  
- SVM: maximum-margin; we use RBF + probability calibration.  
- Naive Bayes: class-conditional independence.  
- KNN: distance vote (scale-sensitive → scaler required).  
- Decision tree / random forest: recursive partitions; RF was the paper’s winner [3], [8].  
- XGBoost: regularized boosting [9].  
- Stacking: cross-validated base probabilities → logistic meta-learner.

## VI. Implementation

Language: Python 3. Modules: `config`, `auth`, `data_loader`, `preprocessing`, `models`, `tuning`, `pipeline`, `app`/`ui`. Persistence: `joblib` model bundle, CSV metrics, SQLite admins. Notebook: `main_notebook.ipynb` (Colab-compatible). Tests: `tests/test_modules.py`.

Default admin is documented in the user manual; passwords are PBKDF2-HMAC-SHA256 hashes, never stored in plain text.

## VII. Results

After running `CardiacAssessmentPipeline`, copy tables from:

- `artifacts/metrics/ieee_metrics_table.csv`  
- `artifacts/metrics/feature_importance.csv`

**How to discuss vs. 88.33%:** SPIN 2024 measured a 60-row test split from 299 real patients. Our hold-out is drawn from a synthetic 20k-row generator. Higher or lower accuracy is **not** a claim that we beat the IEEE paper on the same test set. Always state both *n* and *data origin*.

## VIII. Ethical and clinical disclaimer

Not a diagnostic device. No treatment advice. Synthetic ECG ≠ electrophysiology recording. Do not upload identifiable patient data to public clouds without ethics approval.

## IX. Conclusion

The project operationalizes SPIN 2024 as maintainable software, adds stacking and ECG context requested by the course abstract, and exposes metrics and retraining through a medical-themed UI.

## References

[1] World Health Organization, “Cardiovascular diseases,” 2021.  
[2] P. Ponikowski et al., “Heart failure: preventing disease and death worldwide,” *ESC Heart Fail.*, 2014.  
[3] P. K. Sandilya et al., SPIN 2024.  
[4] T. Ahmad et al., “Survival analysis of heart failure patients,” *PLOS ONE*, 2017.  
[5] D. Chicco and G. Jurman, *BMC Med. Inform. Decis. Mak.*, 2020.  
[6] O. Oladimeji and O. Oladimeji, *JITCE*, 2020.  
[7] A. Almazroi, *Math. Biosci. Eng.*, 2022.  
[8] L. Breiman, “Random forests,” *Machine Learning*, 2001.  
[9] T. Chen and C. Guestrin, KDD 2016.
