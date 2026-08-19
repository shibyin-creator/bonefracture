# Viva / examiner Q&A

## Project identity

**Q. What is the exact problem?**  
Binary classification of `DEATH_EVENT`: whether a heart-failure patient dies during the recorded follow-up period, given labs, demographics, and (in our extension) ECG intervals.

**Q. Is this diagnosis of heart attack?**  
No. It is *prognosis during follow-up* on a heart-failure cohort, following UCI / SPIN 2024.

**Q. Can this project test bone fractures?**  
**No. Zero bones, zero fracture types.** There is no X-ray, CT, or skeletal model. If the GitHub repo is named `bonefracture`, that name is unrelated to this software. Full anatomy list: `docs/08_BONE_FRACTURE_SCOPE.md`.

## Base paper

**Q. Why SPIN 2024?**  
It is the assigned IEEE baseline: same 12 features, StandardScaler, listed classifiers, GridSearch, Streamlit, and the four important features.

**Q. Why was RF best in the paper (88.33%)?**  
After GridSearch (`n_estimators=50`, `max_depth=20`, `min_samples_leaf=4`, `min_samples_split=2`) on a 60-patient test split. Trees capture non-linear lab interactions; tuning reduced overfit vs. 100% train accuracy before tuning.

**Q. Why not use only EF and creatinine (Chicco 2020)?**  
SPIN 2024 explicitly kept all features and showed time and age also rank highly. We keep the full vector plus ECG.

## Data

**Q. Why synthesize 20,000 rows?**  
299 rows overfit easily (see DT/XGB 100% train accuracy in the paper’s Table II). Synthesis is for **software scale and class-balance experiments**, not for claiming a new clinical evidence grade.

**Q. Is CTGAN used?**  
A Gaussian copula implements the same *idea* (sample a joint distribution) without the SDV/CTGAN dependency. SMOTE interpolates minority neighbors.

**Q. Data leakage?**  
Scaler and SMOTE are fit on train only. Test stays original (imbalanced) distribution.

**Q. Why is `time` a feature? Is that cheating?**  
It is in the UCI schema and in SPIN 2024. Clinically it is follow-up duration, which correlates with observed death. For a *prospective* admission-time tool you would drop `time`. Say this unprompted if you want extra credit.

## Models

**Q. Why stacking with LR meta?**  
Matches the submitted abstract. LR meta is linear and less prone to overfit than a second XGB on a small meta-set. Bases: XGB (non-linear), RF (bagging), SVM (margin).

**Q. Why StandardScaler?**  
SPIN 2024 Sec. IV. Required for KNN, SVM, LR; harmless for trees.

**Q. MCC vs accuracy?**  
MCC uses all four confusion-matrix cells; more honest on imbalanced death labels.

## UI / engineering

**Q. How are passwords stored?**  
PBKDF2-HMAC-SHA256, salt, 200k iterations; not bcrypt, but not plaintext.

**Q. What does the ECG plot show?**  
A toy P-QRS-T constructed from typed intervals. It is a visualizer, not a signal-processing diagnostic.

## Ethics

**Q. Can hospitals use this tomorrow?**  
No. Research prototype; no regulatory clearance; synthetic training distribution.

## Quick numbers to memorize

| Item | Value |
| --- | --- |
| UCI n | 299 |
| Paper test accuracy (RF) | 88.33% |
| Split | 80/20 |
| Our n (generated) | > 20,000 (20,500 target) |
| Core features | 12 + target |
| ECG extras | 9 intervals + 5 flags |
| Learners | 8 (7 paper + stacking) |
| Fractures supported | **0** |
