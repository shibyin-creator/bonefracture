# One-page poster / abstract board (A4)

**AI-Powered Cardiac Risk Assessment and Heart Failure Prediction**  
*University project extending IEEE SPIN 2024 (Sandilya et al.)*

### Problem
Heart failure follow-up mortality is difficult to triage from routine reports. Prior IEEE work classified `DEATH_EVENT` on 299 UCI patients (best tuned RF **88.33%**).

### Method
StandardScaler → 80/20 split → SMOTE (train) → NB, KNN, DT, SVM, LR, RF, XGBoost, **Stacking (LR ← XGB+RF+SVM)** → GridSearchCV (accuracy, F1, MCC, ROC-AUC).

### Data
12 SPIN 2024 clinical attributes + ECG intervals (PR, QRS, QT/QTc, ST, RR, P, T) + 5 conduction flags. Gaussian copula + SMOTE → **>20,000** synthetic records. Importance target: **EF, creatinine, time, age**.

### System
Modular Python (OOP) + `main_notebook.ipynb` + Streamlit clinician UI (ECG visualizer, risk %) + `/admin` (PBKDF2 login, upload, retrain, logs, metrics).

### Result
Fill after training: best model ________  test acc ________  F1 ________  MCC ________  ROC-AUC ________

### Disclaimer
Research prototype, not a medical device. **Does not detect bone fractures (0 bones).**

### Key refs
SPIN 2024 doi:10.1109/SPIN60856.2024.10512260 · Chicco & Jurman 2020 · Ahmad et al. 2017 · Chen & Guestrin 2016
