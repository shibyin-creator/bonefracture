# Software architecture

## 1. Layered design

```
Streamlit UI (app.py, ui.py, pages/admin.py)
        │
CardiacAssessmentPipeline (pipeline.py)
        │
┌───────────────┬─────────────────┬──────────────────┐
DatasetHandler  DataPreprocessor  AdvancedHeartModels
HyperparameterTuner               AdminAuthManager
└───────────────┴─────────────────┴──────────────────┘
        │
config.py  │  artifacts/*.joblib  │  SQLite  │  CSV / logs
```

## 2. Classes (OOP map for viva)

| Class | File | Responsibility |
| --- | --- | --- |
| `DatasetHandler` | `data_loader.py` | Load/validate/synthesize >20k rows + ECG |
| `DataPreprocessor` | `preprocessing.py` | Impute, `StandardScaler`, SMOTE |
| `AdvancedHeartModels` | `models.py` | Eight estimators + metrics + importance |
| `HyperparameterTuner` | `tuning.py` | `GridSearchCV` (acc, F1, MCC, ROC-AUC) |
| `AdminAuthManager` | `auth.py` | PBKDF2 users and auth audit trail |
| `CardiacAssessmentPipeline` | `pipeline.py` | Train/eval/save bundle; ECG waveform |

## 3. Data flow (training)

1. `load_or_generate` → DataFrame (`DEATH_EVENT` + features).  
2. Stratified `train_test_split` (test_size = 0.20).  
3. Fit imputer/scaler on **train** only.  
4. SMOTE on scaled **train** only.  
5. Optional GridSearch per model (skip stacking during search; stacking is fitted after).  
6. Evaluate train and test; persist IEEE CSV + joblib bundle (includes preprocessor).

## 4. Data flow (inference)

Patient form → ECG flags → `process_inference_matrix` → `predict_proba_best` → risk band.

## 5. Stacking topology

```
           ┌─────────┐
    X ───► │ XGBoost │ ─┐
           └─────────┘  │
           ┌─────────┐  │  probabilities
    X ───► │   RF    │ ─┼──► Logistic Regression (meta) ──► ŷ
           └─────────┘  │
           ┌─────────┐  │
    X ───► │ SVM+Cal │ ─┘
           └─────────┘
```

SVM uses `CalibratedClassifierCV` (current scikit-learn deprecates `SVC(probability=True)`).

## 6. Security notes (admin)

- Passwords: PBKDF2-HMAC-SHA256, 200,000 iterations, random 16-byte salt.  
- Timing-safe compare (`hmac.compare_digest`).  
- Failed logins recorded in SQLite `auth_events`.  
- Default password must be rotated after the university demo.

## 7. What is *not* in the architecture

No DICOM loader, no X-ray CNN, no bone/fracture ontology. Cardiac tabular + interval ECG only.
