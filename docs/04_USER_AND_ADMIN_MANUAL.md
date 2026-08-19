# User and administrator manual

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train once (required before useful predictions)

```bash
python3 -c "from pipeline import CardiacAssessmentPipeline; CardiacAssessmentPipeline(fast_tuning=True).run()"
```

Outputs:

- `data/heart_failure_multimodal_20k.csv` — >20,000 rows  
- `artifacts/cardiac_risk_bundle.joblib` — models + scaler  
- `artifacts/metrics/ieee_metrics_table.csv`  
- `logs/system_execution.log`

Colab alternative: open `main_notebook.ipynb`, run all cells.

## 3. Start the web app

```bash
streamlit run app.py
```

Browser: `http://localhost:8501`

### Clinician page
1. Enter the 12 SPIN 2024 labs/demographics.  
2. Enter ECG intervals (or leave defaults).  
3. Inspect the synthetic lead-II plot and the five ECG flags.  
4. Click **Run risk assessment**.  
5. Read the **probability of death during follow-up** (not a diagnosis).

Risk bands in the UI:

| Probability | Band |
| --- | --- |
| ≥ 65% | High |
| 35–65% | Moderate |
| < 35% | Lower |

### Admin page
- Sidebar: **Admin portal**  
- Or Streamlit multipage: **admin** → URL path `/admin`  
- Or `http://localhost:8501/?view=admin`

**Default login (change after the demo):**

| Field | Value |
| --- | --- |
| Username | `admin` |
| Password | `CardiacAdmin@2024` |

Admin tabs:

| Tab | Action |
| --- | --- |
| Dataset upload | CSV with the 12 core columns; optional `DEATH_EVENT` for retraining |
| Retrain models | Fast GridSearch recommended for live demos |
| Performance | IEEE metric table + XGBoost importance |
| Execution logs | `system_execution.log` + auth events |

## 4. CSV template (minimum columns)

```text
age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time,DEATH_EVENT
```

Aliases accepted: `anemia`, `cpk`, `death_event`, `follow_up_time`, etc. (`DatasetHandler.validate`).

## 5. What the numbers mean (for the demo audience)

- **Ejection fraction:** how much blood the left ventricle ejects; low EF → worse HF prognosis.  
- **Serum creatinine:** kidney function; high values associated with worse HF outcomes [Chicco & Jurman].  
- **Time:** days of follow-up recorded; in the UCI set, patients who die often have shorter recorded follow-up.  
- **QTc / QRS / ST:** teaching flags for arrhythmia / conduction / ischemia *patterns*, not a 12-lead interpretation.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| “No trained model bundle” | Run the pipeline command in §2 |
| Import errors | `pip install -r requirements.txt` from repo root |
| Admin login fails | Recreate `data/admin_auth.sqlite3` by deleting it and restarting (re-provisions default admin) |
| Training too slow | Keep **Fast GridSearch** checked; reduce `max_train_rows` in notebook |
