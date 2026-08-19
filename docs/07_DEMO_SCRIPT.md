# Live demo script (4 minutes)

**Before the panel enters**

1. Train once so `artifacts/cardiac_risk_bundle.joblib` exists.  
2. `streamlit run app.py`  
3. Browser zoom 125%, dark UI visible.  
4. Have admin password ready.

## Minute 0:00–0:40 — Title and paper
“This is the SPIN 2024 heart-failure survival classifier, extended with a 20k synthetic cohort, ECG intervals, and stacking.” Show `docs/presentation.html` slide 3–4 if a projector is easier than the IDE.

## Minute 0:40–2:10 — Clinical screen
High-risk story (say the numbers out loud):

- Age 82, EF **22%**, creatinine **2.8**, time **12 days**, anaemia 1, high BP 1  
- QRS 140 ms, QTc high, ST −1.2 mm  

Click **Run risk assessment**. Expect a **high** probability.  
“EF and creatinine are the literature’s strongest labs; short follow-up matches how deaths were recorded in UCI.”

Low-risk contrast (optional 20 s): Age 48, EF 55%, creatinine 0.9, time 200 days → lower probability.

Point at the ECG trace: “Teaching waveform from intervals, not a hospital ECG file.”

## Minute 2:10–3:20 — Admin
Open **Admin portal** (or `/admin`).

Login `admin` / `CardiacAdmin@2024`.  
Show **Performance** metrics table (IEEE columns).  
Show feature-importance bar: EF, creatinine, time, age near the top.  
Mention logs tab: “Retrain and login events are auditable.”

Do **not** start a full 20k retrain live unless you already tested it; say “Retrain is one button; we ran it before the viva; tests are in `tests/test_modules.py`.”

## Minute 3:20–4:00 — Close
Limitations + “Questions?”  
If asked about fractures: “This application cannot test any bone. Completely different modality.”
