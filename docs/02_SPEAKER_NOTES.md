# Speaker notes (~10–12 minutes)

Do not read the slides verbatim. Use this as a script.

## Opening (45 s)
“Heart failure remains a major cause of premature death. The IEEE SPIN 2024 paper showed that classical ML, with StandardScaler and GridSearch, can predict death during follow-up from twelve routine labs. Our project keeps that scientific backbone and adds three engineering extensions: a large synthetic cohort, ECG interval features, and a stacking ensemble inside a medical web app.”

## Base paper (60 s)
Name the four authors and the conference. Stress **299 patients**, **80/20 split**, **RF 88.33% after tuning**. Mention they compared against Chicco & Jurman and Oladimeji. Say we cite the same UCI source (Ahmad et al., Faisalabad Institute of Cardiology).

## Contributions (45 s)
Point at the comparison table. Emphasize that 20,000 rows are **generated**, not secretly collected PHI. Copula preserves joint distributions; SMOTE balances the death class on the **training** split only.

## Methods (90 s)
Walk Fig. 1. Pause on StandardScaler: platelets are hundreds of thousands, creatinine is ~1 mg/dL — without scaling, KNN/SVM are dominated by platelets. SMOTE is only for train. Stacking uses LR as a meta-learner so it learns *when* to trust XGB vs RF vs SVM.

## Metrics (45 s)
“Accuracy alone is dangerous on imbalanced medical data. That is why we also report MCC and PR-AUC, as in the SPIN tables.”

## Features (40 s)
“If EF, creatinine, time, and age are not near the top of the importance plot, the model is not reproducing the literature. We designed the synthetic labels so those four remain dominant, then we *measure* the ranking.”

## Demo (2–3 min)
See `07_DEMO_SCRIPT.md`. Narrate risk as **probability of death during follow-up**, not “this patient will die.”

## Limitations (40 s)
Be the first to say: synthetic data, synthetic ECG, not a device. If asked about bones: “Zero fracture types. Wrong modality. See our scope document.”

## Close (20 s)
“The contribution is a faithful IEEE pipeline that is actually runnable: notebook, tests, and a clinician/admin UI.”
