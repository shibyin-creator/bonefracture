"""Streamlit UI views shared by app.py and pages/admin.py."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config
from auth import AdminAuthManager
from data_loader import DatasetHandler
from models import AdvancedHeartModels
from pipeline import CardiacAssessmentPipeline, synthesize_ecg_waveform
from preprocessing import DataPreprocessor

logging.basicConfig(
    filename=config.SYSTEM_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def inject_theme() -> None:
    st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_auth() -> AdminAuthManager:
    return AdminAuthManager()


def _empty_preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


@st.cache_resource
def load_runtime_bundle():
    path = config.DEFAULT_MODEL_BUNDLE
    if not Path(path).exists():
        return None, None, {}
    models, extra = AdvancedHeartModels.load_bundle(path)
    preprocessor = extra.get("preprocessor") or _empty_preprocessor()
    return models, preprocessor, extra


def risk_band(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "High risk of mortality during follow-up", config.UI_THEME["high_risk"]
    if prob >= 0.35:
        return "Moderate risk — closer clinical monitoring indicated", config.UI_THEME["warning"]
    return "Lower predicted risk relative to this cohort", config.UI_THEME["success"]


def patient_form() -> dict[str, float]:
    st.subheader("Clinical intake")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 40.0, 95.0, 65.0)
        anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x else "No")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hbp = st.selectbox("High blood pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c2:
        cpk = st.number_input("Creatinine phosphokinase (mcg/L)", 23.0, 7861.0, 250.0)
        ef = st.slider("Ejection fraction (%)", 14, 80, 38)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000.0, 850000.0, 263000.0)
        creat = st.number_input("Serum creatinine (mg/dL)", 0.5, 9.4, 1.1, step=0.1)
    with c3:
        sodium = st.number_input("Serum sodium (mEq/L)", 114.0, 148.0, 137.0)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female")
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        time = st.number_input("Follow-up time (days)", 4.0, 285.0, 90.0)

    st.subheader("ECG interval features")
    e1, e2, e3 = st.columns(3)
    with e1:
        pr = st.number_input("PR interval (ms)", 90.0, 280.0, 160.0)
        qrs = st.number_input("QRS duration (ms)", 60.0, 180.0, 95.0)
        qt = st.number_input("QT interval (ms)", 300.0, 520.0, 400.0)
    with e2:
        st_dev = st.number_input("ST deviation (mm)", -3.5, 4.0, 0.0, step=0.1)
        hr = st.number_input("Heart rate (bpm)", 40.0, 150.0, 72.0)
        p_wave = st.number_input("P-wave duration (ms)", 60.0, 140.0, 90.0)
    with e3:
        t_amp = st.number_input("T-wave amplitude (mV)", -0.4, 1.2, 0.35, step=0.05)
        rr = 60000.0 / hr
        qtc = qt / np.sqrt(rr / 1000.0)
        st.metric("Derived QTc (ms)", f"{qtc:.1f}")
        st.metric("Derived RR (ms)", f"{rr:.0f}")

    row = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": cpk,
        "diabetes": diabetes,
        "ejection_fraction": ef,
        "high_blood_pressure": hbp,
        "platelets": platelets,
        "serum_creatinine": creat,
        "serum_sodium": sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time,
        "pr_interval_ms": pr,
        "qrs_duration_ms": qrs,
        "qt_interval_ms": qt,
        "qtc_interval_ms": qtc,
        "st_deviation_mm": st_dev,
        "rr_interval_ms": rr,
        "p_wave_ms": p_wave,
        "t_wave_amplitude_mv": t_amp,
        "heart_rate_bpm": hr,
    }
    flags = DatasetHandler().compute_ecg_flags(pd.DataFrame([row])).iloc[0]
    for col in config.DERIVED_ECG_FLAGS:
        row[col] = int(flags[col])
    return row


def clinician_view() -> None:
    inject_theme()
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.title("AI-Powered Cardiac Risk Assessment")
    st.caption(
        "Heart-failure survival classification extending Sandilya et al., "
        "SPIN 2024 (IEEE), with multi-modal ECG interval screening."
    )

    models, preprocessor, extra = load_runtime_bundle()
    if models is None:
        st.warning(
            "No trained model bundle found. Open the **Admin** page to generate "
            "the >20,000-record cohort and train the ensemble, or run `main_notebook.ipynb`."
        )
    else:
        st.success(
            f"Loaded **{models.best_model_name}** trained on "
            f"{extra.get('n_records', 'n/a')} records."
        )

    payload = patient_form()
    t, wave = synthesize_ecg_waveform(
        pr_interval_ms=payload["pr_interval_ms"],
        qrs_duration_ms=payload["qrs_duration_ms"],
        qt_interval_ms=payload["qt_interval_ms"],
        st_deviation_mm=payload["st_deviation_mm"],
        heart_rate_bpm=payload["heart_rate_bpm"],
        p_wave_ms=payload["p_wave_ms"],
    )
    st.subheader("Interactive ECG visualizer (synthetic lead II)")
    chart = pd.DataFrame({"time_s": t, "mV": wave}).set_index("time_s")
    st.line_chart(chart, color=config.UI_THEME["ecg_trace"])

    flags = {k: payload[k] for k in config.DERIVED_ECG_FLAGS}
    flag_labels = {
        "ecg_prolonged_qrs": "Prolonged QRS (≥120 ms)",
        "ecg_prolonged_qt": "Prolonged QTc (≥460 ms)",
        "ecg_st_abnormality": "|ST| deviation ≥ 1 mm",
        "ecg_bradycardia": "Bradycardia (<60 bpm)",
        "ecg_tachycardia": "Tachycardia (>100 bpm)",
    }
    cols = st.columns(len(flags))
    for col, (key, label) in zip(cols, flag_labels.items()):
        col.metric(label, "Present" if flags[key] else "Absent")

    if st.button("Run risk assessment", type="primary"):
        if models is None or preprocessor is None or not getattr(preprocessor, "fitted", False):
            st.error("Train a model from the Admin portal before inference.")
            return
        frame = pd.DataFrame([payload])
        X = preprocessor.process_inference_matrix(frame)
        proba = models.predict_proba_best(X)[0, 1]
        pred = int(proba >= 0.5)
        band, color = risk_band(proba)
        st.markdown(
            f"""
            <div class="risk-card">
                <div class="metric-label">Predicted mortality probability (follow-up)</div>
                <h1 style="color:{color};margin:0.2rem 0;">{proba:.1%}</h1>
                <p>{band}</p>
                <p>Binary decision (threshold 0.5): <b>{"Death event likely" if pred else "Survival likely"}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not models.results_.empty:
            st.caption("Hold-out metrics for the active ensemble")
            st.dataframe(models.results_.head(3), use_container_width=True)


def admin_view() -> None:
    inject_theme()
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.title("Administrator portal")
    auth = get_auth()

    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False
        st.session_state.admin_user = None

    if not st.session_state.admin_ok:
        st.info("Secure login required. Default account is documented in README.md.")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if auth.authenticate(user, pw):
                st.session_state.admin_ok = True
                st.session_state.admin_user = user
                st.rerun()
            else:
                st.error("Invalid credentials.")
        return

    st.success(f"Authenticated as `{st.session_state.admin_user}`")
    if st.button("Sign out"):
        st.session_state.admin_ok = False
        st.session_state.admin_user = None
        st.rerun()

    tab_data, tab_train, tab_metrics, tab_logs = st.tabs(
        ["Dataset upload", "Retrain models", "Performance", "Execution logs"]
    )

    with tab_data:
        uploaded = st.file_uploader("Upload clinical CSV (must include the 12 core attributes)", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            try:
                cleaned = DatasetHandler().validate(raw, require_target=False)
                dest = config.DATA_DIR / "uploaded_cohort.csv"
                cleaned.to_csv(dest, index=False)
                st.session_state["uploaded_path"] = str(dest)
                st.success(f"Validated {len(cleaned):,} rows and stored at `{dest}`.")
                st.dataframe(cleaned.head(20), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
        if st.button("Generate >20,000 synthetic multi-modal records"):
            with st.spinner("Sampling Gaussian-copula + SMOTE cohort..."):
                df = DatasetHandler().load_or_generate(persist=True)
            st.success(f"Cohort ready: {len(df):,} records.")
            st.json(DatasetHandler().summarize(df))

    with tab_train:
        use_upload = st.checkbox("Train on last uploaded dataset (if present)", value=False)
        fast = st.checkbox("Fast GridSearch (recommended)", value=True)
        if st.button("Retrain full model suite", type="primary"):
            csv_path = None
            df = None
            if use_upload and st.session_state.get("uploaded_path"):
                csv_path = st.session_state["uploaded_path"]
                df = DatasetHandler().validate(pd.read_csv(csv_path), require_target=True)
            with st.spinner("Training classifiers, ensembles, and stacking meta-learner..."):
                result = CardiacAssessmentPipeline(fast_tuning=fast).run(df=df, csv_path=csv_path)
            load_runtime_bundle.clear()
            st.success(
                f"Training complete. Best model: {result.models.best_model_name}. "
                f"Records={result.n_records:,}."
            )
            st.dataframe(result.metrics, use_container_width=True)
            st.dataframe(result.importances.head(12), use_container_width=True)

    with tab_metrics:
        metrics_path = config.METRICS_DIR / "ieee_metrics_table.csv"
        fi_path = config.METRICS_DIR / "feature_importance.csv"
        if metrics_path.exists():
            st.subheader("IEEE-style evaluation matrix")
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
        if fi_path.exists():
            st.subheader("Feature importance (XGBoost)")
            fi = pd.read_csv(fi_path)
            st.bar_chart(fi.set_index("feature")["importance"])
            st.dataframe(fi, use_container_width=True)

    with tab_logs:
        if config.SYSTEM_LOG_PATH.exists():
            text = config.SYSTEM_LOG_PATH.read_text(encoding="utf-8", errors="replace")
            st.text_area("system_execution.log", value=text[-12000:], height=360)
        st.subheader("Authentication events")
        st.dataframe(pd.DataFrame(auth.list_auth_events()), use_container_width=True)
