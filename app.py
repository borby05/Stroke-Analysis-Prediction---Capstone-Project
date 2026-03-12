# ============================================================
# STROKE RISK PREDICTION APP — Streamlit
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title = "Stroke Risk Predictor",
    page_icon  = "🧠",
    layout     = "centered"
)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("stroke_model.pkl")
    return model

model = load_model()

# ── App Header ────────────────────────────────────────────────
st.title("🧠 Stroke Risk Prediction")
st.write("Enter patient details below to assess stroke risk.")
st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value = 1,
        max_value = 100,
        value     = 50,
        step      = 1
    )

    hypertension = st.selectbox(
        "Hypertension",
        options = [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        options = ["never smoked", "formerly smoked",
                   "smokes", "Unknown"]
    )

with col2:
    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value = 00.0,
        max_value = 300.0,
        value     = 100.0,
        step      = 0.1
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        options = [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

st.divider()

# ── Predict Button ────────────────────────────────────────────
if st.button("Predict Stroke Risk", use_container_width=True):

    # Build patient dataframe
    patient = pd.DataFrame({
        "age"               : [age],
        "avg_glucose_level" : [avg_glucose_level],
        "hypertension"      : [hypertension],
        "heart_disease"     : [heart_disease],
        "smoking_status"    : [smoking_status]
    })

    # Make prediction
    prediction  = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    # ── Display Results ───────────────────────────────────────
    st.subheader("Prediction Result")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            label = "Stroke Probability",
            value = f"{probability:.1f}%"
        )

    with col4:
        st.metric(
            label = "Risk Level",
            value = "HIGH RISK" if prediction == 1 else "LOW RISK"
        )

    # ── Risk Message ──────────────────────────────────────────
    if prediction == 1:
        st.error("""
            ⚠️ HIGH RISK DETECTED

            This patient shows a high probability of stroke.
            Immediate medical attention is strongly advised.
            Please consult a physician as soon as possible.
        """)
    else:
        st.success("""
            ✅ LOW RISK

            This patient shows a low probability of stroke.
            Continue regular health monitoring and
            maintain a healthy lifestyle.
        """)

    # ── Patient Summary ───────────────────────────────────────
    st.divider()
    st.subheader("Patient Summary")
    summary = pd.DataFrame({
        "Feature"  : ["Age", "Glucose Level", "Hypertension",
                      "Heart Disease", "Smoking Status",
                      "Stroke Probability", "Risk Level"],
        "Value"    : [age, avg_glucose_level,
                      "Yes" if hypertension == 1 else "No",
                      "Yes" if heart_disease == 1 else "No",
                      smoking_status,
                      f"{probability:.1f}%",
                      "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("""
    ⚠️ Disclaimer: This tool is for educational purposes only
    and should not replace professional medical diagnosis.
""")