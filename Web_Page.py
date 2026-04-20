import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ---------------------------
# Feature Names
# ---------------------------
feature_names = [
    "Machine Type",
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear"
]

# ---------------------------
# Load Model & Encoder
# ---------------------------
model = joblib.load("pm_model.pkl")
le = joblib.load("type_encoder.pkl")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="centered"
)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 style='text-align: center;'>🛠️ Predictive Maintenance System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-based Machine Failure Risk Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("🔧 Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    type_input = st.radio("Machine Type", ["L", "M", "H"], horizontal=True)
    air_temp = st.slider("Air Temperature (K)", 250, 350, 300)
    process_temp = st.slider("Process Temperature (K)", 250, 350, 310)

with col2:
    rot_speed = st.slider("Rotational Speed (rpm)", 1000, 3000, 1500)
    torque = st.slider("Torque (Nm)", 10, 80, 40)
    tool_wear = st.slider("Tool Wear (min)", 0, 300, 100)

#st.markdown("---")

# ---------------------------
# Display Settings
# ---------------------------
st.subheader("⚙️ Display Settings")

decimal_places = st.slider(
    "Select number of decimal places for probability",
    min_value=1,
    max_value=6,
    value=2
)

# ---------------------------
# Encode Input
# ---------------------------
type_encoded = le.transform([type_input])[0]

X = np.array([[type_encoded,
               air_temp,
               process_temp,
               rot_speed,
               torque,
               tool_wear]])

# ---------------------------
# Prediction Section
# ---------------------------
if st.button("🚨 Predict Failure Risk"):

    prob = float(model.predict_proba(X)[0][1])
    threshold = 0.35

    formatted_prob = f"{prob:.{decimal_places}f}"
    min_display_value = 10 ** (-decimal_places)

    st.subheader("📊 Prediction Result")

    st.progress(min(prob, 1.0))

    if prob >= threshold:
        st.error(f"🔴 HIGH FAILURE RISK\n\nProbability Score: {formatted_prob}")
        st.markdown("**Recommended Action:** Schedule maintenance immediately.")
        verdict = "High Risk"

    elif prob >= 0.25:
        st.warning(f"🟠 MODERATE RISK\n\nProbability Score: {formatted_prob}")
        st.markdown("**Recommended Action:** Monitor machine closely.")
        verdict = "Moderate Risk"

    else:
        if prob < min_display_value:
            st.success(
                f"🟢 LOW FAILURE RISK\n\nProbability Score: < {min_display_value}"
            )
        else:
            st.success(
                f"🟢 LOW FAILURE RISK\n\nProbability Score: {formatted_prob}"
            )
        st.markdown("**Recommended Action:** Machine operating normally.")
        verdict = "Low Risk"

    # ---------------------------
    # Logging
    # ---------------------------
    log_data = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Machine Type": type_input,
        "Air Temp": air_temp,
        "Process Temp": process_temp,
        "Rot Speed": rot_speed,
        "Torque": torque,
        "Tool Wear": tool_wear,
        "Failure Probability": float(formatted_prob),
        "Verdict": verdict
    }

    log_df = pd.DataFrame([log_data])

    if os.path.exists("prediction_logs.csv"):
        log_df.to_csv("prediction_logs.csv", mode="a", header=False, index=False)
    else:
        log_df.to_csv("prediction_logs.csv", index=False)

#st.markdown("---")

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("📌 Feature Importance (Model Insight)")

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Contribution to Failure Prediction")

    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")

#st.markdown("---")

# ---------------------------
# Prediction History
# ---------------------------
st.subheader("📁 Prediction History")

if os.path.exists("prediction_logs.csv"):
    logs = pd.read_csv("prediction_logs.csv")
    st.dataframe(logs.tail(10))

    if st.button("🗑️ Clear History Logs"):
        os.remove("prediction_logs.csv")
        st.success("Prediction history cleared successfully.")
else:
    st.info("No prediction logs available.")
