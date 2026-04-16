from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


# Page config
st.set_page_config(
    page_title="AI Customer Risk Analyzer",
    page_icon="🚀",
    layout="wide"
)


# 🎨 Enhanced UI Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    [data-testid="stAppViewContainer"] {
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 3rem;
        border-radius: 28px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 25px 60px rgba(102, 126, 234, 0.4);
    }
    .hero h1 {
        margin: 0;
        font-size: 3.5rem;
        letter-spacing: -1px;
        font-weight: 900;
        text-shadow: 0 4px 15px rgba(0,0,0,0.4);
        background: linear-gradient(45deg, #fff, #e0d5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        margin-top: 1rem;
        opacity: 0.95;
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0e6ff 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .stButton>button {
        height: 3.2em;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        letter-spacing: 0.5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2d1b4e;
        font-weight: 700;
    }
    .stMetric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 🎯 Hero Section
st.markdown(
    """
    <div class="hero">
        <h1>� ChurnGuard Pro</h1>
        <p>Advanced customer retention intelligence powered by deep neural networks</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load artifacts
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run model.ipynb first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH}. Run model.ipynb first."
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLUMNS_PATH}. Run model.ipynb first."
        )

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()


# 📊 Input Section
st.subheader("📥 Enter Customer Details")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("💳 Credit Score", 300, 850, 600)
        gender = st.selectbox("👤 Gender", ["Female", "Male"])
        age = st.number_input("🎂 Age", 18, 100, 40)
        tenure = st.number_input("📅 Tenure", 0, 10, 3)
        balance = st.number_input("💰 Balance", 0.0, 250000.0, 60000.0)

    with col2:
        num_products = st.number_input("📦 Products", 1, 4, 2)
        has_cr_card = st.selectbox("💳 Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("⚡ Active Member", ["Yes", "No"])
        estimated_salary = st.number_input("💵 Salary", 0.0, 500000.0, 80000.0)
        geography = st.selectbox("🌍 Location", ["France", "Germany", "Spain"])

    submitted = st.form_submit_button("🚀 Analyze Risk")


# 🔮 Prediction Section
if submitted:
    raw_row = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    processed_row = raw_row.copy()
    processed_row["Gender"] = processed_row["Gender"].map({"Female": 0, "Male": 1})
    processed_row = pd.get_dummies(processed_row, columns=["Geography"], drop_first=True)
    processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

    scaled_row = scaler.transform(processed_row)
    churn_probability = float(model.predict(scaled_row, verbose=0)[0][0])
    stay_probability = 1.0 - churn_probability
    churn_label = "⚠️ HIGH RISK" if churn_probability >= 0.5 else "✅ LOW RISK"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("📊 Risk Analysis Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⚠️ Churn Risk", f"{churn_probability * 100:.2f}%")

    with col2:
        st.metric("✅ Retention Chance", f"{stay_probability * 100:.2f}%")

    st.write(f"### Final Decision: {churn_label}")

    st.progress(churn_probability)

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Model uses ANN with preprocessing: encoding + scaling")