import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import tensorflow as tf
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from datetime import date

st.set_page_config(page_title="Startup Success & Risk Predictor", layout="wide")
st.title("🚀 Startup Success & Risk Insight Dashboard")
st.markdown("""
Welcome to the **Startup Success Predictor**. This tool helps **entrepreneurs** assess the potential success and risk level of their startups based on key parameters such as funding, category, geography, and more. Select a dataset, choose a model, enter startup details, and receive an instant prediction.
""")

# === Paths ===
dataset_option = st.sidebar.selectbox("📂 Choose Dataset", ["Crunchbase (Category & Geography)", "SEMrush (Web Metrics)"])
if dataset_option == "Crunchbase (Category & Geography)":
    data_path = "C:/Users/aanch/Downloads/Startup/Data/updated_startup_data.csv"
    catboost_model_path = "C:/Users/aanch/Downloads/Startup/Crunch/pickel/cat.pkl"
    xgb_model_path = "C:/Users/aanch/Downloads/Startup/Crunch/pickel/xgboost.pkl"
    nn_model_path = "C:/Users/aanch/Downloads/Startup/Crunch/pickel/nn_model.h5"
else:
    data_path = "C:/Users/aanch/Downloads/Startup/Data/web_metrics_standardized_v3.csv"
    catboost_model_path = "C:/Users/aanch/Downloads/Startup/SEm/pickel/catboost_balanced_model.pkl"
    xgb_model_path = "C:/Users/aanch/Downloads/Startup/SEm/pickel/xgboost_web_model.pkl"
    nn_model_path = "C:/Users/aanch/Downloads/Startup/SEm/pickel/nn_model.h5"

# === Load dataset ===
@st.cache_data
def load_data():

    df = pd.read_csv(data_path).fillna(0)
    if "success" in df.columns:
        df = df.drop(columns=["success"])
    return df

df = load_data()

# === Sidebar ===
st.sidebar.header("📊 Input Startup Parameters")
model_option = st.sidebar.selectbox("🤖 Choose Prediction Model", ["CatBoost", "XGBoost", "Neural Network"])

# === Input form ===
sample_input = {}
st.sidebar.markdown("---")
st.sidebar.subheader("Enter your startup details:")
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        val = st.sidebar.number_input(col, value=float(df[col].mean()))
    else:
        val = st.sidebar.text_input(col, value=str(df[col].iloc[0]))
    sample_input[col] = val

input_df = pd.DataFrame([sample_input])

scaler = StandardScaler()
X_base = pd.read_csv(data_path).drop(columns='success', errors='ignore').fillna(0)

for col in X_base.columns:
    if X_base[col].dtype == 'object':
        X_base[col] = X_base[col].astype(str)

        # Skip mapping if column is missing in input_df
        if col not in input_df.columns:
            continue

        input_df[col] = input_df[col].astype(str)
        unique_vals = list(set(X_base[col].unique()) | set(input_df[col].unique()))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        X_base[col] = X_base[col].map(mapping)
        input_df[col] = input_df[col].map(mapping)

scaler.fit(X_base)
scaled_input = scaler.transform(input_df)


# === Prediction Functions ===
def predict_catboost():
    with open(catboost_model_path, "rb") as f:
        model = pickle.load(f)
    prob = model.predict_proba(input_df)[:, 1][0]
    return prob

def predict_xgboost():
    with open(xgb_model_path, "rb") as f:
        model = pickle.load(f)
    prob = model.predict_proba(input_df)[:, 1][0]
    return prob

def predict_nn():
    model = tf.keras.models.load_model(nn_model_path)
    prob = float(model.predict(scaled_input)[0][0])
    return prob

# === Predict Button ===
if st.sidebar.button("🚀 Predict Startup Success"):
    if model_option == "CatBoost":
        prob = predict_catboost()
    elif model_option == "XGBoost":
        prob = predict_xgboost()
    else:
        prob = predict_nn()

    # === Display Results ===
    st.subheader("📈 Prediction Result")
    st.metric("Predicted Success Probability", f"{prob*100:.2f}%")

    if prob >= 0.8:
        risk = "🟢 Low Risk"
    elif prob >= 0.6:
        risk = "🟡 Moderate Risk"
    elif prob >= 0.4:
        risk = "🟠 High Risk"
    else:
        risk = "🔴 Very High Risk"

    st.metric("Startup Risk Level", risk)
    st.success(f"Prediction completed using {model_option} model.")

    # === Recommendations ===
    st.markdown("---")
    st.markdown("### 🧠 Strategic Insight")
    if risk == "🔴 Very High Risk":
        st.warning("Consider refining your product-market fit or improving investor readiness.")
    elif risk == "🟠 High Risk":
        st.info("Focus on customer traction and optimizing your go-to-market strategy.")
    elif risk == "🟡 Moderate Risk":
        st.success("Build stronger brand and digital presence to enhance investor confidence.")
    else:
        st.balloons()
        st.success("Your startup is in a strong position! Focus on scaling and partnerships.")

else:
    st.info("Fill in startup details from the sidebar and click 'Predict Startup Success'.")
