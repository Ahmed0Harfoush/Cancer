import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

st.title("🔮 Breast Cancer Diagnosis - Manual Input")

# --- Load Pretrained Model ---
try:
    model = joblib.load(r"NB_model.pkl")  
    st.success("✅ Model Loaded Successfully!")
except:
    st.warning("⚠ Model not found. Please train and save the model first.")
    model = None

# --- Manual Input ---
st.subheader("📝 Enter Tumor Features")

radius_mean = st.number_input("Radius Mean", min_value=0.0, step=0.1)
texture_mean = st.number_input("Texture Mean", min_value=0.0, step=0.1)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, step=0.1)
area_mean = st.number_input("Area Mean", min_value=0.0, step=0.1)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, step=0.001)

if st.button("🔍 Predict"):
    if model is not None:
        input_data = pd.DataFrame(
            [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]],
            columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
        )

        prediction = model.predict(input_data)[0]
        result = "🔴 Malignant (Cancer)" if prediction == 1 else "🟢 Benign (Non-cancer)"
        st.success(f"✅ Prediction: {result}")
    else:
        st.error("❌ No trained model available.")
