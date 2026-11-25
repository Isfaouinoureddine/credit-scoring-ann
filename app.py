import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Credit Scoring - ANN")

model = load_model("ann_model.keras")
preprocessor = joblib.load("preprocessor.pkl")
with open("threshold.json") as f:
    threshold = float(json.load(f)["threshold"])

uploaded_file = st.file_uploader("ارفع ملف CSV للتجربة", type=["csv"])
if uploaded_file is not None:
 try:
df = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
uploaded_file.seek(0)
df = pd.read_csv(uploaded_file, encoding="latin-1")
    X_proc = preprocessor.transform(df)
    probs = model.predict(X_proc, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)
    df["probability"] = probs
    df["prediction"] = np.where(preds == 1, "خطر / تعثر (1)", "مقبول / سداد (0)")
    st.dataframe(df)
