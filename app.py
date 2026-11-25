import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Credit Scoring - ANN")
st.info("Bonjour, nous sommes Noureddine Isfaoui et Maryem Lahmami, étudiants en Master Data Science et Actuariat à la Faculté d'Économie et de Gestion de Settat (FEG Settat). Nous avons le plaisir de vous présenter notre application dédiée à l'analyse du score de crédit (Credit Scoring), développée dans le cadre de notre projet. N'hésitez pas à la tester et à nous partager vos retours !")

model = load_model("ann_model.keras")
preprocessor = joblib.load("preprocessor.pkl")
with open("threshold.json", encoding="utf-8") as f:
  threshold = float(json.load(f)["threshold"])

uploaded_file = st.file_uploader(
    "ارفع ملف للتجربة (CSV أو Excel)", type=["csv", "xlsx"]
)

if uploaded_file is not None:
  file_name = uploaded_file.name.lower()

  if file_name.endswith(".csv"):
    try:
      df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
      uploaded_file.seek(0)
      df = pd.read_csv(uploaded_file, encoding="latin1")
  elif file_name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
  else:
    st.error("صيغة الملف غير مدعومة. المرجو رفع ملف CSV أو Excel (.xlsx).")
    st.stop()

  X_proc = preprocessor.transform(df)
  probs = model.predict(X_proc, verbose=0).ravel()
  preds = (probs >= threshold).astype(int)

  df["probability"] = probs
  df["prediction"] = np.where(preds == 1, "خطر / تعثر (1)", "سداد (0)")
  st.dataframe(df)
