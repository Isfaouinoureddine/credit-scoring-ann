import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Credit Scoring - ANN")
st.markdown(
    """
    <div style="padding: 14px 18px; border-radius: 10px; background: linear-gradient(135deg, #0f2942 0%, #163a5f 100%); border-right: 4px solid #38bdf8; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);">
        <p style="font-size: 16px; font-weight: 500; line-height: 1.7; margin: 0 0 8px 0; color: #f8fafc; text-align: right;" dir="rtl">
            مرحباً، نحن نورالدين اسفاوي و مريم لحمامي، طلبة ماستر علوم البيانات والاكتواريا (FEG Settat).
        </p>
        <p style="font-size: 14.5px; line-height: 1.6; margin: 0; color: #cbd5e1; text-align: left;" dir="ltr">
            Bonjour, nous sommes étudiants en Master Data Science et Actuariat à la FEG Settat. Nous présentons notre application dédiée à l'analyse du score de crédit (Credit Scoring).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
