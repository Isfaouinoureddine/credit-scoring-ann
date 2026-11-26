import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Credit Scoring - ANN", layout="wide")
st.title("Credit Scoring - ANN")

html_block = """<style>
@keyframes ledBlink {
  0%, 100% { opacity: 1; transform: scale(1.15); box-shadow: 0 0 8px currentColor, 0 0 16px currentColor; }
  50% { opacity: 0.25; transform: scale(0.8); box-shadow: 0 0 2px currentColor; }
}
@keyframes authorPulse {
  0%, 100% { opacity: 1; border-color: rgba(56, 189, 248, 0.65); box-shadow: 0 0 18px rgba(56, 189, 248, 0.18); }
  50% { opacity: 0.45; border-color: rgba(148, 163, 184, 0.25); box-shadow: 0 0 4px rgba(56, 189, 248, 0.04); }
}
@keyframes seqAr {
  0%, 30% { opacity: 1; }
  34%, 100% { opacity: 0; }
}
@keyframes seqFr {
  0%, 33% { opacity: 0; }
  37%, 63% { opacity: 1; }
  67%, 100% { opacity: 0; }
}
@keyframes seqEn {
  0%, 66% { opacity: 0; }
  70%, 96% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes slowProjectOrbit {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
.global-showcase-card {
  width: 100%; border-radius: 20px; padding: 1.5px; margin-bottom: 20px;
  background: linear-gradient(120deg, rgba(56,189,248,0.65), rgba(16,185,129,0.55), rgba(245,158,11,0.55), rgba(168,85,247,0.6));
}
.global-showcase-inner {
  width: 100%; border-radius: 18.5px; padding: 16px 20px; color: #f8fafc;
  background: linear-gradient(145deg, #0b1120 0%, #111827 60%, #090d16 100%);
  font-family: "Plus Jakarta Sans", "Cairo", sans-serif;
}
.top-header-strip {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding-bottom: 12px; margin-bottom: 14px; border-bottom: 1px solid rgba(148,163,184,0.16);
}
.led-cluster { display: flex; align-items: center; gap: 9px; }
.led-dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; animation: ledBlink 1.5s infinite ease-in-out; }
.dot-sky { background: #38bdf8; color: #38bdf8; animation-delay: 0s; }
.dot-emerald { background: #34d399; color: #34d399; animation-delay: 0.3s; }
.dot-amber { background: #fbbf24; color: #fbbf24; animation-delay: 0.6s; }
.dot-violet { background: #c084fc; color: #c084fc; animation-delay: 0.9s; }
.greeting-box { position: relative; min-width: 130px; height: 30px; display: flex; align-items: center; justify-content: flex-end; }
.greeting-word { position: absolute; right: 0; font-weight: 800; font-size: 16px; padding: 2px 12px; border-radius: 999px; white-space: nowrap; opacity: 0; }
.word-ar { font-family: "Cairo", sans-serif; color: #7dd3fc; background: rgba(14,165,233,0.16); animation: seqAr 5.4s infinite ease-in-out; }
.word-fr { color: #6ee7b7; background: rgba(16,185,129,0.16); animation: seqFr 5.4s infinite ease-in-out; }
.word-en { color: #fde68a; background: rgba(245,158,11,0.16); animation: seqEn 5.4s infinite ease-in-out; }
.info-split-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 14px; }
.french-statement-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; padding: 12px 15px; font-size: 14.5px; line-height: 1.65; color: #e2e8f0; }
.fixed-credits-box { background: rgba(15,23,42,0.75); border: 1px solid rgba(56,189,248,0.45); border-radius: 14px; padding: 12px 15px; animation: authorPulse 2.4s infinite ease-in-out; }
.students-names { font-size: 15px; font-weight: 800; color: #f8fafc; margin-bottom: 4px; }
.master-sub { font-size: 13px; color: #93c5fd; font-weight: 600; margin-bottom: 6px; }
.supervisor-line { font-size: 13.5px; font-weight: 700; color: #fde68a; }
.project-slow-track-wrap { width: 100%; overflow: hidden; background: rgba(255,255,255,0.04); border: 1px solid rgba(148,163,184,0.18); border-radius: 12px; padding: 8px 0; }
.project-slow-mover { display: inline-block; white-space: nowrap; font-size: 17px; font-weight: 800; color: #7dd3fc; animation: slowProjectOrbit 24s linear infinite; }
.project-slow-mover:hover { animation-play-state: paused; }
.project-pill { background: rgba(56,189,248,0.2); border: 1px solid rgba(125,211,252,0.35); color: #ffffff; padding: 2px 12px; border-radius: 999px; margin: 0 6px; }
</style>
<div class="global-showcase-card"><div class="global-showcase-inner"><div class="top-header-strip"><div class="led-cluster"><span class="led-dot dot-sky"></span><span class="led-dot dot-emerald"></span><span class="led-dot dot-amber"></span><span class="led-dot dot-violet"></span></div><div class="greeting-box"><span class="greeting-word word-ar">مرحباً</span><span class="greeting-word word-fr">Bonjour</span><span class="greeting-word word-en">Welcome</span></div></div><div class="info-split-grid"><div class="french-statement-box">Ce projet a été développé par nos soins dans le cadre du parcours <strong>Master Data Science et Actuariat</strong>, sous l'encadrement académique de <strong>Prof : Benbrahime</strong>.</div><div class="fixed-credits-box"><div class="students-names">ISFAOUI NOUREDDINE • LAHMAMI</div><div class="master-sub">Étudiants — Master Data Science et Actuariat</div><div class="supervisor-line">Encadré Par Prof : Benbrahime</div></div></div><div class="project-slow-track-wrap"><div class="project-slow-mover">Projet en cours : <span class="project-pill">Credit Scoring - ANN</span> • Modélisation Prédictive du Risque de Crédit • <span class="project-pill">Credit Scoring - ANN</span></div></div></div></div>"""

st.markdown(html_block, unsafe_allow_html=True)

model = load_model("ann_model.keras")
preprocessor = joblib.load("preprocessor.pkl")
with open("threshold.json", encoding="utf-8") as f:
    threshold = float(json.load(f)["threshold"])

uploaded_file = st.file_uploader("ارفع ملف للتجربة (CSV أو Excel)", type=["csv", "xlsx"])

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
    probs = model.predict(X_proc, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)
    df["probability"] = probs
    df["prediction"] = np.where(preds == 1, 1, 0)
    st.dataframe(df)
