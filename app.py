import csv
import io
import json
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Credit Scoring - ANN", layout="wide")
st.title("Credit Scoring - ANN")

html_block = """<style>
@keyframes ledBlink {
  0%, 100% { opacity: 1; transform: scale(1.18); box-shadow: 0 0 8px currentColor, 0 0 18px currentColor; }
  50% { opacity: 0.25; transform: scale(0.8); box-shadow: 0 0 2px currentColor; }
}
@keyframes authorPulse {
  0%, 100% { opacity: 1; border-color: rgba(56, 189, 248, 0.75); box-shadow: 0 0 22px rgba(56, 189, 248, 0.22); }
  50% { opacity: 0.45; border-color: rgba(148, 163, 184, 0.25); box-shadow: 0 0 4px rgba(56, 189, 248, 0.04); }
}
@keyframes seqAr {
  0%, 22% { opacity: 1; }
  25%, 100% { opacity: 0; }
}
@keyframes seqFr {
  0%, 24% { opacity: 0; }
  27%, 47% { opacity: 1; }
  50%, 100% { opacity: 0; }
}
@keyframes seqEn {
  0%, 49% { opacity: 0; }
  52%, 72% { opacity: 1; }
  75%, 100% { opacity: 0; }
}
@keyframes seqAmz {
  0%, 74% { opacity: 0; }
  77%, 97% { opacity: 1; }
  100% { opacity: 0; }
}
@keyframes alreadyGlidingRightToLeft {
  0% { transform: translateX(35%); }
  100% { transform: translateX(-100%); }
}
@keyframes accentScan {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes pulseBar {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}
.global-showcase-card {
  width: 100%; border-radius: 22px; padding: 1.5px; margin-bottom: 20px;
  background: linear-gradient(120deg, rgba(56,189,248,0.75), rgba(16,185,129,0.6), rgba(245,158,11,0.65), rgba(168,85,247,0.7));
  background-size: 250% 250%; animation: accentScan 8s ease infinite;
}
.global-showcase-inner {
  width: 100%; border-radius: 20.5px; padding: 18px 22px; color: #f8fafc;
  background: radial-gradient(circle at 88% 12%, rgba(56,189,248,0.14), transparent 40%), radial-gradient(circle at 12% 88%, rgba(16,185,129,0.12), transparent 42%), linear-gradient(145deg, #090e1a 0%, #111827 58%, #090d16 100%);
  font-family: "Plus Jakarta Sans", "Cairo", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
.live-chip { font-size: 11px; font-weight: 700; letter-spacing: 0.8px; color: #a7f3d0; background: rgba(16,185,129,0.14); border: 1px solid rgba(52,211,153,0.35); padding: 2px 9px; border-radius: 999px; margin-left: 6px; }
.greeting-box { position: relative; min-width: 155px; height: 30px; display: flex; align-items: center; justify-content: flex-end; }
.greeting-word { position: absolute; right: 0; font-weight: 800; font-size: 15.5px; padding: 2px 12px; border-radius: 999px; white-space: nowrap; opacity: 0; }
.word-ar { font-family: "Cairo", sans-serif; color: #7dd3fc; background: rgba(14,165,233,0.18); border: 1px solid rgba(125,211,252,0.3); animation: seqAr 7.2s infinite ease-in-out; }
.word-fr { color: #6ee7b7; background: rgba(16,185,129,0.18); border: 1px solid rgba(110,231,183,0.3); animation: seqFr 7.2s infinite ease-in-out; }
.word-en { color: #fde68a; background: rgba(245,158,11,0.18); border: 1px solid rgba(253,230,138,0.3); animation: seqEn 7.2s infinite ease-in-out; }
.word-amz { color: #e9d5ff; background: rgba(168,85,247,0.2); border: 1px solid rgba(216,180,254,0.35); animation: seqAmz 7.2s infinite ease-in-out; }
.info-split-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 14px; }
.french-statement-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; padding: 14px 16px; font-size: 14.5px; line-height: 1.7; color: #e2e8f0; }
.fixed-credits-box { background: rgba(15,23,42,0.86); border: 1px solid rgba(56,189,248,0.5); border-radius: 14px; padding: 14px 16px; animation: authorPulse 2.4s infinite ease-in-out; }
.realise-label { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; color: #7dd3fc; margin-bottom: 8px; }
.student-line {
  font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15.5px; font-weight: 800; color: #f8fafc; letter-spacing: 0.65px;
  line-height: 1.5; margin-bottom: 6px; padding: 5px 10px; border-radius: 8px;
  background: rgba(255, 255, 255, 0.04); border-left: 3px solid #38bdf8;
}
.student-line-secondary { border-left-color: #34d399; color: #e0f2fe; }
.supervisor-line { font-size: 13.5px; font-weight: 700; color: #fde68a; margin-top: 8px; padding-top: 7px; border-top: 1px dashed rgba(148,163,184,0.22); }
.project-slow-track-wrap { width: 100%; overflow: hidden; background: rgba(255,255,255,0.04); border: 1px solid rgba(148,163,184,0.2); border-radius: 12px; padding: 9px 0 6px 0; }
.project-slow-mover { display: inline-block; white-space: nowrap; font-size: 17px; font-weight: 800; color: #7dd3fc; animation: alreadyGlidingRightToLeft 20s linear infinite; }
.project-slow-mover:hover { animation-play-state: paused; }
.project-pill { background: linear-gradient(90deg, rgba(56,189,248,0.24), rgba(16,185,129,0.24)); border: 1px solid rgba(125,211,252,0.4); color: #ffffff; padding: 2px 13px; border-radius: 999px; margin: 0 6px; }
.scan-line { height: 2px; width: 100%; margin-top: 8px; border-radius: 999px; background: linear-gradient(90deg, transparent, #38bdf8, #34d399, #fbbf24, transparent); animation: pulseBar 3s ease-in-out infinite; }

.section-banner {
  margin: 18px 0 12px 0; padding: 12px 16px; border-radius: 14px;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border-left: 4px solid #38bdf8; color: #f8fafc;
  font-family: "Plus Jakarta Sans", "Cairo", sans-serif;
}
.section-banner-title { font-size: 18px; font-weight: 800; letter-spacing: 0.2px; color: #f8fafc; }
.section-banner-sub { font-size: 12.5px; color: #94a3b8; margin-top: 2px; }

.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px; margin-bottom: 14px; font-family: "Plus Jakarta Sans", "Cairo", sans-serif;
}
.kpi-card {
  border-radius: 16px; padding: 15px 16px; color: #f8fafc;
  border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
}
.kpi-total { background: linear-gradient(145deg, #0c192e 0%, #13294b 100%); border-top: 3px solid #38bdf8; }
.kpi-high { background: linear-gradient(145deg, #2a0f18 0%, #451423 100%); border-top: 3px solid #fb7185; }
.kpi-low { background: linear-gradient(145deg, #0b241d 0%, #113a2e 100%); border-top: 3px solid #34d399; }
.kpi-pd { background: linear-gradient(145deg, #211536 0%, #331e54 100%); border-top: 3px solid #c084fc; }
.kpi-label { font-size: 12.5px; font-weight: 700; color: #cbd5e1; margin-bottom: 6px; }
.kpi-value { font-size: 25px; font-weight: 800; line-height: 1.2; color: #ffffff; }
.kpi-meta { font-size: 12px; margin-top: 6px; color: #e2e8f0; opacity: 0.9; }
</style>
<div class="global-showcase-card"><div class="global-showcase-inner"><div class="top-header-strip"><div class="led-cluster"><span class="led-dot dot-sky"></span><span class="led-dot dot-emerald"></span><span class="led-dot dot-amber"></span><span class="led-dot dot-violet"></span><span class="live-chip">ANN • LIVE</span></div><div class="greeting-box"><span class="greeting-word word-ar">مرحباً</span><span class="greeting-word word-fr">Bonjour</span><span class="greeting-word word-en">Welcome</span><span class="greeting-word word-amz">ⴰⵣⵓⵍ / Azul</span></div></div><div class="info-split-grid"><div class="french-statement-box">Ce projet a été développé par nos soins dans le cadre du parcours <strong>Master Data Science et Actuariat</strong>, sous l'encadrement académique de <strong>Prof Benbrahime</strong>.</div><div class="fixed-credits-box"><div class="realise-label">Réalisé Par :</div><div class="student-line">ISFAOUI NOUREDDINE</div><div class="student-line student-line-secondary">LAHMAMI MARYEM</div><div class="supervisor-line">Encadré Par : Prof Benbrahime</div></div></div><div class="project-slow-track-wrap"><div class="project-slow-mover">Projet : <span class="project-pill">Credit Scoring - ANN</span> • Analyse &amp; Prédiction du Risque de Crédit • <span class="project-pill">Credit Scoring - ANN</span> • Modèle Réseau de Neurones</div><div class="scan-line"></div></div></div></div>"""

st.markdown(html_block, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    with open("threshold.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    return model, preprocessor, threshold

def _norm_key(text):
    return re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", str(text).strip().lower())

def _promote_header_if_needed(df, expected_cols=None):
    if df is None or df.empty:
        return df
    exp_norm = {_norm_key(c) for c in (expected_cols or []) if _norm_key(c)}
    first_row_vals = [str(v).strip() for v in df.iloc[0].tolist() if pd.notna(v) and str(v).strip()]
    first_norm = {_norm_key(v) for v in first_row_vals}

    unnamed_ratio = sum(1 for c in df.columns if str(c).strip().lower().startswith("unnamed:")) / max(1, len(df.columns))
    overlap_match = len(exp_norm.intersection(first_norm)) >= 1 if exp_norm else False

    if (unnamed_ratio >= 0.5 or overlap_match) and len(df) > 1:
        new_cols = [
            str(v).strip() if (pd.notna(v) and str(v).strip()) else f"col_{idx}"
            for idx, v in enumerate(df.iloc[0].tolist())
        ]
        df = df.iloc[1:].copy()
        df.columns = new_cols
        df = df.reset_index(drop=True)
    return df

def read_delimited_smart(raw_bytes):
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin1"):
        try:
            text = raw_bytes.decode(enc)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            sample = "\n".join(lines[:15])
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except Exception:
                counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
                sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
            if not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة الملف النصي أو CSV / Impossible de décoder le fichier CSV.")

def read_excel_smart(raw_bytes):
    for engine in ("openpyxl", "xlrd", "odf", None):
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة ملف Excel / Impossible de lire le fichier Excel.")

def read_pdf_tables(raw_bytes):
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("مكتبة pdfplumber غير مثبّتة لقراءة جداول PDF.") from exc

    tables_dfs = []
    with pdfplumber.
