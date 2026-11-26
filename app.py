import io
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
  0%, 100% { opacity: 1; transform: scale(1.18); box-shadow: 0 0 8px currentColor, 0 0 18px currentColor; }
  50% { opacity: 0.25; transform: scale(0.8); box-shadow: 0 0 2px currentColor; }
}
@keyframes authorPulse {
  0%, 100% { opacity: 1; border-color: rgba(56, 189, 248, 0.75); box-shadow: 0 0 22px rgba(56, 189, 248, 0.22); }
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
.live-chip { font-size: 11px; font-weight: 700; letter-spacing: 0.8px; color: #a7f3d0; background: rgba(16,185,129,0.14); border: 1px solid rgba(52,211,153,0.35); padding: 2px 9px; border-radius: 999px; margin-left: 6px; }
.greeting-box { position: relative; min-width: 130px; height: 30px; display: flex; align-items: center; justify-content: flex-end; }
.greeting-word { position: absolute; right: 0; font-weight: 800; font-size: 16px; padding: 2px 12px; border-radius: 999px; white-space: nowrap; opacity: 0; }
.word-ar { font-family: "Cairo", sans-serif; color: #7dd3fc; background: rgba(14,165,233,0.18); border: 1px solid rgba(125,211,252,0.3); animation: seqAr 5.4s infinite ease-in-out; }
.word-fr { color: #6ee7b7; background: rgba(16,185,129,0.18); border: 1px solid rgba(110,231,183,0.3); animation: seqFr 5.4s infinite ease-in-out; }
.word-en { color: #fde68a; background: rgba(245,158,11,0.18); border: 1px solid rgba(253,230,138,0.3); animation: seqEn 5.4s infinite ease-in-out; }
.info-split-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 14px; }
.french-statement-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; padding: 14px 16px; font-size: 14.5px; line-height: 1.7; color: #e2e8f0; }
.fixed-credits-box { background: rgba(15,23,42,0.82); border: 1px solid rgba(56,189,248,0.5); border-radius: 14px; padding: 14px 16px; animation: authorPulse 2.4s infinite ease-in-out; }
.realise-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; color: #7dd3fc; margin-bottom: 6px; }
.student-line { font-size: 15px; font-weight: 800; color: #f8fafc; letter-spacing: 0.4px; margin-bottom: 4px; }
.supervisor-line { font-size: 13.5px; font-weight: 700; color: #fde68a; margin-top: 6px; padding-top: 6px; border-top: 1px dashed rgba(148,163,184,0.22); }
.project-slow-track-wrap { width: 100%; overflow: hidden; background: rgba(255,255,255,0.04); border: 1px solid rgba(148,163,184,0.2); border-radius: 12px; padding: 9px 0 6px 0; }
.project-slow-mover { display: inline-block; white-space: nowrap; font-size: 17px; font-weight: 800; color: #7dd3fc; animation: alreadyGlidingRightToLeft 20s linear infinite; }
.project-slow-mover:hover { animation-play-state: paused; }
.project-pill { background: linear-gradient(90deg, rgba(56,189,248,0.24), rgba(16,185,129,0.24)); border: 1px solid rgba(125,211,252,0.4); color: #ffffff; padding: 2px 13px; border-radius: 999px; margin: 0 6px; }
.scan-line { height: 2px; width: 100%; margin-top: 8px; border-radius: 999px; background: linear-gradient(90deg, transparent, #38bdf8, #34d399, #fbbf24, transparent); animation: pulseBar 3s ease-in-out infinite; }
</style>
<div class="global-showcase-card"><div class="global-showcase-inner"><div class="top-header-strip"><div class="led-cluster"><span class="led-dot dot-sky"></span><span class="led-dot dot-emerald"></span><span class="led-dot dot-amber"></span><span class="led-dot dot-violet"></span><span class="live-chip">ANN • LIVE</span></div><div class="greeting-box"><span class="greeting-word word-ar">مرحباً</span><span class="greeting-word word-fr">Bonjour</span><span class="greeting-word word-en">Welcome</span></div></div><div class="info-split-grid"><div class="french-statement-box">Ce projet a été développé par nos soins dans le cadre du parcours <strong>Master Data Science et Actuariat</strong>, sous l'encadrement académique de <strong>Prof Benbrahime</strong>.</div><div class="fixed-credits-box"><div class="realise-label">Réalisé Par :</div><div class="student-line">ISFAOUI NOUREDDINE</div><div class="student-line">LAHMAMI MARYEM</div><div class="supervisor-line">Encadré Par : Prof Benbrahime</div></div></div><div class="project-slow-track-wrap"><div class="project-slow-mover">Projet : <span class="project-pill">Credit Scoring - ANN</span> • Analyse &amp; Prédiction du Risque de Crédit • <span class="project-pill">Credit Scoring - ANN</span> • Modèle Réseau de Neurones</div><div class="scan-line"></div></div></div></div>"""

st.markdown(html_block, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    with open("threshold.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    return model, preprocessor, threshold

def read_csv_smart(file_obj):
    raw_bytes = file_obj.getvalue()
    for enc in ("utf-8", "latin1"):
        try:
            text = raw_bytes.decode(enc)
            sep = ";" if text.count(";") > text.count(",") else ","
            return pd.read_csv(io.StringIO(text), sep=sep)
        except Exception:
            continue
    raise ValueError("تعذر قراءة ملف CSV / Impossible de lire le fichier CSV.")

def read_pdf_tables(file_obj):
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("مكتبة pdfplumber غير مثبّتة لقراءة جداول PDF.") from exc

    tables_dfs = []
    with pdfplumber.open(io.BytesIO(file_obj.getvalue())) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables() or []
            for table in extracted:
                if len(table) > 1:
                    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                    rows = table[1:]
                    tables_dfs.append(pd.DataFrame(rows, columns=headers))

    if not tables_dfs:
        raise ValueError("لم يتم العثور على جدول منظم داخل ملف PDF / Aucun tableau structuré trouvé dans le PDF.")
    return pd.concat(tables_dfs, ignore_index=True)

def load_uploaded_dataframe(file_obj):
    name = file_obj.name.lower()
    if name.endswith(".csv"):
        return read_csv_smart(file_obj)
    if name.endswith(".xlsx"):
        return pd.read_excel(file_obj, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        raw_text = file_obj.getvalue().decode("utf-8")
        data = json.loads(raw_text)
        return pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.read_json(io.StringIO(raw_text))
    if name.endswith((".tsv", ".txt")):
        text = file_obj.getvalue().decode("utf-8", errors="replace")
        sep = "\t" if "\t" in text else (";" if text.count(";") > text.count(",") else ",")
        return pd.read_csv(io.StringIO(text), sep=sep)
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(file_obj.getvalue()))
    if name.endswith(".pdf"):
        return read_pdf_tables(file_obj)
    raise ValueError("صيغة الملف غير مدعومة / Format de fichier non supporté.")

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات النموذج / Erreur de chargement : {e}")
    st.stop()

st.caption(f"العتبة المعتمدة / Seuil de décision (threshold) : **{threshold:.4f}**")

uploaded_file = st.file_uploader(
    "ارفع ملف للتجربة / Charger un fichier (CSV, XLSX, XLS, JSON, TSV, TXT, Parquet, PDF)",
    type=["csv", "xlsx", "xls", "json", "tsv", "txt", "parquet", "pdf"]
)

if uploaded_file is not None:
    try:
        df = load_uploaded_dataframe(uploaded_file)
    except Exception as e:
        st.error(f"تعذر معالجة الملف المرفوع / Erreur de lecture : {e}")
        st.stop()

    if df.empty:
        st.warning("الملف فارغ / Le fichier est vide.")
        st.stop()

    try:
        X_proc = preprocessor.transform(df)
        probs = model.predict(X_proc, verbose=0).reshape(-1)
    except Exception as e:
        st.error(f"خطأ أثناء تجهيز البيانات أو التنبؤ / Erreur lors de la prédiction : {e}")
        st.stop()

    preds = (probs >= threshold).astype(int)

    df_out = df.copy()
    df_out["probability"] = np.round(probs, 6)
    df_out["prediction"] = preds
    df_out["النتيجة / Résultat"] = np.where(
        preds == 1,
        "تعثر محتمل / Risque élevé (Défaut)",
        "سداد محتمل / Risque faible (Bon client)"
    )

    total_rows = len(df_out)
    high_risk_count = int((preds == 1).sum())
    low_risk_count = int((preds == 0).sum())
    high_risk_pct = (high_risk_count / total_rows) * 100 if total_rows else 0.0
    low_risk_pct = (low_risk_count / total_rows) * 100 if total_rows else 0.0

    st.subheader("ملخص التوقعات / Résumé des prédictions")
    c1, c2, c3 = st.columns(3)
    c1.metric("إجمالي الملفات / Total dossiers", total_rows)
    c2.metric("مخاطر مرتفعة / Risque élevé", f"{high_risk_count} ({high_risk_pct:.1f}%)")
    c3.metric("مخاطر منخفضة / Risque faible", f"{low_risk_count} ({low_risk_pct:.1f}%)")

    filter_choice = st.radio(
        "تصفية النتائج / Filtrer les résultats :",
        options=[
            "الكل / Tous",
            "مخاطر مرتفعة فقط / Risque élevé",
            "مخاطر منخفضة فقط / Risque faible"
        ],
        horizontal=True
    )

    if filter_choice == "مخاطر مرتفعة فقط / Risque élevé":
        filtered_df = df_out[df_out["prediction"] == 1]
    elif filter_choice == "مخاطر منخفضة فقط / Risque faible":
        filtered_df = df_out[df_out["prediction"] == 0]
    else:
        filtered_df = df_out

    st.subheader("جدول النتائج / Tableau des résultats")
    st.dataframe(filtered_df, use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="تحميل النتائج بصيغة CSV / Télécharger les résultats (CSV)",
        data=csv_bytes,
        file_name="credit_scoring_predictions.csv",
        mime="text/csv"
    )
