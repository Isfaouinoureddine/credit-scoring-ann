import csv
import io
import json
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Credit Scoring - ANN | FEG Settat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

.stApp {
  background: radial-gradient(circle at 12% 8%, #101b33 0%, #090e1a 48%, #060912 100%);
  color: #f1f5f9;
  font-family: "Plus Jakarta Sans", "Cairo", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1326 0%, #070c18 100%);
  border-right: 1px solid rgba(148, 163, 184, 0.14);
}

@keyframes ledPulse {
  0%, 100% { opacity: 1; transform: scale(1.15); box-shadow: 0 0 10px currentColor; }
  50% { opacity: 0.35; transform: scale(0.85); box-shadow: 0 0 2px currentColor; }
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

@keyframes typeLine1 {
  0% { max-width: 0; border-right-color: #38bdf8; }
  28% { max-width: 100%; border-right-color: #38bdf8; }
  33%, 100% { max-width: 100%; border-right-color: transparent; }
}
@keyframes typeLine2 {
  0%, 30% { max-width: 0; opacity: 0; border-right-color: #34d399; }
  31% { max-width: 0; opacity: 1; border-right-color: #34d399; }
  62% { max-width: 100%; opacity: 1; border-right-color: #34d399; }
  67%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}
@keyframes typeLine3 {
  0%, 64% { max-width: 0; opacity: 0; border-right-color: #fbbf24; }
  65% { max-width: 0; opacity: 1; border-right-color: #fbbf24; }
  92% { max-width: 100%; opacity: 1; border-right-color: #fbbf24; }
  97%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

@keyframes projectTicker {
  0% { transform: translateX(30%); }
  100% { transform: translateX(-100%); }
}

.brand-box {
  background: rgba(30, 41, 59, 0.55);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 16px;
  padding: 14px;
  margin-bottom: 14px;
  text-align: center;
}
.brand-title {
  font-size: 19px;
  font-weight: 800;
  color: #f8fafc;
  margin-top: 6px;
}
.brand-sub {
  font-size: 12px;
  color: #94a3b8;
  margin-top: 2px;
}

.top-banner {
  background: linear-gradient(135deg, rgba(17, 28, 51, 0.92) 0%, rgba(12, 20, 38, 0.95) 100%);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 20px;
  padding: 20px 24px;
  margin-bottom: 18px;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35);
}

.banner-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 14px;
  margin-bottom: 14px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.14);
}

.led-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.led-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  display: inline-block;
  animation: ledPulse 1.6s infinite ease-in-out;
}
.d-sky { background: #38bdf8; color: #38bdf8; animation-delay: 0s; }
.d-emerald { background: #34d399; color: #34d399; animation-delay: 0.3s; }
.d-amber { background: #fbbf24; color: #fbbf24; animation-delay: 0.6s; }

.online-pill {
  font-size: 11.5px;
  font-weight: 700;
  color: #6ee7b7;
  background: rgba(16, 185, 129, 0.14);
  border: 1px solid rgba(52, 211, 153, 0.35);
  padding: 2px 10px;
  border-radius: 999px;
}

.greet-stage {
  position: relative;
  min-width: 190px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}
.greet-badge {
  position: absolute;
  right: 0;
  font-size: 21px;
  font-weight: 800;
  padding: 3px 16px;
  border-radius: 999px;
  white-space: nowrap;
  opacity: 0;
}
.gb-ar { font-family: "Cairo", sans-serif; color: #7dd3fc; background: rgba(14, 165, 233, 0.2); border: 1px solid rgba(125, 211, 252, 0.35); animation: seqAr 9.5s infinite ease-in-out; }
.gb-fr { color: #6ee7b7; background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(110, 231, 183, 0.35); animation: seqFr 9.5s infinite ease-in-out; }
.gb-en { color: #fde68a; background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(253, 230, 138, 0.35); animation: seqEn 9.5s infinite ease-in-out; }
.gb-amz { color: #e9d5ff; background: rgba(168, 85, 247, 0.22); border: 1px solid rgba(216, 180, 254, 0.35); animation: seqAmz 9.5s infinite ease-in-out; }

.academic-panel {
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(56, 189, 248, 0.22);
  border-left: 4px solid #38bdf8;
  border-radius: 14px;
  padding: 14px 18px;
  margin-bottom: 12px;
}

.acad-line {
  margin-bottom: 8px;
  line-height: 1.65;
}
.acad-line:last-child {
  margin-bottom: 0;
}

.acad-label {
  font-size: 12.5px;
  font-weight: 800;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: #38bdf8;
  margin-right: 6px;
}

.acad-val {
  font-size: 15.5px;
  font-weight: 700;
  color: #f8fafc;
}

.stream-1 { display: inline-block; overflow: hidden; white-space: nowrap; max-width: 100%; border-right: 2px solid transparent; padding-right: 4px; animation: typeLine1 9.5s steps(38, end) infinite; }
.stream-2 { display: inline-block; overflow: hidden; white-space: nowrap; max-width: 100%; border-right: 2px solid transparent; padding-right: 4px; animation: typeLine2 9.5s steps(56, end) infinite; }
.stream-3 { display: inline-block; overflow: hidden; white-space: nowrap; max-width: 100%; border-right: 2px solid transparent; padding-right: 4px; animation: typeLine3 9.5s steps(62, end) infinite; }

.marquee-box {
  width: 100%;
  overflow: hidden;
  background: rgba(30, 41, 59, 0.65);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 10px;
  padding: 7px 0;
}
.marquee-text {
  display: inline-block;
  white-space: nowrap;
  font-size: 14.5px;
  font-weight: 800;
  color: #e2e8f0;
  animation: projectTicker 20s linear infinite;
}
.marquee-chip {
  background: rgba(56, 189, 248, 0.2);
  border: 1px solid rgba(125, 211, 252, 0.4);
  color: #ffffff;
  padding: 1px 11px;
  border-radius: 999px;
  margin: 0 6px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.kpi-card {
  background: rgba(17, 27, 48, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 15px 17px;
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25);
}
.k-blue { border-top: 3px solid #38bdf8; }
.k-green { border-top: 3px solid #34d399; }
.k-amber { border-top: 3px solid #fbbf24; }
.k-purple { border-top: 3px solid #c084fc; }

.k-title { font-size: 12.5px; font-weight: 700; color: #94a3b8; margin-bottom: 6px; }
.k-num { font-size: 25px; font-weight: 800; color: #ffffff; line-height: 1.2; }
.k-sub { font-size: 12px; color: #cbd5e1; margin-top: 6px; }

.glass-card {
  background: rgba(17, 27, 48, 0.85);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.card-heading {
  font-size: 16.5px;
  font-weight: 800;
  color: #f8fafc;
  margin-bottom: 4px;
}
.card-desc {
  font-size: 12.5px;
  color: #94a3b8;
  margin-bottom: 12px;
}
</style>"""

TOP_BANNER_HTML = """
<div class="top-banner">
  <div class="banner-head">
    <div class="led-row">
      <span class="led-dot d-sky"></span>
      <span class="led-dot d-emerald"></span>
      <span class="led-dot d-amber"></span>
      <span class="online-pill">Application en ligne • ANN</span>
    </div>
    <div class="greet-stage">
      <span class="greet-badge gb-ar">مرحباً</span>
      <span class="greet-badge gb-fr">Bonjour</span>
      <span class="greet-badge gb-en">Welcome</span>
      <span class="greet-badge gb-amz">ⴰⵣⵓⵍ / Azul</span>
    </div>
  </div>

  <div class="academic-panel">
    <div class="acad-line">
      <span class="stream-1">
        <span class="acad-label">ENCADRÉ PAR :</span>
        <span class="acad-val">Prof. Benbrahime</span>
      </span>
    </div>
    <div class="acad-line">
      <span class="stream-2">
        <span class="acad-label">RÉALISÉ PAR :</span>
        <span class="acad-val">ISFAOUI NOUREDDINE &amp; LAHMAMI MARYEM</span>
      </span>
    </div>
    <div class="acad-line">
      <span class="stream-3">
        <span class="acad-label">FILIÈRE &amp; UNIVERSITÉ :</span>
        <span class="acad-val">Master Data Science et Actuariat — FEG Settat</span>
      </span>
    </div>
  </div>

  <div class="marquee-box">
    <div class="marquee-text">
      Projet : <span class="marquee-chip">Credit Scoring - ANN</span> • Prédiction du Risque de Défaut (PD) •
      <span class="marquee-chip">FEG Settat</span> • Master Data Science et Actuariat
    </div>
  </div>
</div>
"""


@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    with open("threshold.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    return model, preprocessor, threshold


def _norm_key(text: object) -> str:
    try:
        if text is None:
            return ""
        if isinstance(text, (list, tuple, dict, set)):
            text = "_".join(str(x) for x in text)
        return re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", str(text).strip().lower())
    except Exception:
        return ""


def _safe_expected_cols(preprocessor):
    try:
        raw_cols = getattr(preprocessor, "feature_names_in_", None)
        if raw_cols is None:
            return None
        cols = [c for c in raw_cols if c is not None and not isinstance(c, (dict, set))]
        return cols if cols else None
    except Exception:
        return None


def _promote_header_if_needed(df: pd.DataFrame, expected_cols=None) -> pd.DataFrame:
    try:
        if df is None or df.empty or len(df) <= 1:
            return df
        exp_norm = {_norm_key(c) for c in (expected_cols or []) if _norm_key(c)}
        first_row = df.iloc[0].tolist()
        first_vals = [str(v).strip() for v in first_row if pd.notna(v) and str(v).strip()]
        first_norm = {_norm_key(v) for v in first_vals if _norm_key(v)}

        col_names = [str(c).strip() for c in df.columns]
        unnamed_ratio = sum(1 for c in col_names if c.lower().startswith("unnamed:")) / max(1, len(col_names))
        overlap_match = bool(exp_norm and first_norm and (exp_norm & first_norm))

        if unnamed_ratio >= 0.5 or overlap_match:
            new_cols = [
                str(v).strip() if (pd.notna(v) and str(v).strip()) else f"col_{idx}"
                for idx, v in enumerate(first_row)
            ]
            promoted = df.iloc[1:].copy()
            promoted.columns = new_cols
            return promoted.reset_index(drop=True)
    except Exception:
        pass
    return df


def read_delimited_smart(raw_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin1"):
        try:
            text = raw_bytes.decode(enc)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            sample = "\n".join(lines[:15])
            try:
                sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except Exception:
                counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
                sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة الملف النصي أو CSV.")


def read_excel_smart(raw_bytes: bytes) -> pd.DataFrame:
    for engine in ("openpyxl", "xlrd", "odf", None):
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة ملف Excel.")


def read_pdf_tables(raw_bytes: bytes) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("مكتبة pdfplumber غير مثبّتة لقراءة جداول PDF.") from exc

    tables_dfs = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                if len(table) > 1:
                    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                    tables_dfs.append(pd.DataFrame(table[1:], columns=headers))
    if not tables_dfs:
        raise ValueError("لم يتم العثور على جدول منظم داخل ملف PDF.")
    return pd.concat(tables_dfs, ignore_index=True)


def load_any_tabular_file(file_obj) -> pd.DataFrame:
    raw_bytes = file_obj.getvalue()
    name = (file_obj.name or "").lower()

    if name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb", ".ods")):
        loaders = [read_excel_smart, read_delimited_smart]
    elif name.endswith((".csv", ".tsv", ".txt", ".data")):
        loaders = [read_delimited_smart, read_excel_smart]
    elif name.endswith(".json"):
        def _json_loader(b):
            raw_text = b.decode("utf-8-sig", errors="replace")
            data = json.loads(raw_text)
            return pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.read_json(io.StringIO(raw_text))
        loaders = [_json_loader, read_delimited_smart]
    elif name.endswith(".parquet"):
        loaders = [lambda b: pd.read_parquet(io.BytesIO(b)), read_delimited_smart]
    elif name.endswith(".pdf"):
        loaders = [read_pdf_tables]
    else:
        loaders = [read_excel_smart, read_delimited_smart, read_pdf_tables]

    last_err = None
    for loader in loaders:
        try:
            df = loader(raw_bytes)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_err = exc
            continue
    raise ValueError(f"تعذر قراءة الملف ({last_err}).")


def align_dataframe_to_preprocessor(df_raw: pd.DataFrame, preprocessor):
    try:
        expected_cols = _safe_expected_cols(preprocessor)
        df = _promote_header_if_needed(df_raw.copy(), expected_cols=expected_cols)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                cleaned = df[col].astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
                numeric_try = pd.to_numeric(cleaned, errors="coerce")
                if len(df) > 0 and numeric_try.notna().sum() >= max(1, int(0.7 * len(df))):
                    df[col] = numeric_try
            except Exception:
                continue

        if not expected_cols:
            return df, [], list(df.columns)

        exact_map = {str(c).strip(): c for c in df.columns}
        norm_map = {_norm_key(c): c for c in df.columns if _norm_key(c)}

        aligned = pd.DataFrame(index=df.index)
        missing_cols = []
        matched_cols = []
        for exp in expected_cols:
            exp_str = str(exp).strip()
            exp_norm = _norm_key(exp_str)
            if exp_str in exact_map:
                aligned[exp] = df[exact_map[exp_str]]
                matched_cols.append(exp_str)
            elif exp_norm and exp_norm in norm_map:
                aligned[exp] = df[norm_map[exp_norm]]
                matched_cols.append(exp_str)
            else:
                aligned[exp] = np.nan
                missing_cols.append(exp_str)

        return aligned, missing_cols, matched_cols
    except Exception:
        fallback_df = df_raw.copy()
        fallback_df.columns = [str(c).strip() for c in fallback_df.columns]
        return fallback_df, [], list(fallback_df.columns)


def safe_predict_probabilities(model, preprocessor, df_model_input: pd.DataFrame) -> np.ndarray:
    try:
        X_proc = preprocessor.transform(df_model_input)
        return model.predict(X_proc, verbose=0).reshape(-1)
    except Exception:
        df_retry = df_model_input.copy()
        for col in df_retry.columns:
            if df_retry[col].dtype == object or str(df_retry[col].dtype).startswith("string"):
                num_try = pd.to_numeric(
                    df_retry[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                if num_try.notna().sum() > 0:
                    df_retry[col] = num_try
        X_proc = preprocessor.transform(df_retry)
        return model.predict(X_proc, verbose=0).reshape(-1)


def classify_risk_bucket(prob: float) -> str:
    if prob < 0.25:
        return "منخفض جداً / Très faible"
    if prob < 0.50:
        return "منخفض إلى متوسط / Faible-Modéré"
    if prob < 0.75:
        return "مرتفع / Élevé"
    return "حرج / Critique"


def build_pd_histogram_df(probs: np.ndarray, bins: int = 10) -> pd.DataFrame:
    counts, edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(counts))]
    return pd.DataFrame({"Intervalle PD": labels, "Nombre de dossiers": counts}).set_index("Intervalle PD")


def build_sensitivity_table(probs: np.ndarray, base_threshold: float) -> pd.DataFrame:
    candidates = sorted(set([0.30, 0.40, 0.50, round(base_threshold, 4), 0.70, 0.80]))
    rows = []
    total = max(1, len(probs))
    for th in candidates:
        high_n = int((probs >= th).sum())
        low_n = int((probs < th).sum())
        rows.append({
            "العتبة / Seuil": f"{th:.4f}",
            "مخاطر مرتفعة / Risque élevé": f"{high_n} ({(high_n/total)*100:.1f}%)",
            "مخاطر منخفضة / Risque faible": f"{low_n} ({(low_n/total)*100:.1f}%)",
        })
    return pd.DataFrame(rows)


def build_top_numeric_deviations(df_orig: pd.DataFrame, row_idx, top_n: int = 4) -> pd.DataFrame:
    num_df = df_orig.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df) < 2:
        return pd.DataFrame()
    row_vals = num_df.loc[row_idx]
    means = num_df.mean()
    stds = num_df.std().replace(0, np.nan)
    z_scores = ((row_vals - means) / stds).abs().dropna()
    if z_scores.empty:
        return pd.DataFrame()
    top_cols = z_scores.sort_values(ascending=False).head(top_n).index.tolist()
    return pd.DataFrame([
        {
            "المتغير / Variable": col,
            "قيمة الملف / Valeur": round(float(row_vals[col]), 4),
            "متوسط المحفظة / Moyenne": round(float(means[col]), 4),
            "الفارق / Écart": round(float(row_vals[col] - means[col]), 4),
        }
        for col in top_cols
    ])


st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
          <div style="font-size:28px;">🏛️</div>
          <div class="brand-title">Credit Scoring - ANN</div>
          <div class="brand-sub">FEG Settat • Master DS &amp; Actuariat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_page = st.radio(
        "التنقل / Navigation",
        options=[
            "لوحة القيادة / Tableau de bord",
            "فحص الملفات والتنبؤ / Prédiction",
            "معلومات المشروع / À propos",
        ],
    )
    st.markdown("---")
    st.caption("Réalisé par : ISFAOUI NOUREDDINE & LAHMAMI MARYEM")
    st.caption("Encadré par : Prof. Benbrahime")

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات النموذج / Erreur de chargement : {e}")
    st.stop()

st.markdown("### Prédiction du Remboursement des Crédits — Credit Scoring ANN")
st.caption(f"Analysez vos données et prédisez le risque de défaut de paiement • Seuil de référence : **{threshold:.4f}**")
st.markdown(TOP_BANNER_HTML, unsafe_allow_html=True)

if nav_page == "معلومات المشروع / À propos":
    st.markdown(
        """
        <div class="glass-card">
          <div class="card-heading">البطاقة الأكاديمية للمشروع / Fiche Académique</div>
          <div class="card-desc">Informations officielles du projet de Master</div>
          <p><strong>Nom du projet :</strong> Credit Scoring - ANN (Prédiction de la Probabilité de Défaut)</p>
          <p><strong>Réalisé par :</strong> ISFAOUI NOUREDDINE &amp; LAHMAMI MARYEM</p>
          <p><strong>Encadré par :</strong> Prof. Benbrahime</p>
          <p><strong>Filière :</strong> Master Data Science et Actuariat</p>
          <p><strong>Établissement :</strong> Faculté d'Économie et de Gestion de Settat (FEG Settat)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

col_main, col_upload = st.columns([2.2, 1.1])

with col_upload:
    st.markdown(
        """
        <div class="glass-card">
          <div class="card-heading">Uploader votre fichier</div>
          <div class="card-desc">CSV, XLSX, XLS, JSON, TSV, TXT, Parquet, PDF</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "اختر ملف البيانات / Choisir un fichier",
        type=None,
        label_visibility="collapsed",
    )

if uploaded_file is not None:
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("last_file_signature") != file_signature:
        try:
            df_original = load_any_tabular_file(uploaded_file)
        except Exception as e:
            st.error(f"تعذر قراءة الملف / Erreur de lecture : {e}")
            st.stop()

        df_original = df_original.dropna(how="all").reset_index(drop=True)
        if df_original.empty:
            st.warning("الملف فارغ / Le fichier est vide.")
            st.stop()

        df_model_input, missing_cols, matched_cols = align_dataframe_to_preprocessor(df_original, preprocessor)

        try:
            probs = safe_predict_probabilities(model, preprocessor, df_model_input)
        except Exception as e:
            st.error(f"تعذر إتمام التنبؤ / Erreur de prédiction : {e}")
            st.stop()

        preds = (probs >= threshold).astype(int)
        tiers = [classify_risk_bucket(float(p)) for p in probs]

        df_out = df_original.copy()
        df_out["probability"] = np.round(probs, 6)
        df_out["prediction"] = preds
        df_out["شريحة الخطورة / Segment"] = tiers
        df_out["النتيجة / Résultat"] = np.where(
            preds == 1,
            "تعثر محتمل / Risque élevé",
            "سداد محتمل / Risque faible",
        )

        total_expected = max(1, len(missing_cols) + len(matched_cols))
        col_coverage_pct = (len(matched_cols) / total_expected) * 100
        cell_fill_pct = (1.0 - (float(df_model_input.isna().sum().sum()) / max(1, df_model_input.size))) * 100
        health_score = round(0.6 * col_coverage_pct + 0.4 * cell_fill_pct, 1)

        st.session_state["last_file_signature"] = file_signature
        st.session_state["last_file_name"] = uploaded_file.name
        st.session_state["df_out"] = df_out
        st.session_state["probs"] = probs
        st.session_state["preds"] = preds
        st.session_state["missing_cols"] = missing_cols
        st.session_state["health_score"] = health_score

    df_out = st.session_state["df_out"]
    probs = st.session_state["probs"]
    preds = st.session_state["preds"]
    missing_cols = st.session_state["missing_cols"]
    health_score = st.session_state["health_score"]
    file_name = st.session_state.get("last_file_name", uploaded_file.name)

    total_rows = len(df_out)
    high_count = int((preds == 1).sum())
    low_count = int((preds == 0).sum())
    high_pct = (high_count / total_rows) * 100 if total_rows else 0.0
    low_pct = (low_count / total_rows) * 100 if total_rows else 0.0
    mean_pd = float(np.mean(probs))

    with col_upload:
        st.success(f"Fichier chargé : {file_name}")
        if missing_cols:
            st.info(f"مواءمة تلقائية لـ {len(missing_cols)} عمود.")

    with col_main:
        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi-card k-blue">
                <div class="k-title">Enregistrements / الملفات</div>
                <div class="k-num">{total_rows:,}</div>
                <div class="k-sub">Qualité données : {health_score}%</div>
              </div>
              <div class="kpi-card k-green">
                <div class="k-title">Clients fiables / مخاطر منخفضة</div>
                <div class="k-num">{low_pct:.2f}%</div>
                <div class="k-sub">{low_count} dossiers (&lt; {threshold:.4f})</div>
              </div>
              <div class="kpi-card k-amber">
                <div class="k-title">Risque de défaut / مخاطر مرتفعة</div>
                <div class="k-num">{high_pct:.2f}%</div>
                <div class="k-sub">{high_count} dossiers (≥ {threshold:.4f})</div>
              </div>
              <div class="kpi-card k-purple">
                <div class="k-title">PD moyen / متوسط الاحتمال</div>
                <div class="k-num">{mean_pd:.4f}</div>
                <div class="k-sub">Seuil : {threshold:.4f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-card">
          <div class="card-heading">Aperçu des Données &amp; Distribution PD</div>
          <div class="card-desc">Répartition des probabilités de défaut (PD) et sensibilité aux seuils</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.bar_chart(build_pd_histogram_df(probs, bins=10), height=230)

    with st.expander("جدول حساسية العتبات / Analyse de sensibilité aux seuils"):
        st.dataframe(build_sensitivity_table(probs, threshold), use_container_width=True)

    st.markdown(
        """
        <div class="glass-card">
          <div class="card-heading">Résultat de la Prédiction Individuelle</div>
          <div class="card-desc">Inspectez un dossier précis, comparez-le à la moyenne et consultez les recommandations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_idx = st.selectbox("اختر رقم السطر / Sélectionner une ligne", options=df_out.index.tolist())
    row_item = df_out.loc[selected_idx]
    item_prob = float(row_item["probability"])
    item_pred = int(row_item["prediction"])

    r1, r2, r3 = st.columns(3)
    r1.metric("Probabilité de défaut", f"{item_prob:.4f}", delta=f"{item_prob - mean_pd:+.4f} vs moyenne")
    r2.metric("Segment de risque", row_item["شريحة الخطورة / Segment"])
    r3.metric("Décision", row_item["النتيجة / Résultat"])

    if item_pred == 1:
        st.error("Recommandation : Risque élevé — revue approfondie ou garantie supplémentaire recommandée.")
    else:
        st.success("Recommandation : Profil favorable — probabilité sous le seuil de référence.")

    dev_df = build_top_numeric_deviations(df_out.drop(columns=["probability", "prediction"], errors="ignore"), selected_idx)
    if not dev_df.empty:
        st.dataframe(dev_df, use_container_width=True)

    filter_choice = st.radio(
        "تصفية النتائج / Filtrer :",
        options=["الكل / Tous", "مخاطر مرتفعة فقط / Risque élevé", "مخاطر منخفضة فقط / Risque faible"],
        horizontal=True,
    )
    if filter_choice == "مخاطر مرتفعة فقط / Risque élevé":
        filtered_df = df_out[df_out["prediction"] == 1]
    elif filter_choice == "مخاطر منخفضة فقط / Risque faible":
        filtered_df = df_out[df_out["prediction"] == 0]
    else:
        filtered_df = df_out

    st.dataframe(filtered_df, use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="تحميل النتائج بصيغة CSV / Télécharger les résultats (CSV)",
        data=csv_bytes,
        file_name="credit_scoring_predictions.csv",
        mime="text/csv",
    )
else:
    with col_main:
        st.markdown(
            """
            <div class="glass-card">
              <div class="card-heading">بانتظار رفع ملف البيانات</div>
              <div class="card-desc">ارفع ملف CSV أو Excel أو أي صيغة أخرى من خانة التحميل لعرض المؤشرات والتوقعات الفورية.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
