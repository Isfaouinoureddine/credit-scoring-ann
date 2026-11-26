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
    page_title="Credit Scoring - ANN | FEG SETTAT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

.stApp {
  background: radial-gradient(circle at 15% 12%, #13223e 0%, #090f1d 52%, #050811 100%);
  color: #f8fafc;
  font-family: "Plus Jakarta Sans", "Cairo", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

@keyframes bulbBlink {
  0%, 100% {
    opacity: 1;
    transform: scale(1.18);
    box-shadow: 0 0 8px currentColor, 0 0 16px currentColor;
  }
  50% {
    opacity: 0.22;
    transform: scale(0.82);
    box-shadow: 0 0 1px currentColor;
  }
}

@keyframes greetAr {
  0%, 22% { opacity: 1; transform: translateY(0); }
  25%, 100% { opacity: 0; transform: translateY(3px); }
}
@keyframes greetAmz {
  0%, 24% { opacity: 0; transform: translateY(3px); }
  27%, 47% { opacity: 1; transform: translateY(0); }
  50%, 100% { opacity: 0; transform: translateY(3px); }
}
@keyframes greetFr {
  0%, 49% { opacity: 0; transform: translateY(3px); }
  52%, 72% { opacity: 1; transform: translateY(0); }
  75%, 100% { opacity: 0; transform: translateY(3px); }
}
@keyframes greetEn {
  0%, 74% { opacity: 0; transform: translateY(3px); }
  77%, 97% { opacity: 1; transform: translateY(0); }
  100% { opacity: 0; transform: translateY(3px); }
}

@keyframes charStep1 {
  0% { max-width: 0; border-right-color: #38bdf8; }
  22% { max-width: 100%; border-right-color: #38bdf8; }
  25%, 100% { max-width: 100%; border-right-color: transparent; }
}

@keyframes charStep2 {
  0%, 24% { max-width: 0; opacity: 0; border-right-color: #34d399; }
  25% { max-width: 0; opacity: 1; border-right-color: #34d399; }
  48% { max-width: 100%; opacity: 1; border-right-color: #34d399; }
  51%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

@keyframes charStep3 {
  0%, 50% { max-width: 0; opacity: 0; border-right-color: #fbbf24; }
  51% { max-width: 0; opacity: 1; border-right-color: #fbbf24; }
  73% { max-width: 100%; opacity: 1; border-right-color: #fbbf24; }
  76%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

@keyframes charStep4 {
  0%, 75% { max-width: 0; opacity: 0; border-right-color: #c084fc; }
  76% { max-width: 0; opacity: 1; border-right-color: #c084fc; }
  95% { max-width: 100%; opacity: 1; border-right-color: #c084fc; }
  98%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

@keyframes goAndBack {
  0% { transform: translateX(0%); }
  50% { transform: translateX(28%); }
  100% { transform: translateX(0%); }
}

.glow-frame {
  padding: 2px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.55), rgba(52, 211, 153, 0.35), rgba(192, 132, 252, 0.5));
  box-shadow: 0 14px 34px rgba(2, 132, 199, 0.22);
  margin-bottom: 16px;
}

.glow-inner {
  background: linear-gradient(160deg, #0d172a 0%, #09101e 100%);
  border-radius: 18px;
  padding: 15px 17px;
}

.header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding-bottom: 11px;
  margin-bottom: 12px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}

.bulbs-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}

.bulb {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
  animation: bulbBlink 1.5s infinite ease-in-out;
}
.b-cyan { background: #38bdf8; color: #38bdf8; animation-delay: 0s; }
.b-green { background: #34d399; color: #34d399; animation-delay: 0.25s; }
.b-amber { background: #fbbf24; color: #fbbf24; animation-delay: 0.5s; }
.b-violet { background: #c084fc; color: #c084fc; animation-delay: 0.75s; }

.greet-box {
  position: relative;
  min-width: 110px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}

.greet-tag {
  position: absolute;
  right: 0;
  font-size: 15px;
  font-weight: 800;
  padding: 2px 12px;
  border-radius: 999px;
  white-space: nowrap;
  opacity: 0;
}
.gt-ar { font-family: "Cairo", sans-serif; color: #7dd3fc; background: rgba(56, 189, 248, 0.18); border: 1px solid rgba(125, 211, 252, 0.35); animation: greetAr 10s infinite ease-in-out; }
.gt-amz { color: #e9d5ff; background: rgba(192, 132, 252, 0.18); border: 1px solid rgba(216, 180, 254, 0.35); animation: greetAmz 10s infinite ease-in-out; }
.gt-fr { color: #6ee7b7; background: rgba(52, 211, 153, 0.18); border: 1px solid rgba(110, 231, 183, 0.35); animation: greetFr 10s infinite ease-in-out; }
.gt-en { color: #fde68a; background: rgba(251, 191, 36, 0.18); border: 1px solid rgba(253, 230, 138, 0.35); animation: greetEn 10s infinite ease-in-out; }

.info-box {
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 12px;
}

.section-lbl {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.9px;
  text-transform: uppercase;
  color: #38bdf8;
  margin-bottom: 6px;
}

.row-line {
  margin-bottom: 6px;
  line-height: 1.45;
  min-height: 22px;
}
.row-line:last-child {
  margin-bottom: 0;
}

.small-txt {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.3px;
  color: #f1f5f9;
}

.role-prefix {
  font-size: 12px;
  font-weight: 700;
  color: #94a3b8;
  margin-right: 4px;
}

.type-1 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 3px;
  color: #f8fafc;
  animation: charStep1 10s steps(22, end) infinite;
}

.type-2 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 3px;
  color: #6ee7b7;
  animation: charStep2 10s steps(18, end) infinite;
}

.type-3 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 3px;
  color: #fde68a;
  animation: charStep3 10s steps(32, end) infinite;
}

.type-4 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 3px;
  color: #e9d5ff;
  animation: charStep4 10s steps(44, end) infinite;
}

.shuttle-track {
  width: 100%;
  overflow: hidden;
  background: rgba(30, 41, 59, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 10px;
  padding: 6px 10px;
}

.shuttle-item {
  display: inline-block;
  font-size: 12.5px;
  font-weight: 800;
  color: #e2e8f0;
  animation: goAndBack 5.5s ease-in-out infinite;
}

.shuttle-pill {
  background: rgba(56, 189, 248, 0.22);
  border: 1px solid rgba(125, 211, 252, 0.42);
  color: #ffffff;
  padding: 1px 9px;
  border-radius: 999px;
  margin: 0 4px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.kpi-card {
  background: rgba(15, 23, 42, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 12px 14px;
}
.k-blue { border-top: 3px solid #38bdf8; }
.k-green { border-top: 3px solid #34d399; }
.k-amber { border-top: 3px solid #fbbf24; }
.k-purple { border-top: 3px solid #c084fc; }

.k-title { font-size: 11.5px; font-weight: 700; color: #94a3b8; margin-bottom: 4px; }
.k-num { font-size: 21px; font-weight: 800; color: #ffffff; line-height: 1.2; }
.k-sub { font-size: 11px; color: #cbd5e1; margin-top: 4px; }

.panel-card {
  background: rgba(15, 23, 42, 0.84);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 13px 15px;
  margin-bottom: 12px;
}
.panel-title { font-size: 14.5px; font-weight: 800; color: #f8fafc; margin-bottom: 2px; }
.panel-sub { font-size: 11.5px; color: #94a3b8; }
</style>"""

SHOWCASE_HTML = (
    '<div class="glow-frame">'
    '<div class="glow-inner">'
    '<div class="header-bar">'
    '<div class="bulbs-wrap">'
    '<span class="bulb b-cyan"></span>'
    '<span class="bulb b-green"></span>'
    '<span class="bulb b-amber"></span>'
    '<span class="bulb b-violet"></span>'
    '</div>'
    '<div class="greet-box">'
    '<span class="greet-tag gt-ar">مرحباً</span>'
    '<span class="greet-tag gt-amz">ⴰⵣⵓⵍ</span>'
    '<span class="greet-tag gt-fr">Bonjour</span>'
    '<span class="greet-tag gt-en">Welcome</span>'
    '</div>'
    '</div>'
    '<div class="info-box">'
    '<div class="section-lbl">RÉALISÉ PAR :</div>'
    '<div class="row-line small-txt"><span class="type-1">ISFAOUI NOUREDDINE</span></div>'
    '<div class="row-line small-txt"><span class="type-2">LAHMAMI MARYEM</span></div>'
    '<div class="row-line small-txt" style="margin-top:8px;padding-top:8px;border-top:1px dashed rgba(148,163,184,0.2);">'
    '<span class="type-3"><span class="role-prefix">Encadré Par :</span> Prof Benbrahime</span>'
    '</div>'
    '<div class="row-line small-txt"><span class="type-4">Master Data Science et Actuariat - FEG SETTAT</span></div>'
    '</div>'
    '<div class="shuttle-track">'
    '<div class="shuttle-item">Projet : <span class="shuttle-pill">Credit Scoring - ANN</span></div>'
    '</div>'
    '</div>'
    '</div>'
)


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
                    rows = table[1:]
                    tables_dfs.append(pd.DataFrame(rows, columns=headers))
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


def build_executive_report(file_name: str, total_rows: int, high_count: int, high_pct: float, mean_pd: float, median_pd: float, threshold: float) -> str:
    return (
        "RAPPORT SYNTHÉTIQUE - CREDIT SCORING (ANN)\n"
        "===========================================\n"
        f"Fichier analysé : {file_name}\n"
        f"Seuil de référence : {threshold:.4f}\n"
        f"Total dossiers : {total_rows}\n"
        f"Risque élevé : {high_count} ({high_pct:.1f}%)\n"
        f"PD moyen : {mean_pd:.4f} | Médiane : {median_pd:.4f}\n"
        "Réalisé par : ISFAOUI NOUREDDINE & LAHMAMI MARYEM\n"
        "Encadré par : Prof Benbrahime\n"
        "Filière : Master Data Science et Actuariat - FEG SETTAT\n"
    )


st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown("### Credit Scoring - ANN")
st.markdown(SHOWCASE_HTML, unsafe_allow_html=True)

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات النموذج / Erreur de chargement : {e}")
    st.stop()

st.caption(f"العتبة المرجعية / Seuil de référence : **{threshold:.4f}**")

uploaded_file = st.file_uploader(
    "ارفع ملف البيانات / Charger un fichier (CSV, XLSX, XLS, JSON, TSV, TXT, Parquet, PDF)",
    type=None,
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

    if missing_cols:
        st.info(f"مواءمة تلقائية لـ {len(missing_cols)} عمود غير موجود في الملف.")

    total_rows = len(df_out)
    high_count = int((preds == 1).sum())
    low_count = int((preds == 0).sum())
    high_pct = (high_count / total_rows) * 100 if total_rows else 0.0
    low_pct = (low_count / total_rows) * 100 if total_rows else 0.0
    mean_pd = float(np.mean(probs))
    median_pd = float(np.median(probs))

    kpi_html = (
        '<div class="kpi-grid">'
        '<div class="kpi-card k-blue">'
        '<div class="k-title">إجمالي الملفات / Total dossiers</div>'
        f'<div class="k-num">{total_rows:,}</div>'
        f'<div class="k-sub">Qualité données : {health_score}%</div>'
        '</div>'
        '<div class="kpi-card k-amber">'
        '<div class="k-title">مخاطر مرتفعة / Risque élevé</div>'
        f'<div class="k-num">{high_count} ({high_pct:.1f}%)</div>'
        f'<div class="k-sub">Probabilité ≥ {threshold:.4f}</div>'
        '</div>'
        '<div class="kpi-card k-green">'
        '<div class="k-title">مخاطر منخفضة / Risque faible</div>'
        f'<div class="k-num">{low_count} ({low_pct:.1f}%)</div>'
        f'<div class="k-sub">Probabilité &lt; {threshold:.4f}</div>'
        '</div>'
        '<div class="kpi-card k-purple">'
        '<div class="k-title">متوسط الاحتمال / PD moyen</div>'
        f'<div class="k-num">{mean_pd:.4f}</div>'
        f'<div class="k-sub">Médiane : {median_pd:.4f}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-card"><div class="panel-title">توزيع الاحتمالات وحساسية العتبة / Distribution &amp; Sensibilité</div>'
        '<div class="panel-sub">توزيع احتمالات التعثر في المحفظة ومقارنة العتبات</div></div>',
        unsafe_allow_html=True,
    )
    st.bar_chart(build_pd_histogram_df(probs, bins=10), height=220)

    with st.expander("جدول حساسية العتبات / Analyse de sensibilité aux seuils"):
        st.dataframe(build_sensitivity_table(probs, threshold), use_container_width=True)

    st.markdown(
        '<div class="panel-card"><div class="panel-title">فحص ملف فردي / Inspecteur individuel</div>'
        '<div class="panel-sub">مقارنة الملف المختار بمتوسط المحفظة وأبرز الفوارق العددية</div></div>',
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

    d_col1, d_col2 = st.columns(2)
    export_df = filtered_df.copy()
    export_df["threshold_used"] = round(threshold, 6)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
    with d_col1:
        st.download_button(
            label="تحميل النتائج بصيغة CSV / Télécharger les résultats (CSV)",
            data=csv_bytes,
            file_name="credit_scoring_predictions.csv",
            mime="text/csv",
        )

    report_text = build_executive_report(
        file_name=file_name,
        total_rows=total_rows,
        high_count=high_count,
        high_pct=high_pct,
        mean_pd=mean_pd,
        median_pd=median_pd,
        threshold=threshold,
    ).encode("utf-8")
    with d_col2:
        st.download_button(
            label="تحميل التقرير الملخص / Télécharger le rapport (TXT)",
            data=report_text,
            file_name="credit_scoring_summary.txt",
            mime="text/plain",
        )
