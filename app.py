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
    page_title="Credit Scoring - ANN | Master Data Science & Actuariat",
    page_icon="📊",
    layout="wide",
)

LIGHT_THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

.stApp {
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  color: #0f172a;
  font-family: "Plus Jakarta Sans", "Cairo", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

@keyframes seqAr {
  0%, 22% { opacity: 1; transform: translateY(0); }
  25%, 100% { opacity: 0; transform: translateY(4px); }
}
@keyframes seqFr {
  0%, 24% { opacity: 0; transform: translateY(4px); }
  27%, 47% { opacity: 1; transform: translateY(0); }
  50%, 100% { opacity: 0; transform: translateY(4px); }
}
@keyframes seqEn {
  0%, 49% { opacity: 0; transform: translateY(4px); }
  52%, 72% { opacity: 1; transform: translateY(0); }
  75%, 100% { opacity: 0; transform: translateY(4px); }
}
@keyframes seqAmz {
  0%, 74% { opacity: 0; transform: translateY(4px); }
  77%, 97% { opacity: 1; transform: translateY(0); }
  100% { opacity: 0; transform: translateY(4px); }
}

@keyframes typeLine1 {
  0% { max-width: 0; border-right-color: #0369a1; }
  28% { max-width: 100%; border-right-color: #0369a1; }
  33%, 100% { max-width: 100%; border-right-color: transparent; }
}

@keyframes typeLine2 {
  0%, 30% { max-width: 0; opacity: 0; border-right-color: #0369a1; }
  31% { max-width: 0; opacity: 1; border-right-color: #0369a1; }
  62% { max-width: 100%; opacity: 1; border-right-color: #0369a1; }
  67%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

@keyframes typeLine3 {
  0%, 64% { max-width: 0; opacity: 0; border-right-color: #0369a1; }
  65% { max-width: 0; opacity: 1; border-right-color: #0369a1; }
  92% { max-width: 100%; opacity: 1; border-right-color: #0369a1; }
  97%, 100% { max-width: 100%; opacity: 1; border-right-color: transparent; }
}

.showcase-card {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 22px;
  padding: 24px 26px;
  margin-bottom: 20px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
}

.big-greetings-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  min-height: 54px;
  padding: 10px 18px;
  margin-bottom: 20px;
  background: linear-gradient(135deg, #f0f9ff 0%, #f8fafc 100%);
  border: 1px solid #bae6fd;
  border-radius: 16px;
}

.big-greet-word {
  position: absolute;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 0.5px;
  padding: 4px 20px;
  border-radius: 999px;
  white-space: nowrap;
  opacity: 0;
}

.big-greet-ar {
  font-family: "Cairo", sans-serif;
  color: #0369a1;
  background: #e0f2fe;
  animation: seqAr 9.5s infinite ease-in-out;
}
.big-greet-fr {
  color: #0f766e;
  background: #ccfbf1;
  animation: seqFr 9.5s infinite ease-in-out;
}
.big-greet-en {
  color: #b45309;
  background: #fef3c7;
  animation: seqEn 9.5s infinite ease-in-out;
}
.big-greet-amz {
  color: #6d28d9;
  background: #ede9fe;
  animation: seqAmz 9.5s infinite ease-in-out;
}

.academic-stage {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 5px solid #0369a1;
  border-radius: 16px;
  padding: 18px 20px;
}

.line-row {
  margin-bottom: 12px;
  line-height: 1.65;
}
.line-row:last-child {
  margin-bottom: 0;
}

.label-unified {
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.9px;
  text-transform: uppercase;
  color: #0369a1;
  margin-right: 6px;
}

.value-styled {
  font-family: "Plus Jakarta Sans", "Cairo", sans-serif;
  font-size: 16.5px;
  font-weight: 700;
  letter-spacing: 0.4px;
  color: #0f172a;
}

.type-stream-1 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 4px;
  animation: typeLine1 9.5s steps(36, end) infinite;
}

.type-stream-2 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 4px;
  animation: typeLine2 9.5s steps(56, end) infinite;
}

.type-stream-3 {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  max-width: 100%;
  border-right: 2px solid transparent;
  padding-right: 4px;
  animation: typeLine3 9.5s steps(48, end) infinite;
}

.sec-head {
  margin: 18px 0 12px 0;
  padding: 11px 16px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #0369a1;
  border-radius: 12px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
}

.sec-title {
  font-size: 17px;
  font-weight: 800;
  color: #0f172a;
}

.sec-sub {
  font-size: 12.5px;
  color: #64748b;
  margin-top: 2px;
}

.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.kpi-tile {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 15px 16px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.tile-blue { border-top: 3px solid #0369a1; }
.tile-red { border-top: 3px solid #dc2626; background: #fffafa; }
.tile-green { border-top: 3px solid #059669; background: #f8fdfb; }
.tile-purple { border-top: 3px solid #7c3aed; }

.tile-label {
  font-size: 12.5px;
  font-weight: 700;
  color: #475569;
  margin-bottom: 6px;
}
.tile-val {
  font-size: 24px;
  font-weight: 800;
  color: #0f172a;
  line-height: 1.2;
}
.tile-note {
  font-size: 12px;
  color: #64748b;
  margin-top: 6px;
}
</style>"""

SHOWCASE_HTML = """
<div class="showcase-card">
  <div class="big-greetings-bar">
    <span class="big-greet-word big-greet-ar">مرحباً</span>
    <span class="big-greet-word big-greet-fr">Bonjour</span>
    <span class="big-greet-word big-greet-en">Welcome</span>
    <span class="big-greet-word big-greet-amz">ⴰⵣⵓⵍ / Azul</span>
  </div>
  <div class="academic-stage">
    <div class="line-row">
      <span class="type-stream-1">
        <span class="label-unified">ENCADRÉ PAR :</span>
        <span class="value-styled">Prof Benbrahime</span>
      </span>
    </div>
    <div class="line-row">
      <span class="type-stream-2">
        <span class="label-unified">RÉALISÉ PAR :</span>
        <span class="value-styled">ISFAOUI NOUREDDINE &amp; LAHMAMI MARYEM</span>
      </span>
    </div>
    <div class="line-row">
      <span class="type-stream-3">
        <span class="label-unified">FILIÈRE :</span>
        <span class="value-styled">Master Data Science et Actuariat</span>
      </span>
    </div>
  </div>
</div>
"""


def render_section_header(title: str, subtitle: str, accent_color: str = "#0369a1"):
    st.markdown(
        f"""
        <div class="sec-head" style="border-left-color: {accent_color};">
          <div class="sec-title">{title}</div>
          <div class="sec-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
        cols = []
        for c in raw_cols:
            if c is not None and not isinstance(c, (dict, set)):
                cols.append(c)
        return cols if cols else None
    except Exception:
        return None


def _promote_header_if_needed(df: pd.DataFrame, expected_cols=None) -> pd.DataFrame:
    try:
        if df is None or df.empty or len(df) <= 1:
            return df
        exp_norm = {_norm_key(c) for c in (expected_cols or []) if _norm_key(c)}
        first_row = df.iloc[0].tolist()
        first_row_vals = [str(v).strip() for v in first_row if pd.notna(v) and str(v).strip()]
        first_norm = {_norm_key(v) for v in first_row_vals if _norm_key(v)}

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
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except Exception:
                counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
                sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة الملف النصي أو CSV / Impossible de décoder le fichier CSV.")


def read_excel_smart(raw_bytes: bytes) -> pd.DataFrame:
    for engine in ("openpyxl", "xlrd", "odf", None):
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    raise ValueError("تعذر قراءة ملف Excel / Impossible de lire le fichier Excel.")


def read_pdf_tables(raw_bytes: bytes) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("مكتبة pdfplumber غير مثبّتة لقراءة جداول PDF.") from exc

    tables_dfs = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
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
    raise ValueError(f"تعذر قراءة الملف بأي صيغة مدعومة ({last_err}).")


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
    records = []
    for col in top_cols:
        val = float(row_vals[col])
        avg = float(means[col])
        records.append({
            "المتغير / Variable": col,
            "قيمة الملف / Valeur": round(val, 4),
            "متوسط المحفظة / Moyenne": round(avg, 4),
            "الفارق / Écart": round(val - avg, 4),
        })
    return pd.DataFrame(records)


def build_executive_report(file_name: str, total_rows: int, high_count: int, high_pct: float, mean_pd: float, median_pd: float, threshold: float) -> str:
    return (
        "RAPPORT SYNTHÉTIQUE - CREDIT SCORING (ANN)\n"
        "===========================================\n"
        f"Fichier analysé : {file_name}\n"
        f"Seuil de référence : {threshold:.4f}\n"
        f"Total dossiers : {total_rows}\n"
        f"Risque élevé : {high_count} ({high_pct:.1f}%)\n"
        f"PD moyen : {mean_pd:.4f} | Médiane : {median_pd:.4f}\n"
        "Encadré par : Prof Benbrahime\n"
        "Réalisé par : ISFAOUI NOUREDDINE & LAHMAMI MARYEM\n"
        "Filière : Master Data Science et Actuariat\n"
    )


st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
st.title("Credit Scoring - ANN")
st.markdown(SHOWCASE_HTML, unsafe_allow_html=True)

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات النموذج / Erreur de chargement : {e}")
    st.stop()

st.caption(f"العتبة المرجعية المعتمدة في النموذج / Seuil de référence : **{threshold:.4f}**")

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
            st.error(f"تعذر قراءة الملف المرفوع / Erreur de lecture : {e}")
            st.stop()

        df_original = df_original.dropna(how="all").reset_index(drop=True)
        if df_original.empty:
            st.warning("الملف فارغ أو لا يحتوي على صفوف صالحة / Le fichier est vide.")
            st.stop()

        df_model_input, missing_cols, matched_cols = align_dataframe_to_preprocessor(df_original, preprocessor)

        try:
            probs = safe_predict_probabilities(model, preprocessor, df_model_input)
        except Exception as e:
            st.error(f"تعذر إتمام التنبؤ على هذا الملف / Erreur lors de la prédiction : {e}")
            st.stop()

        preds = (probs >= threshold).astype(int)
        tiers = [classify_risk_bucket(float(p)) for p in probs]

        df_out = df_original.copy()
        df_out["probability"] = np.round(probs, 6)
        df_out["prediction"] = preds
        df_out["شريحة الخطورة / Segment"] = tiers
        df_out["النتيجة / Résultat"] = np.where(
            preds == 1,
            "تعثر محتمل / Risque élevé (Défaut)",
            "سداد محتمل / Risque faible (Bon client)",
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
        st.info(
            f"مواءمة تلقائية / Alignement : تم تعويض {len(missing_cols)} عمود غير موجود في الملف ({', '.join(missing_cols[:6])}{'...' if len(missing_cols) > 6 else ''})."
        )

    total_rows = len(df_out)
    high_risk_count = int((preds == 1).sum())
    low_risk_count = int((preds == 0).sum())
    high_risk_pct = (high_risk_count / total_rows) * 100 if total_rows else 0.0
    low_risk_pct = (low_risk_count / total_rows) * 100 if total_rows else 0.0
    mean_pd = float(np.mean(probs))
    median_pd = float(np.median(probs))

    render_section_header(
        "ملخص المحفظة والتوقعات / Résumé du portefeuille",
        "مؤشرات المخاطر الرئيسية وجودة البيانات المدخلة",
        accent_color="#0369a1",
    )

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-tile tile-blue">
            <div class="tile-label">إجمالي الملفات / Total dossiers</div>
            <div class="tile-val">{total_rows}</div>
            <div class="tile-note">Score qualité données : {health_score}%</div>
          </div>
          <div class="kpi-tile tile-red">
            <div class="tile-label">مخاطر مرتفعة / Risque élevé</div>
            <div class="tile-val">{high_risk_count} ({high_risk_pct:.1f}%)</div>
            <div class="tile-note">Probabilité ≥ {threshold:.4f}</div>
          </div>
          <div class="kpi-tile tile-green">
            <div class="tile-label">مخاطر منخفضة / Risque faible</div>
            <div class="tile-val">{low_risk_count} ({low_risk_pct:.1f}%)</div>
            <div class="tile-note">Probabilité &lt; {threshold:.4f}</div>
          </div>
          <div class="kpi-tile tile-purple">
            <div class="tile-label">متوسط الاحتمال / PD moyen</div>
            <div class="tile-val">{mean_pd:.4f}</div>
            <div class="tile-note">Médiane : {median_pd:.4f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if high_risk_pct >= 40.0:
        st.warning(f"تنبيه المحفظة / Alerte portefeuille : نسبة المخاطر المرتفعة مرتفعة ({high_risk_pct:.1f}%).")

    render_section_header(
        "توزيع الاحتمالات وحساسية العتبة / Distribution & Sensibilité",
        "توزيع احتمالات التعثر في المحفظة مع مقارنة تأثير العتبات المختلفة",
        accent_color="#7c3aed",
    )
    hist_df = build_pd_histogram_df(probs, bins=10)
    st.bar_chart(hist_df, height=230)

    with st.expander("جدول حساسية العتبات / Analyse de sensibilité aux seuils"):
        st.dataframe(build_sensitivity_table(probs, threshold), use_container_width=True)

    render_section_header(
        "فحص ملف فردي ومقارنته بالمحفظة / Inspecteur individuel",
        "اختر سطراً لمقارنة احتمال التعثر بمعدل المحفظة والاطلاع على أبرز الفوارق العددية",
        accent_color="#059669",
    )

    selected_idx = st.selectbox(
        "اختر رقم السطر / Sélectionner une ligne (Index)",
        options=df_out.index.tolist(),
    )
    row_item = df_out.loc[selected_idx]
    item_prob = float(row_item["probability"])
    item_pred = int(row_item["prediction"])
    diff_vs_mean = item_prob - mean_pd

    c1, c2, c3 = st.columns(3)
    c1.metric("احتمال التعثر / Probabilité", f"{item_prob:.4f}", delta=f"{diff_vs_mean:+.4f} vs PD moyen")
    c2.metric("شريحة الخطورة / Segment", row_item["شريحة الخطورة / Segment"])
    c3.metric("القرار / Décision", row_item["النتيجة / Résultat"])

    if item_pred == 1:
        st.error("توصية / Recommandation : احتمال التعثر يتجاوز العتبة؛ يُنصح بمراجعة إضافية أو طلب ضمانات إضافية.")
    else:
        st.success("توصية / Recommandation : احتمال التعثر ضمن الحدود المقبولة مقارنة بالعتبة المعتمدة.")

    dev_df = build_top_numeric_deviations(df_out.drop(columns=["probability", "prediction"], errors="ignore"), selected_idx)
    if not dev_df.empty:
        st.caption("أبرز المتغيرات الرقمية من حيث الاختلاف عن متوسط المحفظة لهذا الملف:")
        st.dataframe(dev_df, use_container_width=True)

    render_section_header(
        "جدول النتائج والتصدير / Tableau des résultats",
        "تصفية الملفات وتحميل النتائج النهائية أو التقرير الملخص",
        accent_color="#d97706",
    )

    filter_choice = st.radio(
        "تصفية النتائج / Filtrer les résultats :",
        options=[
            "الكل / Tous",
            "مخاطر مرتفعة فقط / Risque élevé",
            "مخاطر منخفضة فقط / Risque faible",
        ],
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
        high_count=high_risk_count,
        high_pct=high_risk_pct,
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
