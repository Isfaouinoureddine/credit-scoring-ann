import io, json, joblib, numpy as np, pandas as pd, streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Credit Scoring - ANN | FEG SETTAT", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
.stApp{background:radial-gradient(circle at 15% 12%,#13223e 0%,#090f1d 55%,#050811 100%);color:#f8fafc;font-family:"Plus Jakarta Sans","Cairo",sans-serif}
@keyframes blink{0%,100%{opacity:1;transform:scale(1.15);box-shadow:0 0 8px currentColor}50%{opacity:.22;transform:scale(.82)}}
@keyframes g1{0%,22%{opacity:1}25%,100%{opacity:0}}
@keyframes g2{0%,24%{opacity:0}27%,47%{opacity:1}50%,100%{opacity:0}}
@keyframes g3{0%,49%{opacity:0}52%,72%{opacity:1}75%,100%{opacity:0}}
@keyframes g4{0%,74%{opacity:0}77%,97%{opacity:1}100%{opacity:0}}
@keyframes t1{0%{max-width:0;border-right:2px solid #38bdf8}22%{max-width:100%;border-right:2px solid #38bdf8}25%,100%{max-width:100%;border-right-color:transparent}}
@keyframes t2{0%,24%{max-width:0;opacity:0}25%{max-width:0;opacity:1;border-right:2px solid #34d399}48%{max-width:100%;opacity:1;border-right:2px solid #34d399}51%,100%{max-width:100%;opacity:1;border-right-color:transparent}}
@keyframes t3{0%,50%{max-width:0;opacity:0}51%{max-width:0;opacity:1;border-right:2px solid #fbbf24}73%{max-width:100%;opacity:1;border-right:2px solid #fbbf24}76%,100%{max-width:100%;opacity:1;border-right-color:transparent}}
@keyframes t4{0%,75%{max-width:0;opacity:0}76%{max-width:0;opacity:1;border-right:2px solid #c084fc}95%{max-width:100%;opacity:1;border-right:2px solid #c084fc}98%,100%{max-width:100%;opacity:1;border-right-color:transparent}}
@keyframes shuttle{0%,100%{transform:translateX(0)}50%{transform:translateX(18%)}}
.frame{padding:2px;border-radius:18px;background:linear-gradient(135deg,#38bdf8,#34d399,#c084fc);margin-bottom:14px}
.inner{background:#0b1324;border-radius:16px;padding:14px 16px}
.top{display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid rgba(148,163,184,.16)}
.bulb{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;animation:blink 1.5s infinite}
.b1{background:#38bdf8;color:#38bdf8}.b2{background:#34d399;color:#34d399;animation-delay:.25s}.b3{background:#fbbf24;color:#fbbf24;animation-delay:.5s}.b4{background:#c084fc;color:#c084fc;animation-delay:.75s}
.gbox{position:relative;min-width:110px;height:28px}.gtag{position:absolute;right:0;font-size:14px;font-weight:800;padding:2px 10px;border-radius:999px;white-space:nowrap;opacity:0;direction:ltr;unicode-bidi:isolate}
.ga{color:#7dd3fc;background:rgba(56,189,248,.18);animation:g1 10s infinite}.gm{color:#e9d5ff;background:rgba(192,132,252,.18);animation:g2 10s infinite}.gf{color:#6ee7b7;background:rgba(52,211,153,.18);animation:g3 10s infinite}.ge{color:#fde68a;background:rgba(251,191,36,.18);animation:g4 10s infinite}
.info{background:rgba(15,23,42,.8);border:1px solid rgba(148,163,184,.15);border-radius:12px;padding:12px;margin-bottom:10px;font-size:13px;font-weight:700}
.lbl{font-size:11px;color:#38bdf8;font-weight:800;margin-bottom:6px}
.line{margin-bottom:6px;min-height:20px}
.s1{display:inline-block;overflow:hidden;white-space:nowrap;max-width:100%;padding-right:3px;animation:t1 10s steps(22,end) infinite}
.s2{display:inline-block;overflow:hidden;white-space:nowrap;max-width:100%;padding-right:3px;color:#6ee7b7;animation:t2 10s steps(18,end) infinite}
.s3{display:inline-block;overflow:hidden;white-space:nowrap;max-width:100%;padding-right:3px;color:#fde68a;animation:t3 10s steps(32,end) infinite}
.s4{display:inline-block;overflow:hidden;white-space:nowrap;max-width:100%;padding-right:3px;color:#e9d5ff;animation:t4 10s steps(44,end) infinite}
.track{overflow:hidden;background:rgba(30,41,59,.75);border:1px solid rgba(148,163,184,.18);border-radius:10px;padding:6px 10px}
.move{display:inline-block;font-size:12px;font-weight:800;animation:shuttle 5.5s ease-in-out infinite}
.pill{background:rgba(56,189,248,.22);border:1px solid rgba(125,211,252,.4);padding:1px 8px;border-radius:999px;margin:0 4px}
.frpill{background:rgba(52,211,153,.2);border:1px solid rgba(110,231,183,.4);color:#a7f3d0;padding:1px 8px;border-radius:999px}
[data-testid="stFileUploader"] section{background:#0f1b33!important;border:1.5px dashed rgba(56,189,248,.65)!important;border-radius:14px!important}
[data-testid="stFileUploader"] section *{color:#e2e8f0!important}
[data-testid="stFileUploader"] section button{background:#0284c7!important;color:#fff!important;border:1px solid #38bdf8!important;border-radius:8px!important}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:12px 0}
.kcard{background:rgba(15,23,42,.88);border:1px solid rgba(148,163,184,.16);border-top:3px solid #38bdf8;border-radius:12px;padding:12px}
</style>"""

BANNER = (
    '<div class="frame"><div class="inner"><div class="top">'
    '<div><span class="bulb b1"></span><span class="bulb b2"></span><span class="bulb b3"></span><span class="bulb b4"></span></div>'
    '<div class="gbox"><span class="gtag ga">مرحباً</span><span class="gtag gm">ⴰⵣⵓⵍ</span><span class="gtag gf">Bonjour</span><span class="gtag ge">Welcome</span></div>'
    '</div><div class="info">'
    '<div class="lbl">RÉALISÉ PAR :</div>'
    '<div class="line"><span class="s1">ISFAOUI NOUREDDINE</span></div>'
    '<div class="line"><span class="s2">LAHMAMI MARYEM</span></div>'
    '<div class="line" style="margin-top:6px;padding-top:6px;border-top:1px dashed rgba(148,163,184,.2)"><span class="s3">Encadré Par : Prof Benbrahime</span></div>'
    '<div class="line"><span class="s4">Master Data Science et Actuariat - FEG SETTAT</span></div>'
    '</div><div class="track"><div class="move">Projet : <span class="pill">Credit Scoring - ANN</span><span class="frpill">Prédiction du remboursement des crédits</span></div></div>'
    '</div></div>'
)


@st.cache_resource
def load_artifacts():
    model = load_model("ann_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    with open("threshold.json", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    return model, preprocessor, threshold


def read_file(f):
    b, name = f.getvalue(), (f.name or "").lower()
    if name.endswith((".xlsx", ".xls", ".ods")):
        return pd.read_excel(io.BytesIO(b))
    if name.endswith(".json"):
        d = json.loads(b.decode("utf-8-sig", errors="replace"))
        return pd.json_normalize(d) if isinstance(d, (list, dict)) else pd.read_json(io.StringIO(b.decode("utf-8-sig", errors="replace")))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(b))
    if name.endswith(".pdf"):
        import pdfplumber
        rows = [pd.DataFrame(t[1:], columns=[str(h or f"c{i}") for i, h in enumerate(t[0])]) for p in pdfplumber.open(io.BytesIO(b)).pages for t in (p.extract_tables() or []) if len(t) > 1]
        return pd.concat(rows, ignore_index=True)
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(io.BytesIO(b), sep=sep, on_bad_lines="skip")
            if df is not None and not df.empty and (df.shape[1] > 1 or sep == "|"):
                return df
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(b), on_bad_lines="skip")


def align_df(df, preprocessor):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        num_try = pd.to_numeric(df[col].astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
        if len(df) > 0 and num_try.notna().sum() >= max(1, int(0.7 * len(df))):
            df[col] = num_try
    cols = getattr(preprocessor, "feature_names_in_", None)
    if cols is None:
        return df, []
    mp = {str(c).strip().lower(): c for c in df.columns}
    out, missing = pd.DataFrame(index=df.index), []
    for c in cols:
        k = str(c).strip().lower()
        if k in mp:
            out[c] = df[mp[k]]
        else:
            out[c] = np.nan
            missing.append(str(c))
    return out, missing


st.markdown(CSS, unsafe_allow_html=True)
st.markdown("### Credit Scoring - ANN")
st.markdown(BANNER, unsafe_allow_html=True)

try:
    model, preprocessor, threshold = load_artifacts()
except Exception as e:
    st.error(f"خطأ في تحميل النموذج / Erreur : {e}")
    st.stop()

st.caption(f"العتبة المرجعية / Seuil de référence : **{threshold:.4f}**")

up = st.file_uploader("ارفع ملف البيانات / Charger un fichier (CSV, XLSX, JSON, Parquet, PDF)", type=None, key="uploader_main")
if up is not None:
    try:
        df_orig = read_file(up).dropna(how="all").reset_index(drop=True)
        if df_orig.empty:
            st.warning("الملف فارغ / Le fichier est vide.")
            st.stop()
        X_in, missing = align_df(df_orig, preprocessor)
        probs = model.predict(preprocessor.transform(X_in), verbose=0).reshape(-1)
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        st.error(f"تعذر معالجة الملف / Erreur : {e}")
        st.stop()

    df_out = df_orig.copy()
    df_out["probability"] = np.round(probs, 6)
    df_out["prediction"] = preds
    df_out["النتيجة / Résultat"] = np.where(preds == 1, "تعثر محتمل / Risque élevé", "سداد محتمل / Risque faible")

    if missing:
        st.info(f"مواءمة تلقائية لـ {len(missing)} عمود.")

    n = len(df_out)
    high = int((preds == 1).sum())
    low = n - high
    mean_pd = float(np.mean(probs))

    st.markdown(
        f'<div class="kgrid">'
        f'<div class="kcard"><small>Total dossiers</small><h3>{n:,}</h3></div>'
        f'<div class="kcard" style="border-top-color:#fbbf24"><small>Risque élevé</small><h3>{high} ({high/max(1,n)*100:.1f}%)</h3></div>'
        f'<div class="kcard" style="border-top-color:#34d399"><small>Risque faible</small><h3>{low} ({low/max(1,n)*100:.1f}%)</h3></div>'
        f'<div class="kcard" style="border-top-color:#c084fc"><small>PD moyen</small><h3>{mean_pd:.4f}</h3></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    counts, edges = np.histogram(probs, bins=10, range=(0.0, 1.0))
    st.bar_chart(pd.DataFrame({"Dossiers": counts}, index=[f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(10)]), height=210)

    idx = st.selectbox("اختر ملفاً / Sélectionner une ligne", df_out.index.tolist())
    c1, c2 = st.columns(2)
    c1.metric("Probabilité de défaut", f"{float(df_out.loc[idx, 'probability']):.4f}", f"{float(df_out.loc[idx, 'probability']) - mean_pd:+.4f} vs moyenne")
    c2.metric("Décision", df_out.loc[idx, "النتيجة / Résultat"])

    flt = st.radio("تصفية / Filtrer", ["الكل / Tous", "مخاطر مرتفعة / Risque élevé", "مخاطر منخفضة / Risque faible"], horizontal=True)
    view_df = df_out[df_out["prediction"] == 1] if flt.startswith("مخاطر مرتفعة") else df_out[df_out["prediction"] == 0] if flt.startswith("مخاطر منخفضة") else df_out
    st.dataframe(view_df, use_container_width=True)
    st.download_button("تحميل النتائج CSV / Télécharger CSV", view_df.to_csv(index=False).encode("utf-8-sig"), "credit_scoring_predictions.csv", "text/csv")
