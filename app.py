import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Credit Scoring - ANN",
    layout="wide"
)

st.title("Credit Scoring - ANN")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@600;700;800&display=swap" rel="stylesheet">

    <style>
    @keyframes lightToggle {
        0%, 100% {
            opacity: 1;
            transform: scale(1.18);
            box-shadow: 0 0 8px currentColor, 0 0 18px currentColor;
        }
        50% {
            opacity: 0.2;
            transform: scale(0.78);
            box-shadow: 0 0 2px currentColor;
        }
    }

    /* ظهور واختفاء بالتتابع: مرحباً -> Bonjour -> Welcome */
    @keyframes greetAr {
        0%, 30%   { opacity: 1; transform: translateY(0px); }
        34%, 100% { opacity: 0; transform: translateY(-4px); }
    }

    @keyframes greetFr {
        0%, 33%   { opacity: 0; transform: translateY(4px); }
        37%, 63%  { opacity: 1; transform: translateY(0px); }
        67%, 100% { opacity: 0; transform: translateY(-4px); }
    }

    @keyframes greetEn {
        0%, 66%   { opacity: 0; transform: translateY(4px); }
        70%, 96%  { opacity: 1; transform: translateY(0px); }
        100%      { opacity: 0; transform: translateY(-4px); }
    }

    /* دوران الشريط بالكامل على عرض الشاشة */
    @keyframes screenScroll {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }

    .bright-marquee-shell {
        width: 100%;
        border-radius: 18px;
        padding: 2px;
        background: linear-gradient(90deg, #38bdf8, #34d399, #fbbf24, #f472b6, #a78bfa, #38bdf8);
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.16);
        margin-bottom: 18px;
    }

    .bright-marquee-panel {
        width: 100%;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 50%, #fefce8 100%);
        border-radius: 16px;
        padding: 12px 16px;
        overflow: hidden;
    }

    .lights-and-hello {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
        flex-wrap: wrap;
    }

    .led-strip {
        display: flex;
        align-items: center;
        gap: 9px;
    }

    .led-lamp {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        animation: lightToggle 1.2s infinite ease-in-out;
    }

    .lamp-1 { background-color: #0ea5e9; color: #0ea5e9; animation-delay: 0s; }
    .lamp-2 { background-color: #10b981; color: #10b981; animation-delay: 0.2s; }
    .lamp-3 { background-color: #f59e0b; color: #f59e0b; animation-delay: 0.4s; }
    .lamp-4 { background-color: #ec4899; color: #ec4899; animation-delay: 0.6s; }
    .lamp-5 { background-color: #8b5cf6; color: #8b5cf6; animation-delay: 0.8s; }

    .hello-switcher {
        position: relative;
        min-width: 120px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 16px;
    }

    .hello-item {
        position: absolute;
        right: 0;
        white-space: nowrap;
        padding: 1px 10px;
        border-radius: 999px;
        opacity: 0;
    }

    .hello-ar {
        color: #0369a1;
        background: #e0f2fe;
        animation: greetAr 4.5s infinite ease-in-out;
    }

    .hello-fr {
        color: #047857;
        background: #d1fae5;
        animation: greetFr 4.5s infinite ease-in-out;
    }

    .hello-en {
        color: #b45309;
        background: #fef3c7;
        animation: greetEn 4.5s infinite ease-in-out;
    }

    .scroll-viewport {
        width: 100%;
        overflow: hidden;
        padding: 5px 0;
    }

    .scroll-content {
        display: inline-block;
        white-space: nowrap;
        font-family: 'Cairo', sans-serif;
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        animation: screenScroll 16s linear infinite;
    }

    .scroll-content:hover {
        animation-play-state: paused;
    }

    .proj-highlight {
        color: #0284c7;
        background: #e0f2fe;
        padding: 1px 8px;
        border-radius: 8px;
    }

    .students-highlight {
        color: #047857;
        background: #d1fae5;
        padding: 1px 8px;
        border-radius: 8px;
    }
    </style>

    <div class="bright-marquee-shell">
        <div class="bright-marquee-panel">
            <div class="lights-and-hello">
                <div class="led-strip">
                    <span class="led-lamp lamp-1"></span>
                    <span class="led-lamp lamp-2"></span>
                    <span class="led-lamp lamp-3"></span>
                    <span class="led-lamp lamp-4"></span>
                    <span class="led-lamp lamp-5"></span>
                </div>
                <div class="hello-switcher">
                    <span class="hello-item hello-ar">مرحباً</span>
                    <span class="hello-item hello-fr">Bonjour</span>
                    <span class="hello-item hello-en">Welcome</span>
                </div>
            </div>
            <div class="scroll-viewport">
                <div class="scroll-content">
                    Nous sommes les étudiants : <span class="students-highlight">ISFAOUI NOUREDDINE</span> et <span class="students-highlight">LAHMAMI MARYEM</span> — طلبة ماستر data science et actuariat — FEG SETTAT | Projet : <span class="proj-highlight">Credit Scoring - ANN</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model("ann_model.keras")
preprocessor = joblib.load("preprocessor.pkl")
with open("threshold.json", encoding="utf-8") as f:
    threshold = float(json.load(f)["threshold"])

uploaded_file = st.file_uploader(
    "ارفع ملف للتجربة (CSV أو Excel)",
    type=["csv", "xlsx"]
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
    probs = model.predict(X_proc, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)

    df["probability"] = probs
    df["prediction"] = np.where(preds == 1, 1, 0)
    st.dataframe(df)
