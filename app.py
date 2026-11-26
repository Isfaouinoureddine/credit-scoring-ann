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
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">

    <style>
    @keyframes ledBlink {
        0%, 100% {
            opacity: 1;
            transform: scale(1.15);
            box-shadow: 0 0 8px currentColor, 0 0 16px currentColor;
        }
        50% {
            opacity: 0.25;
            transform: scale(0.8);
            box-shadow: 0 0 2px currentColor;
        }
    }

    /* نبض الأسماء وتأطير الأستاذ في مكانها الثابت */
    @keyframes authorPulse {
        0%, 100% {
            opacity: 1;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
            border-color: rgba(56, 189, 248, 0.55);
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.14);
        }
        50% {
            opacity: 0.45;
            text-shadow: 0 0 2px rgba(56, 189, 248, 0.08);
            border-color: rgba(148, 163, 184, 0.25);
            box-shadow: 0 0 4px rgba(56, 189, 248, 0.04);
        }
    }

    /* تتابع: مرحباً -> Bonjour -> Welcome */
    @keyframes seqAr {
        0%, 30%   { opacity: 1; transform: translateY(0); }
        34%, 100% { opacity: 0; transform: translateY(-4px); }
    }

    @keyframes seqFr {
        0%, 33%   { opacity: 0; transform: translateY(4px); }
        37%, 63%  { opacity: 1; transform: translateY(0); }
        67%, 100% { opacity: 0; transform: translateY(-4px); }
    }

    @keyframes seqEn {
        0%, 66%   { opacity: 0; transform: translateY(4px); }
        70%, 96%  { opacity: 1; transform: translateY(0); }
        100%      { opacity: 0; transform: translateY(-4px); }
    }

    /* دوران بطيء ومريح لاسم المشروع على كامل عرض الشاشة */
    @keyframes slowProjectOrbit {
        0% {
            transform: translateX(-100%);
        }
        100% {
            transform: translateX(100%);
        }
    }

    .global-showcase-card {
        width: 100%;
        border-radius: 20px;
        padding: 1.5px;
        background: linear-gradient(120deg, rgba(56, 189, 248, 0.65), rgba(16, 185, 129, 0.55), rgba(245, 158, 11, 0.55), rgba(168, 85, 247, 0.6));
        box-shadow: 0 14px 36px rgba(2, 6, 23, 0.28);
        margin-bottom: 20px;
    }

    .global-showcase-inner {
        width: 100%;
        border-radius: 18.5px;
        background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 42%),
                    radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.1), transparent 42%),
                    linear-gradient(145deg, #0b1120 0%, #111827 60%, #090d16 100
