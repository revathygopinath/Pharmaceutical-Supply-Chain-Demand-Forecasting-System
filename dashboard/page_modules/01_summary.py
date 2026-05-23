"""pages/01_summary.py -- Overview"""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from dashboard.components.utils import (
    page_header, section_header, narrative, load_csv, THEME
)

page_header(
    "Overview",
    "Pharmaceutical Supply Chain Demand Forecasting System"
)

narrative(
    "This project is an end-to-end machine learning system for forecasting weekly "
    "pharmaceutical demand using five years of historical sales data (2014-2019). "
    "The system evaluates Naive, Prophet, XGBoost Local, LightGBM Local and Global "
    "LightGBM models across eight drug categories, generates 8-week forecasts, performs "
    "walk-forward retraining simulation, and supports procurement planning through "
    "supply-chain analytics dashboards. Four drugs were deployed based on forecast "
    "accuracy (MAPE &lt; 20%) and demand archetype diversity."
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
section_header("Dataset and Methodology")
c1, c2, c3 = st.columns(3)

with c1:
    lines = [
        "Source: salesweekly.csv",
        "Period: Jan 2014 to Oct 2019",
        "Frequency: Weekly (7-day)",
        "Total weeks: 302",
        "Drug categories: 8",
    ]
    body = "".join(
        f"<div style='font-size:13px;color:{THEME['text']};line-height:1.9'>{l}</div>"
        for l in lines
    )
    st.markdown(f"""
    <div style="background:{THEME['card_bg']};border:1px solid {THEME['border']};
                border-radius:6px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:{THEME['text_sub']};
                    text-transform:uppercase;margin-bottom:8px">Dataset</div>
        {body}
    </div>
    """, unsafe_allow_html=True)

with c2:
    lines = [
        "Train: Jan 2014 - Sep 2018 (244 wks)",
        "Validation: Sep 2018 - Feb 2019 (21 wks)",
        "Test: Feb 2019 - Jul 2019 (24 wks)",
        "Forecast: Jul 2019 - Sep 2019 (9 wks)",
        "Split method: Chronological only",
    ]
    body = "".join(
        f"<div style='font-size:13px;color:{THEME['text']};line-height:1.9'>{l}</div>"
        for l in lines
    )
    st.markdown(f"""
    <div style="background:{THEME['card_bg']};border:1px solid {THEME['border']};
                border-radius:6px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:{THEME['text_sub']};
                    text-transform:uppercase;margin-bottom:8px">Data Splits</div>
        {body}
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="background:{THEME['card_bg']};border:1px solid {THEME['border']};
                border-radius:6px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:{THEME['text_sub']};
                    text-transform:uppercase;margin-bottom:8px">Models Evaluated</div>
        <div style="font-size:13px;color:{THEME['text']};line-height:1.9">
            Naive Baseline<br>
            Prophet<br>
            XGBoost Local
            <span style="font-size:11px;color:{THEME['text_sub']}">
                (one model per drug)</span><br>
            LightGBM Local
            <span style="font-size:11px;color:{THEME['text_sub']}">
                (one model per drug)</span><br>
            LightGBM Global
            <span style="font-size:11px;color:{THEME['text_sub']}">
                (single model across all drugs)</span>
        </div>
        <div style="margin-top:12px;padding-top:10px;
                    border-top:1px solid {THEME['divider']};
                    font-size:12px;font-weight:700;color:{THEME['positive']}">
            Production model: Global LightGBM
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
section_header("Model Performance -- Test Window Feb-Jul 2019 (MAPE %)")
comp_df = load_csv("model_comparison.csv")
if not comp_df.empty:
    st.dataframe(comp_df.round(1), use_container_width=True)
else:
    st.info("Run python train.py to generate model comparison results.")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
section_header("Drug Selection -- 4 Dashboard Drugs")
selection_data = [
    {"Drug": "Anti-inflammatory (COX)", "Code": "M01AB",
     "Test MAPE": "3.1%", "Archetype": "Stable",
     "Reason": "Lowest MAPE of all drugs. Stable, low volatility (CV 24%)."},
    {"Drug": "Analgesics / Paracetamol", "Code": "N02BE",
     "Test MAPE": "7.0%", "Archetype": "Winter seasonal",
     "Reason": "Highest volume (avg 208 units/wk). Winter flu season peak. Largest saving."},
    {"Drug": "Anxiolytics", "Code": "N05B",
     "Test MAPE": "4.5%", "Archetype": "Stable",
     "Reason": "Consistent accuracy across all models. Low seasonal variation."},
    {"Drug": "Antihistamines", "Code": "R06",
     "Test MAPE": "7.1%", "Archetype": "Spring seasonal",
     "Reason": "3.5x seasonal ratio. Spring allergy peak (Mar-May). Upward trend."},
]
st.dataframe(pd.DataFrame(selection_data), use_container_width=True, hide_index=True)
st.caption(
    "Remaining 4 drugs excluded: high volatility (CV > 50%), intermittent demand, "
    "or unpredictable demand shocks requiring external data."
)
