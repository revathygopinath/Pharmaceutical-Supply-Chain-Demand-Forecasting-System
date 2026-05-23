"""
dashboard/components/utils.py
------------------------------
Shared utilities, caching, CSS theme, and helper functions.
White background, dark text, WCAG AA compliant.
No emojis. Professional pharma aesthetic.
"""

import sys
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR, CSV_DIR, THEME, CURRENCIES


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline artifacts...")
def load_artifacts() -> dict:
    path = MODELS_DIR / "pipeline_artifacts.pkl"
    if not path.exists():
        st.error("No trained models found. Run: python train.py")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_kpis() -> dict:
    df = load_csv("dashboard_kpis.csv")
    if df.empty:
        return {}
    return df.set_index("kpi").to_dict(orient="index")


# ---------------------------------------------------------------------------
# Currency helper
# ---------------------------------------------------------------------------
def get_currency(selection: str) -> tuple:
    c = CURRENCIES.get(selection, CURRENCIES["USD ($)"])
    return c["symbol"], c["rate"]


# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------
def apply_theme():
    st.markdown(f"""
    <style>
    /* ── Global background and text ──────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {THEME["bg"]};
        color: {THEME["text"]};
        font-family: Arial, Helvetica, sans-serif;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {THEME["sidebar"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {THEME["text_light"]} !important;
    }}

    /* ── Main content area ───────────────────────────────────────────────── */
    .main .block-container {{
        padding: 1.5rem 2rem;
        max-width: 1200px;
    }}

    /* ── Fix ALL Streamlit input widgets -- force white bg, dark text ─────── */
    /* number_input, text_input */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 4px !important;
    }}
    /* number_input container */
    [data-testid="stNumberInput"] > div > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 4px !important;
    }}
    /* +/- buttons on number_input */
    [data-testid="stNumberInput"] button {{
        background-color: #F5F7FA !important;
        color: #1A1A2E !important;
        border: 1px solid #D1D5DB !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        background-color: #E5E7EB !important;
    }}

    /* selectbox */
    [data-testid="stSelectbox"] > div > div {{
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
        border: 1px solid #D1D5DB !important;
    }}
    [data-testid="stSelectbox"] span {{
        color: #1A1A2E !important;
    }}

    /* ── Widget text -- dark in main content, white in sidebar ──────────── */
    /* Strategy: set dark globally, then override sidebar back to white.
       This is more reliable than scoping to .main which may not match. */

    /* Step 1: all radio/slider/widget text dark */
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span:not([data-testid]),
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
        color: #1A1A2E !important;
        font-size: 13px !important;
    }}
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] span {{
        color: #1A1A2E !important;
        font-size: 13px !important;
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {{
        color: #1A1A2E !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}
    .stRadio > label p,
    .stSlider > label p,
    .stNumberInput > label p,
    .stSelectbox > label p {{
        color: #1A1A2E !important;
    }}

    /* Step 2: sidebar overrides everything back to white -- highest specificity */
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {{
        color: #FFFFFF !important;
    }}

    /* Multiselect */
    [data-testid="stMultiSelect"] > div > div {{
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
        border: 1px solid #D1D5DB !important;
    }}

    /* ── Plotly axis titles -- force dark via container ──────────────────── */
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly .g-xtitle text,
    .js-plotly-plot .plotly .g-ytitle text {{
        fill: #1A1A2E !important;
        color: #1A1A2E !important;
    }}
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {{
        fill: #1A1A2E !important;
    }}

    /* ── Streamlit's default dark widget override ────────────────────────── */
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="select"],
    .stApp [data-baseweb="base-input"] {{
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
    }}

    /* ── Page title ──────────────────────────────────────────────────────── */
    .ph-title {{
        font-size: 28px;
        font-weight: 700;
        color: {THEME["header"]};
        padding-bottom: 8px;
        border-bottom: 2px solid {THEME["accent"]};
        margin-bottom: 4px;
    }}
    .ph-subtitle {{
        font-size: 13px;
        color: {THEME["text_sub"]};
        margin-bottom: 18px;
    }}

    /* ── Section header ──────────────────────────────────────────────────── */
    .ph-section {{
        font-size: 14px;
        font-weight: 700;
        color: {THEME["header"]};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid {THEME["divider"]};
        padding-bottom: 5px;
        margin: 18px 0 10px 0;
    }}

    /* ── Cards and KPI ───────────────────────────────────────────────────── */
    .ph-card {{
        background-color: {THEME["card_bg"]};
        border: 1px solid {THEME["border"]};
        border-radius: 6px;
        padding: 16px 18px;
        height: 100%;
    }}
    .ph-kpi-label {{
        font-size: 11px;
        font-weight: 600;
        color: {THEME["text_sub"]};
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 6px;
    }}
    .ph-kpi-value {{
        font-size: 28px;
        font-weight: 700;
        color: {THEME["text"]};
        line-height: 1.1;
    }}
    .ph-kpi-detail {{
        font-size: 11px;
        color: {THEME["text_sub"]};
        margin-top: 4px;
    }}
    .ph-kpi-benchmark {{
        font-size: 10px;
        color: {THEME["text_sub"]};
        font-style: italic;
        margin-top: 3px;
    }}

    /* ── Narrative ───────────────────────────────────────────────────────── */
    .ph-narrative {{
        background: {THEME["card_bg"]};
        border: 1px solid {THEME["border"]};
        border-left: 3px solid {THEME["accent"]};
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: {THEME["text"]};
        line-height: 1.7;
        margin: 10px 0;
    }}

    /* ── Alert boxes ─────────────────────────────────────────────────────── */
    .ph-alert-good {{
        background: #F0FDF4;
        border-left: 3px solid {THEME["positive"]};
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 8px;
        font-size: 12px;
        color: {THEME["text"]};
    }}
    .ph-alert-warn {{
        background: #FFFBEB;
        border-left: 3px solid {THEME["warning"]};
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 8px;
        font-size: 12px;
        color: {THEME["text"]};
    }}

    /* ── Badges ──────────────────────────────────────────────────────────── */
    .ph-badge-good {{
        display: inline-block;
        background: #D1FAE5;
        color: {THEME["positive"]};
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
    }}
    .ph-badge-fair {{
        display: inline-block;
        background: #FEF3C7;
        color: {THEME["warning"]};
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
    }}
    .ph-badge-review {{
        display: inline-block;
        background: #FEE2E2;
        color: {THEME["danger"]};
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="ph-title">{title}</div>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ph-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="ph-section">{title}</div>',
                unsafe_allow_html=True)


def kpi_card(label: str, value: str,
             detail: str = "", benchmark: str = "",
             border_color: str = None):
    bc = border_color or THEME["accent"]
    st.markdown(f"""
    <div class="ph-card" style="border-top: 3px solid {bc}">
        <div class="ph-kpi-label">{label}</div>
        <div class="ph-kpi-value">{value}</div>
        {"" if not detail    else f'<div class="ph-kpi-detail">{detail}</div>'}
        {"" if not benchmark else f'<div class="ph-kpi-benchmark">{benchmark}</div>'}
    </div>
    """, unsafe_allow_html=True)


def narrative(text: str):
    st.markdown(f'<div class="ph-narrative">{text}</div>',
                unsafe_allow_html=True)


def badge(status: str) -> str:
    s = status.lower()
    if s == "good":
        return '<span class="ph-badge-good">Good</span>'
    elif s == "fair":
        return '<span class="ph-badge-fair">Fair</span>'
    return '<span class="ph-badge-review">Review</span>'


def direction_label(values) -> str:
    if len(values) < 2:
        return "Stable"
    if values[-1] > values[0] * 1.05:
        return "Rising"
    elif values[-1] < values[0] * 0.95:
        return "Declining"
    return "Stable"
