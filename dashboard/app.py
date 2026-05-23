"""
dashboard/app.py
----------------
Pharmaceutical Supply Chain Demand Forecasting System.

IMPORTANT -- Folder naming:
  Page files are in dashboard/page_modules/ (NOT pages/).
  Streamlit only auto-scans a folder named exactly "pages".
  Using "page_modules" completely suppresses the auto sidebar list.

UTF-8: env var only -- do NOT wrap sys.stdout (causes ValueError on reload).
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Pharmaceutical Supply Chain Demand Forecasting System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.components.utils import apply_theme
apply_theme()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="
        padding:16px 0 18px 0;
        border-bottom:1px solid rgba(255,255,255,0.2);
        margin-bottom:14px;
    ">
        <div style="
            font-size:13px;
            font-weight:700;
            color:white;
            letter-spacing:0.3px;
            line-height:1.5;
        ">
            Pharmaceutical Supply Chain<br>Demand Forecasting System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:rgba(255,255,255,0.55);font-size:11px;"
        "text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px'>"
        "Navigation</p>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "Overview",
            "Executive Dashboard",
            "Demand Performance",
            "Business Impact",
            "Scenario Simulator",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        "<p style='color:rgba(255,255,255,0.35);font-size:10px;"
        "text-align:center;margin-top:24px;line-height:1.6'>"
        "LightGBM Global | 4 Drugs<br>2014-2019 | 8-Week Horizon</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing -- reads from page_modules/ (Streamlit never auto-scans this)
# ---------------------------------------------------------------------------
_DIR = Path(__file__).parent / "page_modules"
_MAP = {
    "Overview":             "01_summary.py",
    "Executive Dashboard":  "02_executive.py",
    "Demand Performance":   "03_drug_performance.py",
    "Business Impact":      "04_financial.py",
    "Scenario Simulator":   "05_scenario.py",
}

_path = _DIR / _MAP[page]
with open(_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_path), "exec"),
         {"__file__": str(_path)})
