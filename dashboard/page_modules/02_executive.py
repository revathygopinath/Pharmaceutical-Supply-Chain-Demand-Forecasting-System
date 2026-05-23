"""pages/02_executive.py -- Executive Dashboard"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from dashboard.components.utils import (
    page_header, section_header, narrative,
    load_csv, load_kpis, direction_label, THEME
)
from dashboard.components.charts import demand_history_chart
from src.config import DASHBOARD_DRUGS, NAIVE_MAPE, DEFAULT_SAFETY_BUFFER
from src.data_loader import load_data

page_header(
    "Executive Dashboard",
    "Model performance and supply chain impact -- 4 selected drugs"
)

fc_df   = load_csv("forecast_table.csv")
comp_df = load_csv("model_comparison.csv")
fc_eval = load_csv("forecast_evaluation.csv")
kpis    = load_kpis()
df      = load_data()

# ── Overproduction: recalculate using correct formula ─────────────────────────
# Formula: (naive_overstock - model_overstock) / naive_overstock * 100
# naive_overstock = avg_sales * (naive_mape/100) * safety_buffer * 52
# model_overstock = avg_sales * (lgb_mape/100)   * safety_buffer * 52
# Uses test window LGB_Global MAPE values from model_comparison.csv
def _calc_overprod(df_, comp_df_):
    naive_t = 0.0
    model_t = 0.0
    for drug in DASHBOARD_DRUGS:
        if drug not in df_.columns:
            continue
        avg_s  = df_[drug].mean()
        n_mape = NAIVE_MAPE.get(drug, 0)
        try:
            m_mape = comp_df_.loc[drug, "LightGBM_Global"]
        except Exception:
            try:
                m_mape = float(comp_df_[
                    comp_df_.iloc[:, 0] == drug
                ]["LightGBM_Global"].iloc[0])
            except Exception:
                m_mape = 0
        naive_t += avg_s * (n_mape / 100) * DEFAULT_SAFETY_BUFFER * 52
        model_t += avg_s * (m_mape / 100) * DEFAULT_SAFETY_BUFFER * 52
    if naive_t == 0:
        return 0.0
    return round((naive_t - model_t) / naive_t * 100, 1)

overprod = _calc_overprod(df, comp_df) if not comp_df.empty else 0.0

# ── KPI tile renderer -- fully inline styles (no CSS class dependency) ─────────
def _kpi(col, label, value, detail, border_color):
    with col:
        st.markdown(f"""
        <div style="
            background:#F5F7FA;
            border:1px solid #D1D5DB;
            border-top:3px solid {border_color};
            border-radius:6px;
            padding:18px 16px;
            min-height:110px;
        ">
            <div style="
                font-size:11px;
                font-weight:600;
                color:#4A5568;
                text-transform:uppercase;
                letter-spacing:0.4px;
                margin-bottom:8px;
            ">{label}</div>
            <div style="
                font-size:28px;
                font-weight:700;
                color:#1A1A2E;
                line-height:1.1;
                margin-bottom:6px;
            ">{value}</div>
            <div style="
                font-size:11px;
                color:#4A5568;
            ">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

# ── 4 KPI tiles ───────────────────────────────────────────────────────────────
section_header("Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)

ma_row = kpis.get("model_accuracy",    {})
fa_row = kpis.get("forecast_accuracy", {})
er_row = kpis.get("error_reduction",   {})

_kpi(c1, "Model Accuracy",
     ma_row.get("label", "N/A"),
     ma_row.get("detail", "Test window Feb-Jul 2019"),
     THEME["positive"])

_kpi(c2, "Forecast Accuracy",
     fa_row.get("label", "N/A"),
     "8-week forecast window Jul-Sep 2019",
     THEME["accent"])

_kpi(c3, "Error Reduction vs Naive",
     er_row.get("label", "N/A"),
     "Average improvement across 4 drugs",
     THEME["warning"])

_kpi(c4, "Estimated Inventory Reduction",
     f"{overprod}%",
     "Scenario-based estimate at 20% safety buffer",
     THEME["header"])

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Narrative ─────────────────────────────────────────────────────────────────
ma = ma_row.get("label", "N/A")
fa = fa_row.get("label", "N/A")
er = er_row.get("label", "N/A")
sv = kpis.get("total_saving", {}).get("label", "N/A")

narrative(
    f"The forecasting system achieves <b>{ma}</b> model accuracy on the test window "
    f"(Feb-Jul 2019) and <b>{fa}</b> on the unseen 8-week forecast horizon "
    f"(Jul-Sep 2019). The LightGBM Global model is <b>{er}</b> more accurate "
    f"than the naive baseline, reducing estimated inventory overstock by "
    f"<b>{overprod}%</b> at a 20% safety buffer. "
    f"Across 4 drugs, the system delivers an estimated annual saving of "
    f"<b>${sv.replace('$', '')}</b>."
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Chart + alert feed ────────────────────────────────────────────────────────
col_chart, col_alerts = st.columns([2, 1])

with col_chart:
    section_header("Historical Demand -- All 4 Drugs (2014-2019)")
    if not df.empty:
        st.plotly_chart(
            demand_history_chart(df[DASHBOARD_DRUGS]),
            use_container_width=True
        )
    else:
        st.info("Run python train.py to generate data.")

with col_alerts:
    section_header("Current Forecast Signals")
    if not fc_df.empty:
        for drug in DASHBOARD_DRUGS:
            drug_fc   = fc_df[fc_df["drug_name"] == drug]["forecast"].values
            if len(drug_fc) == 0:
                continue
            direction = direction_label(list(drug_fc))
            avg_fc    = np.mean(drug_fc)
            short     = drug.split("/")[0].split("(")[0].strip()

            # Inline styles -- no CSS class dependency
            if direction == "Rising":
                bg    = "#FFFBEB"
                left  = "#B7800A"
                dir_c = "#B7800A"
                msg   = "Demand increasing — review stock levels"
            elif direction == "Declining":
                bg    = "#F0FDF4"
                left  = "#1A6B3A"
                dir_c = "#1A6B3A"
                msg   = "Demand softening — monitor inventory"
            else:
                bg    = "#F5F7FA"
                left  = "#6B7280"
                dir_c = "#6B7280"
                msg   = "Demand stable — routine monitoring"

            st.markdown(f"""
            <div style="
                background:{bg};
                border-left:3px solid {left};
                padding:10px 12px;
                border-radius:0 4px 4px 0;
                margin-bottom:8px;
            ">
                <div style="font-size:12px;font-weight:600;color:#1A1A2E;margin-bottom:3px;">
                    {short}
                    <span style="font-size:11px;font-weight:400;color:{dir_c};margin-left:6px;">
                        {direction}
                    </span>
                </div>
                <div style="font-size:11px;color:#1A1A2E;">{msg}</div>
                <div style="font-size:10px;color:#4A5568;margin-top:3px;">
                    Avg forecast: {avg_fc:.0f} units / week
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run python train.py to generate forecast data.")

# ── MAPE reference table ──────────────────────────────────────────────────────
section_header("Forecast Accuracy by Drug")
if not comp_df.empty and not fc_eval.empty:
    rows = []
    for drug in DASHBOARD_DRUGS:
        try:
            test_m = comp_df.loc[drug, "LightGBM_Global"]
        except Exception:
            try:
                test_m = float(
                    comp_df[comp_df.iloc[:, 0] == drug]["LightGBM_Global"].iloc[0]
                )
            except Exception:
                test_m = 0
        fc_row = fc_eval[fc_eval["drug"] == drug]
        fc_m   = fc_row["MAPE"].iloc[0] if not fc_row.empty else 0
        status = "Good" if fc_m < 15 else ("Fair" if fc_m < 30 else "Review")
        rows.append({
            "Drug":              drug,
            "Test MAPE (%)":     round(test_m, 1),
            "Forecast MAPE (%)": round(fc_m, 1),
            "Status":            status,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Test MAPE: Feb-Jul 2019 (24 weeks)  |  "
        "Forecast MAPE: Jul-Sep 2019 (9 weeks)    "

    )
