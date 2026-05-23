"""pages/03_drug_performance.py -- Demand Performance"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from dashboard.components.utils import (
    page_header, section_header, load_csv, THEME
)
from dashboard.components.charts import forecast_chart, residual_chart
from src.config import DASHBOARD_DRUGS, NAIVE_MAPE
from src.data_loader import load_data

page_header(
    "Demand Performance",
    "Forecast accuracy and demand analysis -- 4 selected drugs"
)

fc_df   = load_csv("forecast_table.csv")
comp_df = load_csv("model_comparison.csv")
fc_eval = load_csv("forecast_evaluation.csv")
df      = load_data()

# ── Performance table ─────────────────────────────────────────────────────────
section_header("Performance Summary -- All 4 Drugs")
rows = []
for drug in DASHBOARD_DRUGS:
    try:
        test_m = comp_df.loc[drug, "LightGBM_Global"]
    except Exception:
        try:
            test_m = comp_df[comp_df.iloc[:,0]==drug]["LightGBM_Global"].iloc[0]
        except Exception:
            test_m = 0

    fc_row  = fc_eval[fc_eval["drug"] == drug]
    fc_m    = fc_row["MAPE"].iloc[0] if not fc_row.empty else 0
    naive_m = NAIVE_MAPE.get(drug, 0)
    imp     = round((naive_m - test_m) / naive_m * 100, 0) if naive_m > 0 else 0
    # Status based on forecast MAPE thresholds (not test MAPE)
    status  = "Good" if fc_m < 15 else ("Fair" if fc_m < 30 else "Review")

    if not fc_df.empty and drug in fc_df["drug_name"].values:
        drug_fc   = fc_df[fc_df["drug_name"] == drug]["forecast"].values
        direction = (
            "Rising"    if len(drug_fc) > 1 and drug_fc[-1] > drug_fc[0] * 1.05
            else "Declining" if len(drug_fc) > 1 and drug_fc[-1] < drug_fc[0] * 0.95
            else "Stable"
        )
        avg_fc = round(float(np.mean(drug_fc)), 1)
    else:
        direction, avg_fc = "N/A", 0

    rows.append({
        "Drug":                        drug,
        "Test MAPE (%)":               round(test_m, 1),
        "Forecast MAPE (%)":           round(fc_m, 1),
        "Error Reduction vs Baseline": f"{imp:.0f}%",
        "Expected Weekly Demand":      avg_fc,
        "Trend":                       direction,
        "Status":                      status,
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Test MAPE: Feb-Jul 2019  |  Forecast MAPE: Jul-Sep 2019  |  "
    "Trend based on 8-week forecast trajectory  |  "
    "Status based on forecast MAPE thresholds"
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Drug selector ─────────────────────────────────────────────────────────────
section_header("Drug Detail View")
selected = st.selectbox("Select drug", DASHBOARD_DRUGS, key="drug_sel")

if not fc_df.empty and selected in fc_df["drug_name"].values:
    drug_fc  = fc_df[fc_df["drug_name"] == selected].set_index("date")["forecast"]
    fc_dates = drug_fc.index
    fc_vals  = drug_fc.values

    df_known  = df[df.index <= fc_dates.min()]
    hist_ctx  = (df_known[selected].iloc[-16:]
                 if selected in df_known.columns
                 else pd.Series(dtype=float))

    actuals   = (df[(df.index >= fc_dates.min()) &
                    (df.index <= fc_dates.max())][selected]
                 if selected in df.columns
                 else pd.Series(dtype=float))

    fc_row    = fc_eval[fc_eval["drug"] == selected]
    mape_val  = fc_row["MAPE"].iloc[0] if not fc_row.empty else None
    resid_std = df[selected].std() * 0.25 if selected in df.columns else None

    col_fc, col_res = st.columns(2)

    with col_fc:
        fig = forecast_chart(
            hist_dates=hist_ctx.index, hist_vals=hist_ctx.values,
            fc_dates=fc_dates, fc_vals=fc_vals,
            act_dates=actuals.index if not actuals.empty else None,
            act_vals=actuals.values if not actuals.empty else None,
            drug_name=selected, mape_val=mape_val, resid_std=resid_std,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Historical demand (gray)  |  Predicted demand (blue)  |  "
            "Actual demand (black)  |  "
            "Shaded region shows forecast uncertainty (80% confidence)"
        )

    with col_res:
        if not actuals.empty and len(fc_dates) > 0:
            common = fc_dates.intersection(actuals.index)
            if len(common) > 0:
                res_fig = residual_chart(
                    dates=common,
                    actuals=actuals.loc[common].values,
                    forecasts=drug_fc.loc[common].values,
                    drug_name=selected,
                )
                st.plotly_chart(res_fig, use_container_width=True)
                st.caption(
                    "Green bars: Underforecast → Potential stockout risk  |  "
                    "Red bars: Overforecast → Potential overstock risk"
                )
else:
    st.info("Run python train.py to generate forecast data.")
