"""pages/04_financial.py -- Business Impact"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from dashboard.components.utils import (
    page_header, section_header,
    load_csv, load_kpis, THEME
)
from dashboard.components.charts import saving_bar_chart
from src.config import (
    DASHBOARD_DRUGS, DEFAULT_UNIT_COST,
    DEFAULT_IMPL_COST, NAIVE_MAPE
)
from src.data_loader import load_data

page_header(
    "Business Impact",
    "Cost savings from improved forecast accuracy -- 4 selected drugs"
)

comp_df = load_csv("model_comparison.csv")
kpis    = load_kpis()
df      = load_data()

# ── Inputs ────────────────────────────────────────────────────────────────────
section_header("Assumptions")
ca, _ = st.columns([1, 3])
with ca:
    unit_cost = st.number_input(
        "Drug unit cost ($)", min_value=1,
        value=int(DEFAULT_UNIT_COST), step=5,
    )
    impl_cost = st.number_input(
        "Implementation cost ($)", min_value=1000,
        value=int(DEFAULT_IMPL_COST), step=1000,
    )

# ── Recalculate saving ────────────────────────────────────────────────────────
# Projected Annual Saving =
# (Forecast Error Reduction x Annual Drug Volume x Unit Cost x Inventory Impact Factor)
# + Operational Efficiency Gain
# Simplified: avg_sales * (naive_mape - best_mape)/100 * unit_cost * 52
# Inventory Impact Factor = 1.0 (base conservative estimate)
# Operational Efficiency Gain = 0 (not included -- conservative)
total_saving = 0.0
impact_rows  = []
if not comp_df.empty:
    for drug in DASHBOARD_DRUGS:
        try:
            best_m = float(comp_df.loc[drug].min())
        except Exception:
            try:
                best_m = float(
                    comp_df[comp_df.iloc[:, 0] == drug].iloc[0, 1:].min()
                )
            except Exception:
                best_m = 0
        naive_m = NAIVE_MAPE.get(drug, 0)
        avg_s   = df[drug].mean() if drug in df.columns else 0
        saving  = avg_s * ((naive_m - best_m) / 100) * unit_cost * 52
        total_saving += saving
        impact_rows.append({
            "drug":        drug,
            "naive_mape":  naive_m,
            "best_mape":   round(best_m, 1),
            "improvement": round(naive_m - best_m, 1),
            "saving_usd":  round(saving, 0),
        })

roi     = round(total_saving / impl_cost, 1) if impl_cost > 0 else 0
rev_aln = kpis.get("revenue_alignment", {})

# ── KPI tile helper -- fully inline ───────────────────────────────────────────
def _kpi(label, value, detail, detail2="", border_color=None):
    bc = border_color or THEME["accent"]
    d2 = (f"<div style='font-size:10px;color:#4A5568;margin-top:3px'>{detail2}</div>"
          if detail2 else "")
    st.markdown(f"""
    <div style="
        background:#F5F7FA;
        border:1px solid #D1D5DB;
        border-top:3px solid {bc};
        border-radius:6px;
        padding:18px 16px;
        min-height:120px;
    ">
        <div style="font-size:11px;font-weight:600;color:#4A5568;
                    text-transform:uppercase;letter-spacing:0.4px;
                    margin-bottom:8px">{label}</div>
        <div style="font-size:26px;font-weight:700;color:#1A1A2E;
                    line-height:1.1;margin-bottom:6px">{value}</div>
        <div style="font-size:11px;color:#4A5568">{detail}</div>
        {d2}
    </div>
    """, unsafe_allow_html=True)

# ── 4 KPI tiles ───────────────────────────────────────────────────────────────
section_header("Financial KPIs")
c1, c2, c3, c4 = st.columns(4)

with c1:
    _kpi("Projected Annual Saving",
         f"${total_saving:,.0f}",
         "4 dashboard drugs",
         border_color=THEME["positive"])

with c2:
    _kpi("Estimated ROI",
         f"{roi}x",
         f"At ${impl_cost:,.0f} implementation cost",
         border_color=THEME["accent"])

with c3:
    aln_val    = rev_aln.get("value", 0)
    raw_detail = rev_aln.get("detail", "")
    # Clean up detail text
    clean_det  = (raw_detail
                  .replace("weeks within 15%", "observations within +/-15%")
                  .replace("of actual", "of actual demand")
                  if raw_detail else "Test window: Feb-Jul 2019")
    _kpi("Forecast Alignment",
         f"{aln_val}%",
         clean_det,
         detail2="Business target: >70%  |  Test window: Feb-Jul 2019",
         border_color=THEME["warning"])

with c4:
    total_units = sum(
        df[d].mean() * 52 for d in DASHBOARD_DRUGS if d in df.columns
    )
    cpu = round(total_saving / total_units, 2) if total_units > 0 else 0
    _kpi("Estimated Saving per Unit",
         f"${cpu:.2f}",
         "Per unit sold annually",
         border_color=THEME["header"])

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Annual saving bar chart only (scatter removed) ────────────────────────────
imp_df = pd.DataFrame(impact_rows)
if not imp_df.empty:
    section_header("Annual Saving Breakdown")
    st.plotly_chart(
        saving_bar_chart(imp_df, "$"),
        use_container_width=True
    )

# ── Saving table ──────────────────────────────────────────────────────────────
section_header("Detailed Saving Table")
if impact_rows:
    disp = pd.DataFrame(impact_rows)
    disp["Projected Annual Saving ($)"] = disp["saving_usd"].apply(
        lambda x: f"${x:,.0f}"
    )
    disp["% of Total"] = (
        disp["saving_usd"] / (disp["saving_usd"].sum() + 1e-9) * 100
    ).round(1).astype(str) + "%"
    st.dataframe(
        disp[["drug", "naive_mape", "best_mape", "improvement",
              "Projected Annual Saving ($)", "% of Total"]],
        use_container_width=True, hide_index=True
    )
    st.markdown(
        f"<div style='text-align:right;font-size:13px;font-weight:700;"
        f"color:{THEME['positive']};padding:8px 0'>"
        f"Total: ${total_saving:,.0f}</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Projected Annual Saving = (Forecast Error Reduction x Annual Drug Volume "
        f"x ${unit_cost}/unit x Inventory Impact Factor) + Operational Efficiency Gain  |  "
        "Best MAPE from test window Feb-Jul 2019"
    )
