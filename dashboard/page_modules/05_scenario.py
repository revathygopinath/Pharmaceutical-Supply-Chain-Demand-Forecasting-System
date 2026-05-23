"""pages/05_scenario.py -- Scenario Simulator"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from dashboard.components.utils import (
    page_header, section_header,
    load_csv, load_kpis, THEME
)
from dashboard.components.charts import scenario_chart
from src.config import DASHBOARD_DRUGS, DEFAULT_UNIT_COST

page_header(
    "Scenario Simulator",
    "Demand surge and drop simulations for procurement risk management"
)

fc_df = load_csv("forecast_table.csv")
kpis  = load_kpis()

# ── KPI tile helper -- fully inline ───────────────────────────────────────────
def _kpi(label, value, detail, border_color):
    st.markdown(f"""
    <div style="
        background:#F5F7FA;
        border:1px solid #D1D5DB;
        border-top:3px solid {border_color};
        border-radius:6px;
        padding:18px 16px;
        min-height:105px;
    ">
        <div style="font-size:11px;font-weight:600;color:#4A5568;
                    text-transform:uppercase;letter-spacing:0.4px;
                    margin-bottom:8px">{label}</div>
        <div style="font-size:22px;font-weight:700;color:#1A1A2E;
                    line-height:1.2;margin-bottom:6px">{value}</div>
        <div style="font-size:11px;color:#4A5568">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Simulation controls ───────────────────────────────────────────────────────
section_header("Simulation Controls")
ctrl_col, chart_col = st.columns([1, 2])

with ctrl_col:
    sim_drug  = st.selectbox("Select drug", DASHBOARD_DRUGS, key="sim_drug")
    sim_type  = st.radio(
        "Scenario type",
        ["Demand Surge", "Demand Drop"],
        key="sim_type",
    )
    if sim_type == "Demand Surge":
        change_pct = st.slider("Surge magnitude (%)", 10, 100, 30, step=10,
                               help="How much demand increases above base forecast")
    else:
        change_pct = -st.slider("Drop magnitude (%)", 10, 60, 20, step=10,
                                help="How much demand falls below base forecast")

    duration  = st.slider("Duration (weeks)", 1, 8, 4,
                          help="How many weeks the scenario persists")
    lead_time = st.slider("Procurement lead time (weeks)", 1, 6, 3,
                          help="Weeks between order and delivery")
    unit_cost = st.number_input(
        "Unit cost ($)", min_value=1,
        value=int(DEFAULT_UNIT_COST), step=5,
    )
    run_btn = st.button("Run Simulation", type="primary")

with chart_col:
    if not fc_df.empty and sim_drug in fc_df["drug_name"].values:
        base_fc = (
            fc_df[fc_df["drug_name"] == sim_drug]
            .set_index("date")["forecast"]
            .sort_index()
        )
        scen_fc = base_fc.copy()
        for i in range(min(duration, len(scen_fc))):
            scen_fc.iloc[i] = base_fc.iloc[i] * (1 + change_pct / 100)

        label = f"{'Surge' if change_pct > 0 else 'Drop'} ({change_pct:+d}%)"
        fig   = scenario_chart(
            base_fc.index, base_fc.values,
            scen_fc.values, label, sim_drug
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chart caption with colour explanation
        st.markdown("""
        <div style="font-size:12px;color:#4A5568;
                    border-left:3px solid #D1D5DB;
                    padding:6px 12px;margin-top:4px;
                    background:#F9FAFB;border-radius:0 4px 4px 0">
            <b style="color:#2E6DA4">Blue dashed</b> = baseline forecast
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b style="color:#C0392B">Red solid</b> = simulated demand scenario
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Scenario applies only during selected duration.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run python train.py to generate forecast data.")

# ── Simulation results ────────────────────────────────────────────────────────
if run_btn and not fc_df.empty and sim_drug in fc_df["drug_name"].values:
    base_fc = (
        fc_df[fc_df["drug_name"] == sim_drug]
        .set_index("date")["forecast"]
        .sort_index()
    )
    scen_fc = base_fc.copy()
    for i in range(min(duration, len(scen_fc))):
        scen_fc.iloc[i] = base_fc.iloc[i] * (1 + change_pct / 100)

    delta_units = float((scen_fc - base_fc).sum())
    delta_cost  = abs(delta_units) * unit_cost
    order_wk    = max(1, lead_time - 1)
    is_drop     = change_pct < 0

    # Risk level with plain English explanation
    if abs(change_pct) >= 50:
        risk       = "High"
        risk_desc  = "Significant procurement impact"
        risk_color = THEME["danger"]
    elif abs(change_pct) >= 25:
        risk       = "Medium"
        risk_desc  = "Inventory action needed"
        risk_color = THEME["warning"]
    else:
        risk       = "Low"
        risk_desc  = "Minor adjustment"
        risk_color = THEME["positive"]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Section title -- larger font, professional ─────────────────────────────
    st.markdown(f"""
    <div style="font-size:17px;font-weight:700;color:{THEME['header']};
                margin:16px 0 12px 0;padding-bottom:6px;
                border-bottom:2px solid {THEME['accent']}">
        Simulation Results
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)

    if is_drop:
        with sc1:
            _kpi("Reduced Units",
                 f"{abs(delta_units):,.0f} units",
                 "Units below base forecast",
                 THEME["accent"])
        with sc2:
            _kpi("Inventory Saving",
                 f"${delta_cost:,.0f}",
                 "Avoided procurement cost",
                 THEME["positive"])
    else:
        with sc1:
            _kpi("Additional Units Needed",
                 f"{delta_units:+,.0f} units",
                 "Above base forecast",
                 THEME["accent"])
        with sc2:
            _kpi("Additional Cost",
                 f"${delta_cost:,.0f}",
                 "At current unit cost",
                 THEME["warning"])

    with sc3:
        _kpi("Order Action By",
             f"Week {order_wk}",
             f"Based on {lead_time}-week lead time",
             THEME["header"])

    with sc4:
        _kpi(f"Risk: {risk}",
             risk_desc,
             f"Magnitude: {abs(change_pct)}% over {duration} wks",
             risk_color)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Week-by-week table title -- larger font ────────────────────────────────
    st.markdown(f"""
    <div style="font-size:15px;font-weight:700;color:{THEME['header']};
                margin:12px 0 8px 0;padding-bottom:5px;
                border-bottom:1px solid {THEME['divider']}">
        Week-by-Week Simulation
    </div>
    """, unsafe_allow_html=True)

    sim_rows = []
    for d in base_fc.index:
        b    = float(base_fc.loc[d])
        s    = float(scen_fc.loc[d])
        diff = s - b
        # Column label changes for surge vs drop
        if is_drop:
            cost_label = "Avoided Inventory Cost ($)"
        else:
            cost_label = "Extra Cost ($)"

        sim_rows.append({
            "Week":          str(d.date()),
            "Base Forecast": round(b, 1),
            "Scenario":      round(s, 1),
            "Difference":    f"{diff:+.1f}",
            cost_label:      f"${abs(diff * unit_cost):,.0f}",
        })

    st.dataframe(
        pd.DataFrame(sim_rows),
        use_container_width=True,
        hide_index=True,
    )

    # Risk legend -- plain English
    st.markdown(f"""
    <div style="background:#F5F7FA;border:1px solid #D1D5DB;
                border-radius:6px;padding:12px 16px;margin-top:12px;
                font-size:12px;color:#1A1A2E">
        <b>Risk Level Guide</b><br>
        <span style="color:{THEME['positive']}">Low</span>
        = Minor adjustment required &nbsp;|&nbsp;
        <span style="color:{THEME['warning']}">Medium</span>
        = Inventory action needed &nbsp;|&nbsp;
        <span style="color:{THEME['danger']}">High</span>
        = Significant procurement impact
    </div>
    """, unsafe_allow_html=True)
