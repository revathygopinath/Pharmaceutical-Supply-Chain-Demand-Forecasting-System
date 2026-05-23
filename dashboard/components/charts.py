"""
dashboard/components/charts.py
-------------------------------
All Plotly chart functions.
White background. Dark readable text everywhere.

RULE: _layout() defines ONLY plot_bgcolor, paper_bgcolor, font, hovermode.
      Every other key (xaxis, yaxis, legend, margin, title, height, etc.)
      is passed by each chart function directly -- never via _layout --
      to avoid 'dict() got multiple values for keyword argument' errors.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.config import THEME, DRUG_COLORS, DASHBOARD_DRUGS

# ---------------------------------------------------------------------------
# Shared defaults (used as values, NOT spread into _layout)
# ---------------------------------------------------------------------------
_FONT_DARK  = dict(family="Arial, Helvetica, sans-serif",
                   color="#1A1A2E", size=12)

# _AXIS: used for all chart axes
# color="#1A1A2E" at top level sets axis line + tick marks + tick label colour
# tickfont overrides tick label font explicitly
# title_font sets axis title explicitly
_AXIS = dict(
    showgrid        = True,
    gridcolor       = "#E5E7EB",
    linecolor       = "#9CA3AF",
    linewidth       = 1.5,
    showline        = True,
    showticklabels  = True,
    ticks           = "outside",
    tickcolor       = "#4A5568",
    color           = "#1A1A2E",
    tickfont        = dict(size=12, color="#1A1A2E",
                           family="Arial, Helvetica, sans-serif"),
    title_font      = dict(size=13, color="#1A1A2E",
                           family="Arial, Helvetica, sans-serif"),
    title_standoff  = 12,
    automargin      = True,
)
_AXIS_CLEAN = dict(
    showgrid        = False,
    linecolor       = "#9CA3AF",
    linewidth       = 1.5,
    showline        = True,
    showticklabels  = True,
    ticks           = "outside",
    tickcolor       = "#4A5568",
    color           = "#1A1A2E",
    tickfont        = dict(size=12, color="#1A1A2E",
                           family="Arial, Helvetica, sans-serif"),
    title_font      = dict(size=13, color="#1A1A2E",
                           family="Arial, Helvetica, sans-serif"),
    title_standoff  = 12,
    automargin      = True,
)
_LEGEND     = dict(bgcolor="#FFFFFF", bordercolor="#D1D5DB",
                   borderwidth=1,
                   font=dict(size=11, color="#1A1A2E"))


def _base() -> dict:
    """Minimal base -- only keys that NEVER conflict with per-chart overrides."""
    return dict(
        plot_bgcolor  = "#FFFFFF",
        paper_bgcolor = "#FFFFFF",
        font          = _FONT_DARK,
    )


# ---------------------------------------------------------------------------
def forecast_chart(hist_dates, hist_vals,
                   fc_dates, fc_vals,
                   act_dates=None, act_vals=None,
                   drug_name="", mape_val=None,
                   resid_std=None) -> go.Figure:
    color = DRUG_COLORS.get(drug_name, THEME["accent"])
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(hist_dates), y=list(hist_vals),
        mode="lines", name="Historical",
        line=dict(color="#999999", width=1.5, dash="dot"),
    ))

    if resid_std and len(fc_dates) > 0:
        lo = np.clip(np.array(fc_vals) - 1.28 * resid_std, 0, None)
        hi = np.array(fc_vals) + 1.28 * resid_std
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself", fillcolor="rgba(46,109,164,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence", hoverinfo="skip",
        ))

    if len(fc_dates) > 0:
        fig.add_trace(go.Scatter(
            x=list(fc_dates), y=list(fc_vals),
            mode="lines+markers", name="Forecast",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=7, color=color),
        ))

    if act_dates is not None and act_vals is not None:
        fig.add_trace(go.Scatter(
            x=list(act_dates), y=list(act_vals),
            mode="lines+markers", name="Actual",
            line=dict(color="#1A1A2E", width=2.5),
            marker=dict(size=8, symbol="square", color="#1A1A2E"),
        ))

    if len(fc_dates) > 0:
        fig.add_shape(type="line",
            x0=str(fc_dates[0]), x1=str(fc_dates[0]),
            y0=0, y1=1, xref="x", yref="paper",
            line=dict(color=THEME["danger"], width=1.5, dash="dash"))
        fig.add_annotation(
            x=str(fc_dates[0]), y=1, xref="x", yref="paper",
            text="Forecast Start", showarrow=False,
            font=dict(size=10, color=THEME["danger"]),
            yanchor="bottom", bgcolor="#FFFFFF")

    title = f"{drug_name} -- 8-Week Forecast vs Actual"
    if mape_val is not None:
        title += f"   |   MAPE: {mape_val:.1f}%"

    fig.update_layout(
        **_base(),
        title=dict(text=title, font=dict(size=13, color=THEME["header"]), x=0),
        hovermode="x unified", height=370,
        margin=dict(l=55, r=20, t=55, b=45),
        legend=_LEGEND,
        xaxis=dict(**_AXIS, title="Week"),
        yaxis=dict(**_AXIS, title="Units Sold"),
    )
    return fig


# ---------------------------------------------------------------------------
def residual_chart(dates, actuals, forecasts, drug_name="") -> go.Figure:
    residuals = np.array(actuals) - np.array(forecasts)
    colors    = [THEME["positive"] if r >= 0 else THEME["danger"]
                 for r in residuals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(dates), y=list(residuals),
        marker_color=colors,
        text=[f"{r:+.1f}" for r in residuals],
        textposition="outside",
        textfont=dict(size=10, color="#1A1A2E"),
    ))
    fig.add_hline(y=0, line_color="#1A1A2E", line_width=1.5)
    fig.update_layout(
        **_base(),
        title=dict(
            text=f"{drug_name} -- Forecast Residuals (Actual minus Forecast)",
            font=dict(size=13, color=THEME["header"]), x=0),
        height=310, showlegend=False,
        margin=dict(l=55, r=20, t=55, b=45),
        xaxis=dict(**_AXIS, title="Week"),
        yaxis=dict(**_AXIS, title="Residual (units)"),
    )
    return fig


# ---------------------------------------------------------------------------
def mape_heatmap_chart(mape_pivot: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        mape_pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto", text_auto=".1f",
        labels=dict(color="MAPE (%)"),
    )
    fig.update_layout(
        **_base(),
        title=dict(text="Model MAPE Comparison (%) -- Lower is Better",
                   font=dict(size=13, color=THEME["header"]), x=0),
        height=310,
        margin=dict(l=200, r=20, t=55, b=45),
        xaxis=dict(**_AXIS, title="Model"),
        yaxis=dict(**_AXIS, title="Drug"),
        coloraxis_colorbar=dict(title="MAPE %"),
    )
    return fig


# ---------------------------------------------------------------------------
def saving_bar_chart(impact_df: pd.DataFrame, sym: str = "$") -> go.Figure:
    """
    Horizontal bar chart of annual saving by drug.
    - Text inside bars to avoid clipping (Paracetamol $73,228 was cut off)
    - Auto x-axis range with 15% extra space via rangemode
    - Larger right margin for labels
    - Clear x-axis tick labels at 12px dark
    """
    df  = impact_df.copy()
    df["saving_c"] = df["saving_usd"]
    df  = df.sort_values("saving_c")
    clr = [DRUG_COLORS.get(d, THEME["accent"]) for d in df["drug"]]
    df["drug_short"] = df["drug"].apply(
        lambda x: x.split("/")[0].split("(")[0].strip()[:22]
    )

    fig = go.Figure(go.Bar(
        x=df["saving_c"], y=df["drug_short"],
        orientation="h", marker_color=clr,
        text=[f"{sym}{v:,.0f}" for v in df["saving_c"]],
        textposition="inside",          # inside bars -- never clipped
        insidetextanchor="end",         # right-aligned inside each bar
        textfont=dict(size=12, color="#FFFFFF", family="Arial"),
        constraintext="none",           # never shrink text
    ))
    # Extend x-axis range 20% beyond max value so labels have room
    max_val = df["saving_c"].max()
    fig.update_layout(
        **_base(),
        title=dict(
            text=f"Annual Saving Breakdown by Drug ({sym})",
            font=dict(size=13, color=THEME["header"]), x=0),
        height=310, showlegend=False,
        margin=dict(l=190, r=30, t=55, b=55),
        xaxis=dict(
            showgrid=True, gridcolor="#F0F0F0", linecolor="#D1D5DB",
            range=[0, max_val * 1.05],
            tickfont=dict(size=12, color="#1A1A2E"),
            title_font=dict(color="#1A1A2E", size=12),
            title=f"Projected Annual Saving ({sym})",
            tickformat="$,.0f",
        ),
        yaxis=dict(
            showgrid=False, linecolor="#D1D5DB",
            tickfont=dict(size=12, color="#1A1A2E"),
            title_font=dict(color="#1A1A2E", size=12),
            title="",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
def accuracy_vs_saving_scatter(mape_pivot, impact_df,
                                sym="$") -> go.Figure:
    fig = go.Figure()
    for drug in DASHBOARD_DRUGS:
        if drug not in mape_pivot.index:
            continue
        m_val = (mape_pivot.loc[drug, "LightGBM_Global"]
                 if "LightGBM_Global" in mape_pivot.columns else 0)
        row   = impact_df[impact_df["drug"] == drug]
        if row.empty:
            continue
        s_val = row["saving_usd"].iloc[0]
        color = DRUG_COLORS.get(drug, THEME["accent"])
        label = drug.split("/")[0].split("(")[0].strip()[:18]
        fig.add_trace(go.Scatter(
            x=[m_val], y=[s_val],
            mode="markers+text", name=drug,
            marker=dict(size=14, color=color),
            text=[label], textposition="top center",
            textfont=dict(size=10, color="#1A1A2E"),
        ))
    fig.update_layout(
        **_base(),
        title=dict(
            text="Forecast Accuracy vs Annual Saving",
            font=dict(size=13, color=THEME["header"]), x=0),
        height=340, showlegend=False,
        margin=dict(l=55, r=20, t=55, b=45),
        xaxis=dict(**_AXIS, title="MAPE (%) -- Lower is Better"),
        yaxis=dict(**_AXIS, title=f"Annual Saving ({sym})"),
    )
    return fig


# ---------------------------------------------------------------------------
def demand_history_chart(df: pd.DataFrame) -> go.Figure:
    """
    Multi-line demand history for 4 dashboard drugs.
    legend passed directly -- never via _base() -- no duplicate keys.
    """
    fig = go.Figure()
    for drug in DASHBOARD_DRUGS:
        if drug not in df.columns:
            continue
        color = DRUG_COLORS.get(drug, THEME["accent"])
        short = drug.split("/")[0].split("(")[0].strip()
        fig.add_trace(go.Scatter(
            x=df.index, y=df[drug],
            mode="lines", name=short,
            line=dict(color=color, width=1.8), opacity=0.9,
        ))
    fig.update_layout(
        **_base(),
        title=dict(
            text="Weekly Drug Demand -- Historical (2014-2019)",
            font=dict(size=13, color=THEME["header"]), x=0),
        hovermode="x unified", height=350,
        margin=dict(l=55, r=20, t=55, b=45),
        legend=dict(
            bgcolor="#FFFFFF", bordercolor="#D1D5DB", borderwidth=1,
            font=dict(size=11, color="#1A1A2E"), orientation="v",
        ),
        xaxis=dict(**_AXIS, title="Date"),
        yaxis=dict(**_AXIS, title="Units Sold per Week"),
    )
    return fig


# ---------------------------------------------------------------------------
def scenario_chart(dates, base_fc, scenario_fc,
                   label="Scenario", drug_name="") -> go.Figure:
    """
    Blue dashed = base forecast, Red solid = scenario.
    legend passed directly -- never via _base() -- no duplicate keys.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=list(base_fc),
        mode="lines+markers", name="Base Forecast",
        line=dict(color=THEME["accent"], width=2.5, dash="dash"),
        marker=dict(size=7, color=THEME["accent"]),
    ))
    fig.add_trace(go.Scatter(
        x=list(dates), y=list(scenario_fc),
        mode="lines+markers", name=label,
        line=dict(color=THEME["danger"], width=2.5),
        marker=dict(size=7, symbol="diamond", color=THEME["danger"]),
        fill="tonexty", fillcolor="rgba(192,57,43,0.07)",
    ))
    fig.update_layout(
        **_base(),
        title=dict(
            text=f"{drug_name} -- Base Forecast vs {label}",
            font=dict(size=14, color=THEME["header"]), x=0),
        hovermode="x unified", height=380,
        margin=dict(l=55, r=20, t=55, b=45),
        legend=dict(
            bgcolor="#FFFFFF", bordercolor="#D1D5DB", borderwidth=1,
            font=dict(size=12, color="#1A1A2E"),
        ),
        xaxis=dict(**_AXIS, title="Week"),
        yaxis=dict(**_AXIS, title="Units per Week"),
    )
    return fig


# ---------------------------------------------------------------------------
def walk_forward_chart(wf_df: pd.DataFrame, drug_name="") -> go.Figure:
    fig   = go.Figure()
    max_y = max(wf_df["mape"].max() + 5, 30)
    fig.add_hrect(y0=0,  y1=12,    fillcolor="#D1FAE5", opacity=0.4, line_width=0)
    fig.add_hrect(y0=12, y1=20,    fillcolor="#FEF3C7", opacity=0.4, line_width=0)
    fig.add_hrect(y0=20, y1=max_y, fillcolor="#FEE2E2", opacity=0.4, line_width=0)
    for y, lbl, c in [(12, "Watch (12%)", THEME["warning"]),
                      (20, "Retrain (20%)", THEME["danger"])]:
        fig.add_hline(y=y, line_dash="dot", line_color=c, line_width=1.2,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font=dict(size=10, color=c))
    pt_colors = [THEME["positive"] if v < 12
                 else (THEME["warning"] if v < 20 else THEME["danger"])
                 for v in wf_df["mape"]]
    fig.add_trace(go.Scatter(
        x=wf_df["retrain_date"].astype(str), y=wf_df["mape"],
        mode="lines+markers", name="MAPE",
        line=dict(color=THEME["accent"], width=2.5),
        marker=dict(size=8, color=pt_colors),
    ))
    fig.update_layout(
        **_base(),
        title=dict(text=f"Walk-Forward Retraining MAPE -- {drug_name}",
                   font=dict(size=13, color=THEME["header"]), x=0),
        yaxis_range=[0, max_y], height=330,
        margin=dict(l=55, r=110, t=55, b=60),
        legend=_LEGEND,
        xaxis=dict(**_AXIS, title="Retrain Date"),
        yaxis=dict(**_AXIS, title="MAPE (%)"),
    )
    return fig
