"""
plots.py
--------

"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import lightgbm as lgb

from src.config import (
    PLOTS_DIR, DASHBOARD_DRUGS, DRUG_COLORS,
    FEATURE_COLS, GLOBAL_FEATURES, TARGET_COL,
    TRAIN_END, VAL_END, TEST_END, FORECAST_START,
)

# Consistent style
plt.rcParams.update({
    "figure.figsize":    (14, 5),
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
})
sns.set_theme(style="whitegrid")

COLORS_8 = ["steelblue","tomato","seagreen","darkorange",
             "purple","brown","teal","crimson"]


def _save(name: str):
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close("all")
    return path


# ---------------------------------------------------------------------------
# EDA charts
# ---------------------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, drug in enumerate(df.columns):
        axes[i].boxplot(df[drug].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6))
        axes[i].set_title(drug, fontsize=9, fontweight="bold")
        axes[i].set_xticks([])
    plt.suptitle("Distribution of Weekly Sales per Drug",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("01_eda_boxplots.png")


def plot_demand_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    for i, drug in enumerate(df.columns):
        axes[i].plot(df.index, df[drug],
                     color=COLORS_8[i], linewidth=1.2, alpha=0.8)
        axes[i].set_title(drug, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Units Sold")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.suptitle("Weekly Drug Sales 2014-2019 -- All 8 Drugs",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save("02_eda_demand_overview.png")


def plot_moving_averages(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    for i, drug in enumerate(DASHBOARD_DRUGS):
        axes[i].plot(df.index, df[drug],
                     alpha=0.4, color="gray", label="Actual")
        axes[i].plot(df.index, df[drug].rolling(4).mean(),
                     color="steelblue", lw=2, label="4-week MA")
        axes[i].plot(df.index, df[drug].rolling(12).mean(),
                     color="tomato", lw=2, label="12-week MA")
        axes[i].set_title(drug, fontweight="bold")
        axes[i].set_ylabel("Units Sold")
        axes[i].legend(fontsize=8)
    plt.suptitle("Moving Average Trend -- 4 Dashboard Drugs",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("03_eda_moving_averages.png")


def plot_seasonality_heatmap(df: pd.DataFrame):
    df_m = df.copy()
    df_m["Month"] = df_m.index.month
    monthly_avg  = df_m.groupby("Month")[df.columns.tolist()].mean()
    monthly_norm = ((monthly_avg - monthly_avg.min()) /
                    (monthly_avg.max() - monthly_avg.min()))
    monthly_norm.index = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
    plt.figure(figsize=(16, 5))
    sns.heatmap(monthly_norm.T, annot=True, fmt=".2f",
                cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Normalised Demand (0=low, 1=peak)"})
    plt.title("Seasonal Demand Heatmap -- All 8 Drugs (Normalised)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("04_eda_seasonality_heatmap.png")


def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Cross-Drug Correlation Matrix",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("05_eda_correlation_matrix.png")


def plot_decompositions(df: pd.DataFrame):
    paths = []
    for drug in DASHBOARD_DRUGS:
        result = seasonal_decompose(df[drug], model="additive",
                                    period=52, extrapolate_trend="freq")
        fig, axes = plt.subplots(4, 1, figsize=(16, 10))
        result.observed.plot( ax=axes[0], color="steelblue")
        axes[0].set_ylabel("Observed")
        result.trend.plot(    ax=axes[1], color="tomato")
        axes[1].set_ylabel("Trend")
        result.seasonal.plot( ax=axes[2], color="seagreen")
        axes[2].set_ylabel("Seasonality")
        result.resid.plot(    ax=axes[3], color="gray")
        axes[3].set_ylabel("Residual")
        plt.suptitle(f"Seasonal Decomposition: {drug}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe = drug.replace("/","").replace(" ","_") \
                   .replace("(","").replace(")","")
        paths.append(_save(f"06_decomposition_{safe}.png"))
    return paths


def plot_adf_results(df: pd.DataFrame):
    drugs, pvals = [], []
    for drug in df.columns:
        _, p, *_ = adfuller(df[drug].dropna())
        drugs.append(drug)
        pvals.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["steelblue" if p < 0.05 else "tomato" for p in pvals]
    ax.bar(drugs, pvals, color=colors, alpha=0.8)
    ax.axhline(0.05, color="red", ls="--", lw=1.5,
               label="Significance threshold (0.05)")
    ax.set_title("ADF Stationarity Test -- p-values",
                 fontweight="bold")
    ax.set_ylabel("p-value")
    ax.legend()
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    return _save("07_adf_stationarity.png")


def plot_data_split(df: pd.DataFrame, train, test, forecast_actual):
    plt.figure(figsize=(16, 4))
    plt.plot(train.index, train["Antihistamines"],
             color="steelblue", lw=2, label="Train")
    plt.plot(test.index, test["Antihistamines"],
             color="seagreen", lw=2, label="Test")
    plt.plot(forecast_actual.index, forecast_actual["Antihistamines"],
             color="tomato", lw=2, label="Forecast Window")
    for xv, c in [(pd.to_datetime(TRAIN_END), "steelblue"),
                  (pd.to_datetime(TEST_END),   "seagreen"),
                  (pd.to_datetime(FORECAST_START), "tomato")]:
        plt.axvline(xv, color=c, ls="--", alpha=0.6, lw=1.2)
    plt.title("Train / Test / Forecast Split -- Antihistamines",
              fontweight="bold")
    plt.legend()
    plt.tight_layout()
    return _save("08_data_split.png")


# ---------------------------------------------------------------------------
# Model charts
# ---------------------------------------------------------------------------
def plot_mape_heatmap(mape_pivot: pd.DataFrame):
    plt.figure(figsize=(14, 6))
    sns.heatmap(mape_pivot, annot=True, fmt=".1f",
                cmap="RdYlGn_r", linewidths=0.5,
                cbar_kws={"label": "MAPE (%) -- Lower = Better"})
    plt.title("Model MAPE Comparison -- All Drugs x All Models\n"
              "Test Window Feb-Jul 2019",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("09_model_comparison_heatmap.png")


def plot_feature_importance(lgb_global):
    fi = (pd.DataFrame({"feature": GLOBAL_FEATURES,
                         "importance": lgb_global.feature_importances_})
            .sort_values("importance", ascending=True))
    plt.figure(figsize=(10, 7))
    plt.barh(fi["feature"], fi["importance"],
             color="steelblue", alpha=0.75)
    plt.title("LightGBM Global -- Feature Importance",
              fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    return _save("10_feature_importance.png")


def plot_forecast_vs_actual(df, df_known, future_df,
                             forecast_actual, best_models):
    colors_4 = {
        "Anti-inflammatory (COX)":   "steelblue",
        "Analgesics / Paracetamol":  "darkorange",
        "Anxiolytics":               "purple",
        "Antihistamines":            "crimson",
    }
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    for i, drug in enumerate(DASHBOARD_DRUGS):
        color   = colors_4.get(drug, "steelblue")
        drug_fc = (future_df[future_df["drug_name"] == drug]
                   .set_index("date")["forecast"])
        actuals = (forecast_actual[drug]
                   if drug in forecast_actual.columns
                   else pd.Series(dtype=float))
        hist    = df_known[drug].iloc[-16:]

        axes[i].plot(hist.index, hist.values,
                     color="gray", alpha=0.5, lw=1.5,
                     label="History (last 16 wks)")
        if len(drug_fc) > 0:
            axes[i].plot(drug_fc.index, drug_fc.values,
                         color=color, lw=2.5, ls="dashed",
                         marker="o", ms=6,
                         label=f"Forecast ({best_models.get(drug,'')})")
        if not actuals.empty:
            axes[i].plot(actuals.index, actuals.values,
                         color="black", lw=2.5, marker="s",
                         ms=6, label="Actual")
        if len(drug_fc) > 0:
            fc0 = drug_fc.index[0]
            axes[i].axvline(fc0, color="red", ls="--",
                            lw=1.2, alpha=0.7)

        # MAPE badge
        if not actuals.empty and len(drug_fc) > 0:
            common = drug_fc.index.intersection(actuals.index)
            if len(common) > 0:
                fc_v = drug_fc.loc[common].values
                ac_v = actuals.loc[common].values
                mask = ac_v != 0
                if mask.sum() > 0:
                    m = (np.mean(np.abs((ac_v[mask]-fc_v[mask])
                                        /ac_v[mask])) * 100)
                    axes[i].text(0.02, 0.95,
                                 f"MAPE: {m:.1f}%",
                                 transform=axes[i].transAxes,
                                 fontsize=9, fontweight="bold",
                                 va="top",
                                 bbox=dict(boxstyle="round",
                                           facecolor="lightyellow",
                                           alpha=0.8))

        axes[i].set_title(drug, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Units Sold")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[i].tick_params(axis="x", rotation=20)
        axes[i].legend(fontsize=8)

    plt.suptitle("8-Week Recursive Forecast vs Real Actuals\n"
                 "Forecast Window Jul-Sep 2019",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save("11_forecast_vs_actual.png")


def plot_walk_forward(df, feat_df_dict):
    drug_wf   = "Antihistamines"
    feat_wf   = feat_df_dict[drug_wf]
    init_size = 200
    horizon   = 4
    rows      = []

    for start in range(init_size, len(feat_wf) - horizon, horizon):
        X_tr = feat_wf.iloc[:start][FEATURE_COLS]
        y_tr = feat_wf.iloc[:start][TARGET_COL]
        X_te = feat_wf.iloc[start:start+horizon][FEATURE_COLS]
        y_te = feat_wf.iloc[start:start+horizon][TARGET_COL]
        m    = lgb.LGBMRegressor(n_estimators=300,
                                  learning_rate=0.05,
                                  verbose=-1, random_state=42)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_te), 0, None)
        mask  = y_te.values != 0
        if mask.sum() > 0:
            mv = (np.mean(np.abs((y_te.values[mask]-preds[mask])
                                  /y_te.values[mask])) * 100)
        else:
            mv = 0.0
        rows.append({"date": feat_wf.index[start], "mape": round(mv, 1)})

    wf_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(0,  12, alpha=0.12, color="green")
    ax.axhspan(12, 20, alpha=0.12, color="orange")
    ax.axhspan(20, max(wf_df["mape"].max()+5, 30),
               alpha=0.12, color="red")
    ax.axhline(12, color="green", ls="--", lw=1.2,
               label="Watch threshold (12%)")
    ax.axhline(20, color="red",   ls="--", lw=1.2,
               label="Retrain threshold (20%)")

    point_colors = [
        "green" if v < 12 else ("orange" if v < 20 else "red")
        for v in wf_df["mape"]
    ]
    ax.plot(wf_df["date"], wf_df["mape"],
            color="steelblue", lw=2.5, zorder=2)
    ax.scatter(wf_df["date"], wf_df["mape"],
               c=point_colors, s=50, zorder=3)
    ax.set_title("Walk-Forward Retraining MAPE -- Antihistamines\n"
                 "Retrain every 4 weeks",
                 fontweight="bold")
    ax.set_ylabel("MAPE (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=20)
    plt.tight_layout()
    return _save("12_walk_forward_mape.png")


def plot_business_impact(df, mape_pivot):
    from src.config import NAIVE_MAPE, DEFAULT_UNIT_COST, WEEKS_PER_YEAR
    rows = []
    for d in DASHBOARD_DRUGS:
        if d not in mape_pivot.index:
            continue
        naive_m = NAIVE_MAPE.get(d, 0)
        best_m  = mape_pivot.loc[d].min()
        saving  = df[d].mean() * ((naive_m-best_m)/100) * DEFAULT_UNIT_COST * WEEKS_PER_YEAR
        rows.append({"drug": d.split("/")[0].strip()[:22],
                     "saving": saving})
    imp = pd.DataFrame(rows).sort_values("saving")

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [DRUG_COLORS.get(d, "steelblue")
                  for d in [r for r in DASHBOARD_DRUGS
                             if r.split("/")[0].strip()[:22] in imp["drug"].values]]
    ax.barh(imp["drug"], imp["saving"],
            color=["steelblue","darkorange","purple","crimson"][:len(imp)],
            alpha=0.8)
    for i, (_, row) in enumerate(imp.iterrows()):
        ax.text(row["saving"] + 200, i,
                f"${row['saving']:,.0f}", va="center", fontsize=9)
    ax.set_title("Estimated Annual Saving by Drug (USD)\n"
                 "Formula: avg_sales x MAPE_improvement x $50 x 52",
                 fontweight="bold")
    ax.set_xlabel("USD")
    plt.tight_layout()
    return _save("13_business_impact.png")


def plot_walk_forward_from_df(wf_df: pd.DataFrame, drug_name: str = "") -> Path:
    """
    Plot walk-forward MAPE from a pre-computed DataFrame.
    Called from train.py after walk_forward module runs.
    Replaces the old plot_walk_forward() which recomputed from scratch.
    """
    import pandas as _pd
    fig, ax = plt.subplots(figsize=(14, 5))

    max_y = max(wf_df["mape"].max() + 5, 30)
    ax.axhspan(0,  12,    alpha=0.12, color="green")
    ax.axhspan(12, 20,    alpha=0.12, color="orange")
    ax.axhspan(20, max_y, alpha=0.12, color="red")

    ax.axhline(12, color="green", ls="--", lw=1.2,
               label="Watch threshold (12%)")
    ax.axhline(20, color="red",   ls="--", lw=1.2,
               label="Retrain threshold (20%)")

    point_colors = [
        "green" if v < 12 else ("orange" if v < 20 else "red")
        for v in wf_df["mape"]
    ]
    ax.plot(wf_df["retrain_date"], wf_df["mape"],
            color="steelblue", lw=2.5, zorder=2)
    ax.scatter(wf_df["retrain_date"], wf_df["mape"],
               c=point_colors, s=50, zorder=3)

    ax.set_title(f"Walk-Forward Retraining MAPE -- {drug_name}\n"
                 "Simulated monthly retraining | Model health monitoring",
                 fontweight="bold")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Retrain Date")
    ax.set_ylim(0, max_y)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=20)
    plt.tight_layout()
    return _save("12_walk_forward_mape.png")
