"""
kpis.py

"""

import numpy as np
import pandas as pd
from src.config import (
    DASHBOARD_DRUGS, NAIVE_MAPE,
    DEFAULT_UNIT_COST, DEFAULT_SAFETY_BUFFER,
    DEFAULT_IMPL_COST, WEEKS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# Executive Summary KPIs
# ---------------------------------------------------------------------------
def calc_model_accuracy(mape_pivot: pd.DataFrame) -> dict:
    """
    Model Accuracy -- test window (Feb-Jul 2019).
    Formula: 100 - mean(LGB_Global MAPE for 4 dashboard drugs)
    """
    if mape_pivot.empty or "LightGBM_Global" not in mape_pivot.columns:
        return {"value": 0.0, "label": "N/A"}

    mapes = [mape_pivot.loc[d, "LightGBM_Global"]
             for d in DASHBOARD_DRUGS if d in mape_pivot.index]
    if not mapes:
        return {"value": 0.0, "label": "N/A"}

    avg  = np.mean(mapes)
    val  = round(100 - avg, 1)
    return {"value": val, "label": f"{val}%",
            "detail": f"Mean MAPE: {avg:.1f}% on test window",
            "benchmark": "Industry standard: 75-85%"}


def calc_forecast_accuracy(holdout_mapes: dict) -> dict:
    """
    Forecast Accuracy -- forecast window (Jul-Sep 2019).
    Formula: 100 - mean(forecast MAPE for 4 dashboard drugs)
    """
    mapes = [holdout_mapes[d]["mape"]
             for d in DASHBOARD_DRUGS if d in holdout_mapes]
    if not mapes:
        return {"value": 0.0, "label": "N/A"}

    avg = np.mean(mapes)
    val = round(100 - avg, 1)
    return {"value": val, "label": f"{val}%",
            "detail": f"Mean MAPE: {avg:.1f}% on forecast window",
            "benchmark": "True holdout Jul-Sep 2019"}


def calc_error_reduction(mape_pivot: pd.DataFrame) -> dict:
    """
    Error Reduction vs Naive -- test window.
    Formula: mean( (naive_mape - lgb_mape) / naive_mape * 100 ) for 4 drugs
    """
    reductions = []
    for d in DASHBOARD_DRUGS:
        if d not in mape_pivot.index:
            continue
        naive_m = NAIVE_MAPE.get(d, 0)
        lgb_m   = mape_pivot.loc[d, "LightGBM_Global"] \
                  if "LightGBM_Global" in mape_pivot.columns else 0
        if naive_m > 0:
            reductions.append((naive_m - lgb_m) / naive_m * 100)

    if not reductions:
        return {"value": 0.0, "label": "N/A"}

    val = round(np.mean(reductions), 1)
    return {"value": val, "label": f"{val}%",
            "detail": "Average improvement over naive baseline",
            "benchmark": "Test window Feb-Jul 2019"}


def calc_overproduction_reduced(df: pd.DataFrame,
                                mape_pivot: pd.DataFrame,
                                safety_buf: float = DEFAULT_SAFETY_BUFFER) -> dict:
    """
    Overproduction Reduced -- test window.
    Formula: (naive_overstock - model_overstock) / naive_overstock * 100
    overstock = avg_sales * (mape/100) * safety_buffer * 52 * unit_cost
    """
    naive_total = 0.0
    model_total = 0.0

    for d in DASHBOARD_DRUGS:
        if d not in df.columns or d not in mape_pivot.index:
            continue
        avg_s  = df[d].mean()
        n_mape = NAIVE_MAPE.get(d, 0)
        m_mape = mape_pivot.loc[d, "LightGBM_Global"] \
                 if "LightGBM_Global" in mape_pivot.columns else 0
        naive_total += avg_s * (n_mape / 100) * safety_buf * 52
        model_total += avg_s * (m_mape / 100) * safety_buf * 52

    if naive_total == 0:
        return {"value": 0.0, "label": "N/A"}

    val = round((naive_total - model_total) / naive_total * 100, 1)
    return {"value": val, "label": f"{val}%",
            "detail": "Reduction in estimated excess inventory",
            "benchmark": f"At {safety_buf*100:.0f}% safety buffer | Test window"}


# ---------------------------------------------------------------------------
# Drug Performance KPI
# ---------------------------------------------------------------------------
def calc_direction_accuracy(df_global: pd.DataFrame,
                             lgb_global,
                             train_end_dt: pd.Timestamp,
                             val_end_dt:   pd.Timestamp,
                             test_end_dt:  pd.Timestamp) -> dict:
    """
    Correct Procurement Signals -- test window.
    Count weeks where forecast direction matches actual direction (up OR down).
    """
    from src.config import GLOBAL_FEATURES

    test_mask = ((df_global.index > val_end_dt) &
                 (df_global.index <= test_end_dt))
    df_test   = df_global[test_mask].copy()
    df_test["prediction"] = np.clip(
        lgb_global.predict(df_global[test_mask][GLOBAL_FEATURES]), 0, None
    )

    total_correct = 0
    total_weeks   = 0
    per_drug      = {}

    for drug in DASHBOARD_DRUGS:
        sub = df_test[df_test["drug_name"] == drug].sort_index()
        if len(sub) < 2:
            continue

        actual_v = sub["sales"].values
        pred_v   = sub["prediction"].values

        correct = sum(
            1 for i in range(1, len(actual_v))
            if (actual_v[i] > actual_v[i-1]) == (pred_v[i] > pred_v[i-1])
        )
        total = len(actual_v) - 1
        total_correct += correct
        total_weeks   += total
        per_drug[drug] = round(correct / total * 100, 1) if total > 0 else 0

    rate = round(total_correct / total_weeks * 100, 1) if total_weeks > 0 else 0
    return {
        "value":    rate,
        "label":    f"{total_correct} of {total_weeks} weeks ({rate:.0f}%)",
        "per_drug": per_drug,
        "detail":   "Weeks where forecast direction matched actual direction",
        "benchmark": "Random baseline: 50%",
    }


# ---------------------------------------------------------------------------
# Financial KPIs
# ---------------------------------------------------------------------------
def calc_annual_saving(df: pd.DataFrame,
                       mape_pivot: pd.DataFrame,
                       unit_cost:     float = DEFAULT_UNIT_COST,
                       currency_rate: float = 1.0) -> pd.DataFrame:
    """
    Annual saving per drug and total.
    Formula matches notebook Phase 12 exactly.
    """
    rows = []
    for d in DASHBOARD_DRUGS:
        if d not in df.columns or d not in mape_pivot.index:
            continue
        naive_m  = NAIVE_MAPE.get(d, 0)
        best_m   = mape_pivot.loc[d].min()
        avg_s    = df[d].mean()
        saving   = avg_s * ((naive_m - best_m) / 100) * unit_cost * WEEKS_PER_YEAR
        rows.append({
            "drug":        d,
            "naive_mape":  naive_m,
            "best_mape":   round(best_m, 1),
            "improvement": round(naive_m - best_m, 1),
            "saving_usd":  round(saving, 0),
            "saving_conv": round(saving * currency_rate, 0),
        })
    return pd.DataFrame(rows)


def calc_revenue_alignment(df_global: pd.DataFrame,
                            lgb_global,
                            val_end_dt:  pd.Timestamp,
                            test_end_dt: pd.Timestamp,
                            threshold:   float = 0.15) -> dict:
    """
    Revenue Alignment -- test window.
    % of weeks where |actual - forecast| / actual < threshold.
    """
    from src.config import GLOBAL_FEATURES

    test_mask = ((df_global.index > val_end_dt) &
                 (df_global.index <= test_end_dt))
    df_test   = df_global[test_mask].copy()
    df_test["prediction"] = np.clip(
        lgb_global.predict(df_global[test_mask][GLOBAL_FEATURES]), 0, None
    )

    aligned = 0
    total   = 0
    for drug in DASHBOARD_DRUGS:
        sub = df_test[df_test["drug_name"] == drug]
        for _, row in sub.iterrows():
            if row["sales"] > 0:
                total += 1
                if abs(row["sales"] - row["prediction"]) / row["sales"] < threshold:
                    aligned += 1

    pct = round(aligned / total * 100, 1) if total > 0 else 0
    return {
        "value":     pct,
        "label":     f"{pct}%",
        "detail":    f"{aligned} of {total} weeks within {int(threshold*100)}% of actual",
        "benchmark": "Target: >70% | Test window Feb-Jul 2019",
    }


def calc_roi(total_saving: float, impl_cost: float = DEFAULT_IMPL_COST) -> dict:
    if impl_cost <= 0:
        return {"value": 0.0, "label": "N/A"}
    val = round(total_saving / impl_cost, 1)
    return {"value": val, "label": f"{val}x",
            "detail": f"Saving / implementation cost"}


# ---------------------------------------------------------------------------
# Scenario KPI
# ---------------------------------------------------------------------------
def calc_decision_agility(df_global: pd.DataFrame,
                           lgb_global,
                           val_end_dt:  pd.Timestamp,
                           test_end_dt: pd.Timestamp) -> dict:
    """
    Decision Agility -- test window.
    Max lag where direction match rate > 60%.
    """
    from src.config import GLOBAL_FEATURES

    test_mask = ((df_global.index > val_end_dt) &
                 (df_global.index <= test_end_dt))
    df_test   = df_global[test_mask].copy()
    df_test["prediction"] = np.clip(
        lgb_global.predict(df_global[test_mask][GLOBAL_FEATURES]), 0, None
    )

    fc_all, ac_all = [], []
    for drug in DASHBOARD_DRUGS:
        sub = df_test[df_test["drug_name"] == drug].sort_index()
        fc_all.extend(sub["prediction"].values.tolist())
        ac_all.extend(sub["sales"].values.tolist())

    best_lag = 0
    lag_rows = []
    for lag in range(1, 5):
        matches = total = 0
        for i in range(lag, len(fc_all)):
            if ac_all[i] == 0:
                continue
            total += 1
            if (fc_all[i] > fc_all[i-1]) == (ac_all[i] > ac_all[i-1]):
                matches += 1
        rate = matches / total * 100 if total > 0 else 0
        lag_rows.append({"lag": lag, "rate": round(rate, 1)})
        if rate >= 60:
            best_lag = lag

    return {
        "value":   best_lag,
        "label":   f"{best_lag} weeks",
        "detail":  f"Model predicts direction correctly {best_lag} weeks ahead",
        "lag_table": pd.DataFrame(lag_rows),
    }
