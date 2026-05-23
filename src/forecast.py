"""
forecast.py
-----------

"""

import numpy as np
import pandas as pd
from src.config import (
    FEATURE_COLS, GLOBAL_FEATURES, TARGET_COL,
    SEASONAL_FLAGS, FORECAST_WEEKS,
)
from src.evaluate import mape


def _build_row(hist: list, future_date: pd.Timestamp,
               drug_name: str, drug_id: int,
               drug_mean: float, drug_std: float,
               drug_cv: float) -> dict:
    """Build one feature row for one recursive forecast step."""
    m        = future_date.month
    peak_m   = SEASONAL_FLAGS.get(drug_name, [])
    is_peak  = 1 if m in peak_m else 0

    lag1  = hist[-1]
    lag4  = hist[-4]  if len(hist) >= 4  else np.mean(hist)
    lag8  = hist[-8]  if len(hist) >= 8  else np.mean(hist)
    lag12 = hist[-12] if len(hist) >= 12 else np.mean(hist)
    lag52 = hist[-52] if len(hist) >= 52 else np.mean(hist)

    r4mean  = np.mean(hist[-4:])  if len(hist) >= 4  else np.mean(hist)
    r12mean = np.mean(hist[-12:]) if len(hist) >= 12 else np.mean(hist)
    r4std   = float(np.std(hist[-4:]))  if len(hist) >= 4  else 0.0

    quarter = (m - 1) // 3 + 1
    week    = int(future_date.isocalendar()[1])
    year    = future_date.year

    return {
        "lag_1": lag1, "lag_4": lag4, "lag_8": lag8,
        "lag_12": lag12, "lag_52": lag52,
        "roll_mean_4": r4mean, "roll_mean_12": r12mean,
        "roll_std_4": r4std,
        "month": m, "quarter": quarter,
        "week": week, "year": year,
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "is_peak_season": is_peak,
        "yoy_growth": (lag1 - lag52) / (lag52 + 1),
        "peak_lag52_interaction": lag52 * is_peak,
        "drug_id": drug_id, "drug_mean": drug_mean,
        "drug_std": drug_std, "drug_cv": drug_cv,
    }


def generate_recursive_forecast(df_known:         pd.DataFrame,
                                 df_global:        pd.DataFrame,
                                 best_models:      dict,
                                 lgb_global,
                                 xgb_models:       dict,
                                 lgb_local_models: dict,
                                 le,
                                 forecast_end:     pd.Timestamp) -> pd.DataFrame:
    """
    Recursive n-week forecast for all drugs.
    Uses best model per drug from MAPE comparison.
    """
    future_dates = pd.date_range(
        start=df_known.index.max() + pd.Timedelta("7D"),
        end=forecast_end,
        freq="W",
    )

    all_preds = []

    for drug_name in df_known.columns:
        best = best_models.get(drug_name, "LightGBM_Global")

        if best == "LightGBM_Global":
            model, use_global = lgb_global, True
        elif best == "XGBoost":
            model, use_global = xgb_models.get(drug_name, lgb_global), False
        else:
            model, use_global = lgb_local_models.get(drug_name, lgb_global), False

        drug_id  = int(le.transform([drug_name])[0])
        sub      = df_global[df_global["drug_name"] == drug_name]
        d_mean   = float(sub["drug_mean"].iloc[0])
        d_std    = float(sub["drug_std"].iloc[0])
        d_cv     = float(sub["drug_cv"].iloc[0])

        hist = list(df_known[drug_name].values)

        for fd in future_dates:
            row    = _build_row(hist, fd, drug_name,
                                drug_id, d_mean, d_std, d_cv)
            row_df = pd.DataFrame([row])
            feat   = GLOBAL_FEATURES if use_global else FEATURE_COLS
            pred   = float(np.clip(
                model.predict(row_df[feat])[0], 0, None
            ))
            pred   = round(pred, 1)

            all_preds.append({
                "date":       fd,
                "drug_name":  drug_name,
                "forecast":   pred,
                "model_used": best,
            })
            hist.append(pred)   # KEY: makes forecast recursive

    return pd.DataFrame(all_preds)


def evaluate_forecast_window(future_df:       pd.DataFrame,
                              forecast_actual: pd.DataFrame) -> pd.DataFrame:
    """
    Compare recursive forecast against real actuals.
    Returns DataFrame with MAPE per drug.
    """
    rows = []
    for drug in forecast_actual.columns:
        fc = future_df[future_df["drug_name"] == drug].set_index("date")["forecast"]
        ac = forecast_actual[drug]
        common = fc.index.intersection(ac.index)
        if len(common) == 0:
            continue
        fm     = mape(ac.loc[common].values, fc.loc[common].values)
        status = "Good" if fm < 15 else ("Fair" if fm < 30 else "Review")
        rows.append({
            "drug":      drug,
            "model":     future_df[future_df["drug_name"] == drug]["model_used"].iloc[0],
            "MAPE":      round(fm, 1),
            "n_weeks":   len(common),
            "status":    status,
        })
    return pd.DataFrame(rows)
