"""
walk_forward.py
---------------
Walk-forward retraining simulation -- matches notebook Phase 15 exactly.

What it does:
  Simulates production MLOps -- retrain every `horizon` weeks on growing data.
  At each step:
    - Train on all data up to that point
    - Predict next `horizon` weeks
    - Record MAPE and model health status

Why walk-forward (not simple train/test):
  A static train/test split tells you how the model performs once.
  Walk-forward tells you how the model performs as new data arrives over time
  -- much closer to real production behaviour.

Threshold logic (matches notebook):
  MAPE < 12%   --> OK      (green)
  MAPE 12-20%  --> WATCH   (yellow)
  MAPE > 20%   --> RETRAIN (red)

Output:
  DataFrame with columns: retrain_date, train_weeks, mape, status
  Also saved to outputs/csv/walk_forward_results.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from src.config import FEATURE_COLS, TARGET_COL, CSV_DIR
from src.evaluate import mape as calc_mape


INIT_SIZE          = 200    # minimum weeks before first retrain
HORIZON            = 4      # retrain every 4 weeks (matches notebook)
WATCH_THRESHOLD    = 12.0   # MAPE % -- flag for monitoring
RETRAIN_THRESHOLD  = 20.0   # MAPE % -- flag for retraining

# Lightweight params for walk-forward (fast retraining, not production quality)
WF_PARAMS = dict(
    n_estimators  = 300,
    learning_rate = 0.05,
    verbose       = -1,
    random_state  = 42,
)


def run_walk_forward(
    feat_df_dict: dict,
    drug_name:    str  = "Antihistamines",
    init_size:    int  = INIT_SIZE,
    horizon:      int  = HORIZON,
    save_csv:     bool = True,
) -> pd.DataFrame:
    """
    Run walk-forward retraining simulation for one drug.

    Parameters
    ----------
    feat_df_dict : dict
        Feature DataFrames per drug from create_features()
    drug_name : str
        Drug to run simulation on (default: Antihistamines -- most seasonal)
    init_size : int
        Minimum training weeks before first retrain
    horizon : int
        Number of weeks between retrains
    save_csv : bool
        Whether to save results to outputs/csv/

    Returns
    -------
    pd.DataFrame
        Columns: retrain_date, train_weeks, mape, status
    """
    if drug_name not in feat_df_dict:
        raise ValueError(f"Drug '{drug_name}' not found in feat_df_dict")

    feat_df = feat_df_dict[drug_name]
    rows    = []

    for start in range(init_size, len(feat_df) - horizon, horizon):
        # Expanding training window
        X_tr = feat_df.iloc[:start][FEATURE_COLS]
        y_tr = feat_df.iloc[:start][TARGET_COL]

        # Next horizon weeks = test
        X_te = feat_df.iloc[start : start + horizon][FEATURE_COLS]
        y_te = feat_df.iloc[start : start + horizon][TARGET_COL]

        model = lgb.LGBMRegressor(**WF_PARAMS)
        model.fit(X_tr, y_tr)

        preds  = np.clip(model.predict(X_te), 0, None)
        mape_v = round(calc_mape(y_te.values, preds), 1)
        status = _status_label(mape_v)

        rows.append({
            "retrain_date":  feat_df.index[start].date(),
            "train_weeks":   start,
            "mape":          mape_v,
            "status":        status,
        })

    wf_df = pd.DataFrame(rows)
    wf_df["retrain_date"] = pd.to_datetime(wf_df["retrain_date"])

    if save_csv:
        out_path = CSV_DIR / "walk_forward_results.csv"
        wf_df.to_csv(out_path, index=False)

    return wf_df


def run_walk_forward_all_dashboard_drugs(
    feat_df_dict: dict,
    dashboard_drugs: list,
    init_size: int = INIT_SIZE,
    horizon:   int = HORIZON,
) -> dict:
    """
    Run walk-forward for all 4 dashboard drugs.
    Returns dict: {drug_name: DataFrame}
    Also saves combined CSV to outputs/csv/walk_forward_all.csv
    """
    results = {}
    all_rows = []

    for drug in dashboard_drugs:
        if drug not in feat_df_dict:
            continue
        wf_df = run_walk_forward(
            feat_df_dict, drug,
            init_size=init_size,
            horizon=horizon,
            save_csv=False,   # save combined below
        )
        wf_df["drug"] = drug
        results[drug] = wf_df
        all_rows.append(wf_df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(CSV_DIR / "walk_forward_all.csv", index=False)

    return results


def get_model_health(wf_df: pd.DataFrame,
                     recent_n: int = 5) -> dict:
    """
    Derive current model health from last N walk-forward steps.

    Returns
    -------
    dict with keys:
        status      : "Healthy" | "Watch" | "Retrain"
        avg_mape    : float
        trend       : "Improving" | "Stable" | "Degrading"
        last_mape   : float
        recommendation : str
    """
    if wf_df.empty:
        return {"status": "Unknown", "avg_mape": 0,
                "trend": "Unknown", "last_mape": 0,
                "recommendation": "No walk-forward data available"}

    recent    = wf_df.tail(recent_n)
    avg_mape  = round(recent["mape"].mean(), 1)
    last_mape = round(wf_df["mape"].iloc[-1], 1)

    # Trend: compare first half vs second half of recent window
    mid = len(recent) // 2
    if mid > 0:
        first_half = recent.iloc[:mid]["mape"].mean()
        second_half = recent.iloc[mid:]["mape"].mean()
        if second_half < first_half * 0.95:
            trend = "Improving"
        elif second_half > first_half * 1.05:
            trend = "Degrading"
        else:
            trend = "Stable"
    else:
        trend = "Stable"

    # Status
    if avg_mape < WATCH_THRESHOLD:
        status = "Healthy"
        rec    = "Model is performing within acceptable range. Continue monitoring."
    elif avg_mape < RETRAIN_THRESHOLD:
        status = "Watch"
        rec    = f"MAPE averaging {avg_mape}%. Schedule retraining within 2 weeks."
    else:
        status = "Retrain"
        rec    = f"MAPE averaging {avg_mape}%. Retrain immediately on latest data."

    return {
        "status":         status,
        "avg_mape":       avg_mape,
        "last_mape":      last_mape,
        "trend":          trend,
        "recommendation": rec,
    }


def print_walk_forward_summary(wf_df: pd.DataFrame, drug_name: str = ""):
    """Print formatted summary to terminal -- called from train.py."""
    print(f"\n  Walk-Forward Simulation: {drug_name}")
    print(f"  {'Date':<14} {'Weeks':>8} {'MAPE':>8} {'Status':>12}")
    print(f"  {'-'*46}")
    for _, row in wf_df.iterrows():
        print(f"  {str(row['retrain_date'].date()):<14} "
              f"{int(row['train_weeks']):>8} "
              f"{row['mape']:>7.1f}% "
              f"{row['status']:>12}")
    print(f"\n  Avg MAPE across all retraining rounds: "
          f"{wf_df['mape'].mean():.1f}%")

    health = get_model_health(wf_df)
    print(f"  Current model status: {health['status']} "
          f"(last 5 rounds avg: {health['avg_mape']}%)")
    print(f"  Trend: {health['trend']}")
    print(f"  Recommendation: {health['recommendation']}")


def _status_label(mape_val: float) -> str:
    if mape_val > RETRAIN_THRESHOLD:
        return "RETRAIN"
    elif mape_val > WATCH_THRESHOLD:
        return "WATCH"
    return "OK"
