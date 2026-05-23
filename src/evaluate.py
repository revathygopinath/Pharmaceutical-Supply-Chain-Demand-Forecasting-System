"""
evaluate.py
-----------
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Skips zero actuals."""
    a, p = np.array(actual), np.array(predicted)
    mask = a != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE. Used for Hypnotics (36 zero weeks)."""
    a, p  = np.array(actual), np.array(predicted)
    denom = np.where((np.abs(a) + np.abs(p)) == 0, 1, np.abs(a) + np.abs(p))
    return float(np.mean(2 * np.abs(a - p) / denom) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def evaluate_model(actual, predicted,
                   model_name: str = "",
                   drug_name:  str = "",
                   verbose:    bool = True) -> dict:
    """
    Compute all metrics and return as dict.
    Prints formatted line to terminal if verbose=True.
    """
    actual    = np.array(actual)
    predicted = np.array(predicted)

    mae_v   = float(mean_absolute_error(actual, predicted))
    rmse_v  = rmse(actual, predicted)
    mape_v  = mape(actual, predicted)
    smape_v = smape(actual, predicted)
    r2_v    = float(r2_score(actual, predicted))

    if verbose:
        print(f"  [{model_name}] {drug_name}: "
              f"MAE={mae_v:.2f}  RMSE={rmse_v:.2f}  "
              f"MAPE={mape_v:.1f}%  R2={r2_v:.3f}")

    return {
        "model": model_name, "drug": drug_name,
        "MAE": mae_v, "RMSE": rmse_v,
        "MAPE": mape_v, "sMAPE": smape_v, "R2": r2_v,
    }


def build_mape_table(results: list) -> pd.DataFrame:
    """Build MAPE pivot: rows=drugs, columns=models."""
    df = pd.DataFrame(results)
    return (df.pivot_table(index="drug", columns="model", values="MAPE")
              .round(1))


def get_best_models(mape_pivot: pd.DataFrame) -> dict:
    """Return {drug: best_model_name} based on lowest MAPE."""
    return mape_pivot.idxmin(axis=1).to_dict()
