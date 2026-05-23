"""
data_loader.py
--------------
Load raw CSV, parse dates, run quality checks, return clean DataFrame.
Matches notebook Phase 2-3 exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_FILE, DRUG_NAMES


def load_data(filepath: Path = DATA_FILE) -> pd.DataFrame:
    """Load salesweekly.csv and return clean DataFrame with DatetimeIndex."""
    df = pd.read_csv(filepath, encoding="utf-8")
    df["datum"] = pd.to_datetime(df["datum"])
    df = df.set_index("datum").sort_index()
    df.columns = [DRUG_NAMES[c] for c in df.columns]
    return df


def quality_check(df: pd.DataFrame) -> dict:
    """
    Run 4 data quality checks.
    Returns dict with results for terminal logging.
    """
    missing   = df.isnull().sum()
    zeros     = (df == 0).sum()
    diffs     = df.index.to_series().diff().dropna()
    gaps      = diffs[diffs != pd.Timedelta("7 days")]

    return {
        "total_missing":    int(missing.sum()),
        "missing_per_drug": missing.to_dict(),
        "duplicate_dates":  int(df.index.duplicated().sum()),
        "frequency_gaps":   len(gaps),
        "zero_values":      zeros.to_dict(),
        "n_weeks":          len(df),
        "n_drugs":          len(df.columns),
        "date_start":       df.index.min().date(),
        "date_end":         df.index.max().date(),
        "passed":           (missing.sum() == 0 and
                             df.index.duplicated().sum() == 0 and
                             len(gaps) == 0),
    }


def get_splits(df: pd.DataFrame,
               train_end: str, val_end: str,
               test_end: str, forecast_start: str,
               forecast_end: str):
    """
    Create chronological train/val/test/forecast splits.
    Never use random splitting on time series data.
    Returns: train, val, test, forecast_actual
    """
    te  = pd.to_datetime(train_end)
    ve  = pd.to_datetime(val_end)
    tse = pd.to_datetime(test_end)
    fse = pd.to_datetime(forecast_start)
    fee = pd.to_datetime(forecast_end)

    train           = df[df.index <= te]
    val             = df[(df.index > te)  & (df.index <= ve)]
    test            = df[(df.index > ve)  & (df.index <= tse)]
    forecast_actual = df[(df.index > fse) & (df.index <= fee)]

    return train, val, test, forecast_actual
