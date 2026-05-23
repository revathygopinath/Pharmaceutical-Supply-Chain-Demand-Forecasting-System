"""
features.py
-----------

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import (
    FEATURE_COLS, GLOBAL_FEATURES, TARGET_COL,
    SEASONAL_FLAGS, DRUG_CODES,
)


def create_features(series: pd.Series, drug_name: str) -> pd.DataFrame:
    """
    Build feature matrix for one drug.
    Matches notebook create_features() exactly -- 17 feature columns.
    """
    data = pd.DataFrame({"sales": series})

    # Lag features
    data["lag_1"]  = data["sales"].shift(1)
    data["lag_4"]  = data["sales"].shift(4)
    data["lag_8"]  = data["sales"].shift(8)
    data["lag_12"] = data["sales"].shift(12)
    data["lag_52"] = data["sales"].shift(52)

    # Rolling statistics
    data["roll_mean_4"]  = data["sales"].shift(1).rolling(4,  min_periods=1).mean()
    data["roll_mean_12"] = data["sales"].shift(1).rolling(12, min_periods=1).mean()
    data["roll_std_4"]   = (data["sales"].shift(1)
                            .rolling(4, min_periods=1).std().fillna(0))

    # Calendar features
    data["month"]   = data.index.month
    data["quarter"] = data.index.quarter
    data["week"]    = data.index.isocalendar().week.astype(int)
    data["year"]    = data.index.year

    # Cyclical encoding
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    # Fill lag_52 NaNs before creating interaction
    data["lag_52"] = data["lag_52"].fillna(data["roll_mean_12"])

    # Peak season flag
    peak_months = SEASONAL_FLAGS.get(drug_name, [])
    data["is_peak_season"] = data["month"].apply(
        lambda m: 1 if m in peak_months else 0
    )

    # Year-over-year growth
    data["yoy_growth"] = (
        (data["sales"] - data["lag_52"]) / (data["lag_52"].abs() + 1)
    )

    # Interaction
    data["peak_lag52_interaction"] = data["lag_52"] * data["is_peak_season"]

    # Fill remaining NaNs
    data = data.bfill().ffill()

    return data


def build_global_dataframe(df: pd.DataFrame):
    """
    Reshape from wide (8 drugs x 302 weeks) to long format for Global LightGBM.
    Preserves DatetimeIndex through merge -- matches notebook Phase 6 exactly.
    Returns: (df_global, label_encoder)
    """
    global_frames = []
    for drug_code, drug_name in zip(DRUG_CODES, df.columns):
        drug_df = create_features(df[drug_name], drug_name).copy()
        drug_df["drug_name"] = drug_name
        drug_df["drug_code"] = drug_code
        global_frames.append(drug_df)

    df_global = pd.concat(global_frames)

    # Label encode
    le = LabelEncoder()
    df_global["drug_id"] = le.fit_transform(df_global["drug_name"])

    # Per-drug stats -- compute BEFORE merge
    drug_stats = (
        df_global.groupby("drug_name")["sales"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "drug_mean", "std": "drug_std"})
    )

    # Reset index BEFORE merge (pd.merge always drops index)
    df_global = df_global.reset_index()
    if "index" in df_global.columns:
        df_global = df_global.rename(columns={"index": "datum"})

    df_global = df_global.merge(drug_stats, on="drug_name", how="left")
    df_global["drug_cv"] = df_global["drug_std"] / df_global["drug_mean"]

    # Restore DatetimeIndex AFTER merge
    df_global["datum"] = pd.to_datetime(df_global["datum"])
    df_global = df_global.set_index("datum").sort_index()

    assert isinstance(df_global.index, pd.DatetimeIndex), \
        "Index lost after merge -- check build_global_dataframe()"
    assert df_global.index.min().year == 2014, "Wrong start year"

    return df_global, le
