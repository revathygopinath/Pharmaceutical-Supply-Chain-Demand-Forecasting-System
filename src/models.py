"""
models.py
---------


"""

import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    FEATURE_COLS, GLOBAL_FEATURES, TARGET_COL,
    XGB_PARAMS, LGB_LOCAL_PARAMS, LGB_GLOBAL_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
)
from src.evaluate import evaluate_model


def _clean_metric_name(name: str) -> str:
    """
    Remove characters MLflow rejects in metric names.
    MLflow allows: alphanumerics, underscores, dashes, periods, spaces, slashes.
    Drug names contain parentheses () which cause MlflowException.
    """
    return re.sub(r"[^a-zA-Z0-9_\-\. /]", "", name).replace(" ", "_")[:40]


def setup_mlflow(tracking_uri: str):
    """
    Configure MLflow with SQLite backend.
    Must be called after os.environ["MLFLOW_TRACKING_URI"] is already set
    (set at the very top of train.py before any mlflow import).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# Naive Baseline
# ---------------------------------------------------------------------------
def train_naive(train: pd.DataFrame,
                test:  pd.DataFrame) -> tuple:
    """Repeat last training value for all test weeks."""
    results, forecasts = [], {}

    with mlflow.start_run(run_name="Naive_Baseline"):
        for drug in train.columns:
            last_val = float(train[drug].iloc[-1])
            naive_fc = np.full(len(test), last_val)
            res      = evaluate_model(test[drug].values, naive_fc,
                                      "Naive", drug)
            results.append(res)
            forecasts[drug] = naive_fc

        mlflow.log_metric("avg_mape",
                          np.mean([r["MAPE"] for r in results]))

    return results, forecasts


# ---------------------------------------------------------------------------
# XGBoost Local
# ---------------------------------------------------------------------------
def train_xgboost(feat_df_dict: dict,
                  train_end_dt: pd.Timestamp,
                  val_end_dt:   pd.Timestamp,
                  test_end_dt:  pd.Timestamp) -> tuple:
    """One XGBoost model per drug."""
    results, forecasts, models = [], {}, {}

    with mlflow.start_run(run_name="XGBoost_Local"):
        mlflow.log_params(XGB_PARAMS)

        for drug, feat_df in feat_df_dict.items():
            X_tr = feat_df[feat_df.index <= train_end_dt][FEATURE_COLS]
            y_tr = feat_df[feat_df.index <= train_end_dt][TARGET_COL]
            X_te = feat_df[(feat_df.index > val_end_dt) &
                           (feat_df.index <= test_end_dt)][FEATURE_COLS]
            y_te = feat_df[(feat_df.index > val_end_dt) &
                           (feat_df.index <= test_end_dt)][TARGET_COL]

            if len(X_te) == 0:
                continue

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_te), 0, None)

            res = evaluate_model(y_te.values, preds, "XGBoost", drug)
            results.append(res)
            forecasts[drug] = {"forecast": preds, "index": y_te.index,
                               "actual": y_te.values}
            models[drug] = model
            mlflow.log_metric(f"mape_{_clean_metric_name(drug)}", res["MAPE"])

        mlflow.log_metric("avg_mape",
                          np.mean([r["MAPE"] for r in results]))

    return results, forecasts, models


# ---------------------------------------------------------------------------
# LightGBM Local
# ---------------------------------------------------------------------------
def train_lgb_local(feat_df_dict: dict,
                    train_end_dt: pd.Timestamp,
                    val_end_dt:   pd.Timestamp,
                    test_end_dt:  pd.Timestamp) -> tuple:
    """One LightGBM model per drug."""
    results, forecasts, models = [], {}, {}

    with mlflow.start_run(run_name="LightGBM_Local"):
        mlflow.log_params(LGB_LOCAL_PARAMS)

        for drug, feat_df in feat_df_dict.items():
            X_tr = feat_df[feat_df.index <= train_end_dt][FEATURE_COLS]
            y_tr = feat_df[feat_df.index <= train_end_dt][TARGET_COL]
            X_te = feat_df[(feat_df.index > val_end_dt) &
                           (feat_df.index <= test_end_dt)][FEATURE_COLS]
            y_te = feat_df[(feat_df.index > val_end_dt) &
                           (feat_df.index <= test_end_dt)][TARGET_COL]

            if len(X_te) == 0:
                continue

            model = lgb.LGBMRegressor(**LGB_LOCAL_PARAMS)
            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_te), 0, None)

            res = evaluate_model(y_te.values, preds, "LightGBM_Local", drug)
            results.append(res)
            forecasts[drug] = {"forecast": preds, "index": y_te.index,
                               "actual": y_te.values}
            models[drug] = model
            mlflow.log_metric(f"mape_{_clean_metric_name(drug)}", res["MAPE"])

        mlflow.log_metric("avg_mape",
                          np.mean([r["MAPE"] for r in results]))

    return results, forecasts, models


# ---------------------------------------------------------------------------
# LightGBM Global
# ---------------------------------------------------------------------------
def train_lgb_global(df_global:    pd.DataFrame,
                     train_end_dt: pd.Timestamp,
                     val_end_dt:   pd.Timestamp,
                     test_end_dt:  pd.Timestamp) -> tuple:
    """
    One Global LightGBM trained on all 8 drugs simultaneously.
    Pools 1952 training rows vs 244 per drug for local models.
    """
    results, forecasts = [], {}

    train_mask = df_global.index <= train_end_dt
    test_mask  = ((df_global.index > val_end_dt) &
                  (df_global.index <= test_end_dt))

    X_tr = df_global[train_mask][GLOBAL_FEATURES]
    y_tr = df_global[train_mask][TARGET_COL]
    X_te = df_global[test_mask][GLOBAL_FEATURES]

    assert len(X_tr) > 0, "Global train set empty"
    assert len(X_te) > 0, "Global test set empty"

    with mlflow.start_run(run_name="LightGBM_Global"):
        mlflow.log_params(LGB_GLOBAL_PARAMS)
        mlflow.log_param("train_rows", len(X_tr))
        mlflow.log_param("test_rows",  len(X_te))

        model = lgb.LGBMRegressor(**LGB_GLOBAL_PARAMS)
        model.fit(X_tr, y_tr)

        # Feature importance
        fi = dict(zip(GLOBAL_FEATURES,
                      model.feature_importances_.tolist()))
        mlflow.log_dict(fi, "feature_importance.json")

        # Predict and evaluate per drug
        df_test_g = df_global[test_mask].copy()
        df_test_g["prediction"] = np.clip(
            model.predict(X_te), 0, None
        )

        for drug in df_global["drug_name"].unique():
            sub = df_test_g[df_test_g["drug_name"] == drug]
            if len(sub) == 0:
                continue
            res = evaluate_model(sub["sales"].values,
                                 sub["prediction"].values,
                                 "LightGBM_Global", drug)
            results.append(res)
            forecasts[drug] = {
                "forecast": sub["prediction"].values,
                "index":    sub.index,
            }
            mlflow.log_metric(f"mape_{_clean_metric_name(drug)}", res["MAPE"])

        avg_mape = np.mean([r["MAPE"] for r in results])
        mlflow.log_metric("avg_mape", avg_mape)

        # Log model artifact (no model registry -- avoids Windows registry URI bug)
        mlflow.sklearn.log_model(model, artifact_path="lgb_global_model")

    return results, forecasts, model, df_test_g
