"""
train.py
--------
Master pipeline. Run once to train everything.

Usage:
    python train.py

Terminal output:  step-by-step progress, model metrics, KPI values
Outputs:
    models/pipeline_artifacts.pkl
    outputs/csv/model_comparison.csv
    outputs/csv/forecast_table.csv
    outputs/csv/forecast_evaluation.csv
    outputs/csv/business_impact.csv
    outputs/csv/dashboard_kpis.csv
    outputs/plots/*.png  (13 charts)
MLflow:
    .mlflow/pharmacast.db
    View: mlflow ui --backend-store-uri sqlite:///.mlflow/pharmacast.db
"""

# ============================================================
# CRITICAL: set MLflow env var BEFORE any other import
# This is the Windows SQLite fix -- must be first
# ============================================================
import os
import sys
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent
_DB_PATH = _ROOT / ".mlflow" / "pharmacast.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_SQLITE_URI = "sqlite:///" + str(_DB_PATH).replace("\\", "/")

os.environ["MLFLOW_TRACKING_URI"] = _SQLITE_URI
os.environ.pop("MLFLOW_REGISTRY_URI", None)

# Windows UTF-8 fix
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )
# ============================================================

import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow

# src imports (after env is set)
from src.config import (
    TRAIN_END, VAL_END, TEST_END, FORECAST_START, FORECAST_END,
    MODELS_DIR, CSV_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    DASHBOARD_DRUGS, NAIVE_MAPE,
)
from src.data_loader  import load_data, quality_check, get_splits
from src.features     import create_features, build_global_dataframe
from src.models       import (
    setup_mlflow,
    train_naive, train_xgboost, train_lgb_local, train_lgb_global,
)
from src.evaluate     import build_mape_table, get_best_models
from src.forecast      import generate_recursive_forecast, evaluate_forecast_window
from src.walk_forward  import (
    run_walk_forward,
    run_walk_forward_all_dashboard_drugs,
    print_walk_forward_summary,
    get_model_health,
)
from src.kpis         import (
    calc_model_accuracy, calc_forecast_accuracy,
    calc_error_reduction, calc_overproduction_reduced,
    calc_direction_accuracy, calc_annual_saving,
    calc_revenue_alignment, calc_roi, calc_decision_agility,
)
from src.plots        import (
    plot_boxplots, plot_demand_overview, plot_moving_averages,
    plot_seasonality_heatmap, plot_correlation_matrix,
    plot_decompositions, plot_adf_results, plot_data_split,
    plot_mape_heatmap, plot_feature_importance,
    plot_forecast_vs_actual, plot_business_impact,
)

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
W   = 62
SEP = "=" * W
DIV = "-" * W

def hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{DIV}")

def step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}")

def ok(msg: str):
    print(f"  OK   {msg}")

def info(msg: str):
    print(f"  --   {msg}")

def warn(msg: str):
    print(f"  !!   {msg}")

def result(label: str, value: str):
    print(f"  >>   {label:<40} {value}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    t_start = time.time()
    STEPS   = 7

    hdr("PharmaCast Pipeline  --  Starting")
    print(f"  Time : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  MLflow URI : {_SQLITE_URI}")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    setup_mlflow(_SQLITE_URI)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    step(1, STEPS, "Loading and validating data")
    t0 = time.time()

    df = load_data()
    qc = quality_check(df)

    ok(f"Loaded {qc['n_weeks']} weeks x {qc['n_drugs']} drugs  "
       f"({qc['date_start']} to {qc['date_end']})")

    if qc["passed"]:
        ok("All quality checks passed -- no missing values, no gaps")
    else:
        if qc["total_missing"] > 0:
            warn(f"{qc['total_missing']} missing values found")
        if qc["frequency_gaps"] > 0:
            warn(f"{qc['frequency_gaps']} frequency gaps found")

    n05c_zeros = qc["zero_values"].get("Hypnotics and Sedatives", 0)
    if n05c_zeros > 0:
        info(f"Hypnotics and Sedatives: {n05c_zeros} zero weeks")

    info(f"Step 1 complete  ({time.time()-t0:.1f}s)")

    # =========================================================================
    # STEP 2: Feature engineering
    # =========================================================================
    step(2, STEPS, "Feature engineering")
    t0 = time.time()

    feat_df_dict = {d: create_features(df[d], d) for d in df.columns}
    ok(f"Local feature matrices: {len(feat_df_dict)} drugs  (17 features each)")

    df_global, le = build_global_dataframe(df)
    ok(f"Global dataframe: {df_global.shape[0]} rows  "
       f"({df_global.shape[0]//8} weeks x 8 drugs)")
    info(f"Drug encoding: "
         f"{dict(zip(le.classes_, le.transform(le.classes_).tolist()))}")
    info(f"Step 2 complete  ({time.time()-t0:.1f}s)")

    # =========================================================================
    # STEP 3: Data splits
    # =========================================================================
    step(3, STEPS, "Creating chronological data splits")

    train, val, test, forecast_actual = get_splits(
        df, TRAIN_END, VAL_END, TEST_END, FORECAST_START, FORECAST_END
    )
    train_end_dt    = pd.to_datetime(TRAIN_END)
    val_end_dt      = pd.to_datetime(VAL_END)
    test_end_dt     = pd.to_datetime(TEST_END)
    forecast_end_dt = pd.to_datetime(FORECAST_END)

    info(f"Train    : {train.index.min().date()} to "
         f"{train.index.max().date()}  ({len(train)} weeks)")
    info(f"Val      : {val.index.min().date()} to "
         f"{val.index.max().date()}  ({len(val)} weeks)")
    info(f"Test     : {test.index.min().date()} to "
         f"{test.index.max().date()}  ({len(test)} weeks)")
    info(f"Forecast : {forecast_actual.index.min().date()} to "
         f"{forecast_actual.index.max().date()}  "
         f"({len(forecast_actual)} weeks)")

    # =========================================================================
    # STEP 4: Export EDA charts
    # =========================================================================
    step(4, STEPS, "Exporting EDA charts to outputs/plots/")
    t0 = time.time()

    plot_boxplots(df)
    plot_demand_overview(df)
    plot_moving_averages(df)
    plot_seasonality_heatmap(df)
    plot_correlation_matrix(df)
    plot_decompositions(df)
    plot_adf_results(df)
    plot_data_split(df, train, test, forecast_actual)

    ok(f"8 EDA charts saved to outputs/plots/  ({time.time()-t0:.1f}s)")

    # =========================================================================
    # STEP 5: Train models
    # =========================================================================
    step(5, STEPS, "Training models")
    all_results = []

    # Naive
    print(f"\n  Naive Baseline", end=" ... ", flush=True)
    t0 = time.time()
    naive_r, naive_f = train_naive(train, test)
    all_results.extend(naive_r)
    print(f"Done  ({time.time()-t0:.1f}s)")
    for r in naive_r:
        info(f"[Naive] {r['drug']}: MAPE={r['MAPE']:.1f}%")

    # XGBoost
    print(f"\n  XGBoost Local", end=" ... ", flush=True)
    t0 = time.time()
    xgb_r, xgb_f, xgb_models = train_xgboost(
        feat_df_dict, train_end_dt, val_end_dt, test_end_dt
    )
    all_results.extend(xgb_r)
    print(f"Done  ({time.time()-t0:.1f}s)")
    for r in xgb_r:
        info(f"[XGBoost] {r['drug']}: MAPE={r['MAPE']:.1f}%")

    # LightGBM Local
    print(f"\n  LightGBM Local", end=" ... ", flush=True)
    t0 = time.time()
    lgb_lr, lgb_lf, lgb_local_models = train_lgb_local(
        feat_df_dict, train_end_dt, val_end_dt, test_end_dt
    )
    all_results.extend(lgb_lr)
    print(f"Done  ({time.time()-t0:.1f}s)")
    for r in lgb_lr:
        info(f"[LGB_Local] {r['drug']}: MAPE={r['MAPE']:.1f}%")

    # LightGBM Global
    print(f"\n  LightGBM Global", end=" ... ", flush=True)
    t0 = time.time()
    lgb_gr, lgb_gf, lgb_global, df_test_g = train_lgb_global(
        df_global, train_end_dt, val_end_dt, test_end_dt
    )
    all_results.extend(lgb_gr)
    print(f"Done  ({time.time()-t0:.1f}s)")
    for r in lgb_gr:
        info(f"[LGB_Global] {r['drug']}: MAPE={r['MAPE']:.1f}%")

    ok("All models trained and logged to MLflow")

    # =========================================================================
    # STEP 6: Evaluate and compare
    # =========================================================================
    step(6, STEPS, "Evaluating models and generating comparison table")

    mape_pivot  = build_mape_table(all_results)
    best_models = get_best_models(mape_pivot)

    print(f"\n{DIV}")
    print(f"  MAPE TABLE (%) -- Test Window {VAL_END} to {TEST_END}")
    print(f"  Lower = Better\n")
    print(mape_pivot.to_string())
    print(f"\n{DIV}")
    print(f"  BEST MODEL PER DRUG:")
    for drug, model in best_models.items():
        bm = mape_pivot.loc[drug, model]
        nm = NAIVE_MAPE.get(drug, mape_pivot.loc[drug, "Naive"]
                             if "Naive" in mape_pivot.columns else 0)
        imp = ((nm - bm) / nm * 100) if nm > 0 else 0
        print(f"    {drug:<35}: {model:<22} "
              f"MAPE={bm:.1f}%  ({imp:.0f}% better than Naive)")
    print(DIV)

    mape_pivot.to_csv(CSV_DIR / "model_comparison.csv")
    ok("Model comparison saved to outputs/csv/model_comparison.csv")

    # Model charts
    plot_mape_heatmap(mape_pivot)
    plot_feature_importance(lgb_global)
    ok("Model charts saved to outputs/plots/")

    # =========================================================================
    # STEP 7: Forecast, KPIs, save all outputs
    # =========================================================================
    step(7, STEPS, "Generating forecast and computing dashboard KPIs")

    # Recursive forecast
    df_known  = df[df.index <= pd.to_datetime(FORECAST_START)]
    future_df = generate_recursive_forecast(
        df_known=df_known, df_global=df_global,
        best_models=best_models, lgb_global=lgb_global,
        xgb_models=xgb_models, lgb_local_models=lgb_local_models,
        le=le, forecast_end=forecast_end_dt,
    )
    fc_eval = evaluate_forecast_window(future_df, forecast_actual)

    print(f"\n  8-WEEK FORECAST VALIDATION (forecast window Jul-Sep 2019):")
    print(f"  {fc_eval[['drug','model','MAPE','status']].to_string(index=False)}")

    future_df.to_csv(CSV_DIR / "forecast_table.csv", index=False)
    fc_eval.to_csv(CSV_DIR / "forecast_evaluation.csv",  index=False)
    ok("Forecast tables saved to outputs/csv/")

    # Business impact CSV
    impact_df = calc_annual_saving(df, mape_pivot)
    impact_df.to_csv(CSV_DIR / "business_impact.csv", index=False)
    total_saving = impact_df["saving_usd"].sum()

    # Forecast charts
    plot_forecast_vs_actual(df, df_known, future_df,
                            forecast_actual, best_models)
    plot_business_impact(df, mape_pivot)
    ok("Forecast and impact charts saved to outputs/plots/")

    # ── Walk-forward retraining simulation ────────────────────────────────────
    print(f"\n  Walk-Forward Retraining Simulation")
    print(f"  Antihistamines | Retrain every 4 weeks | MAPE thresholds: OK<12%, WATCH 12-20%, RETRAIN>20%")

    wf_results = run_walk_forward_all_dashboard_drugs(
        feat_df_dict, DASHBOARD_DRUGS,
        init_size=200, horizon=4,
    )
    # Print summary for Antihistamines (most seasonal -- best demonstration)
    if "Antihistamines" in wf_results:
        print_walk_forward_summary(wf_results["Antihistamines"], "Antihistamines")

    # Print health status for all 4 dashboard drugs
    print(f"\n  MODEL HEALTH STATUS -- ALL DASHBOARD DRUGS:")
    for drug in DASHBOARD_DRUGS:
        if drug not in wf_results:
            continue
        health = get_model_health(wf_results[drug])
        short  = drug.split("/")[0].split("(")[0].strip()[:25]
        print(f"    {short:<25}: {health['status']:<10} "
              f"avg MAPE={health['avg_mape']}%  trend={health['trend']}")

    ok("Walk-forward results saved to outputs/csv/walk_forward_all.csv")

    # Export walk-forward chart using saved results
    from src.plots import plot_walk_forward_from_df
    if "Antihistamines" in wf_results:
        plot_walk_forward_from_df(wf_results["Antihistamines"], "Antihistamines")
    ok("Walk-forward chart saved to outputs/plots/12_walk_forward_mape.png")

    # ── KPI calculations ──────────────────────────────────────────────────────
    holdout_mapes = {}
    for drug in DASHBOARD_DRUGS:
        row = fc_eval[fc_eval["drug"] == drug]
        holdout_mapes[drug] = {
            "mape":  row["MAPE"].iloc[0] if not row.empty else 0
        }

    kpi_model_acc  = calc_model_accuracy(mape_pivot)
    kpi_fc_acc     = calc_forecast_accuracy(holdout_mapes)
    kpi_err_red    = calc_error_reduction(mape_pivot)
    kpi_overprod   = calc_overproduction_reduced(df, mape_pivot)
    kpi_dir_acc    = calc_direction_accuracy(
        df_global, lgb_global, train_end_dt, val_end_dt, test_end_dt
    )
    kpi_rev_aln    = calc_revenue_alignment(
        df_global, lgb_global, val_end_dt, test_end_dt
    )
    kpi_roi        = calc_roi(total_saving)
    kpi_agility    = calc_decision_agility(
        df_global, lgb_global, val_end_dt, test_end_dt
    )

    # ── Print KPI summary to terminal ─────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  DASHBOARD KPI VALUES")
    print(DIV)
    print(f"\n  EXECUTIVE SUMMARY PAGE")
    result("Model Accuracy (test window)",    kpi_model_acc["label"])
    result("Forecast Accuracy (forecast wnd)", kpi_fc_acc["label"])
    result("Error Reduction vs Naive",        kpi_err_red["label"])
    result("Overproduction Reduced",          kpi_overprod["label"])
    print(f"\n  DRUG PERFORMANCE PAGE")
    result("Direction Accuracy (test window)", kpi_dir_acc["label"])
    for drug, rate in kpi_dir_acc["per_drug"].items():
        info(f"  {drug}: {rate}%")
    print(f"\n  FINANCIAL IMPACT PAGE")
    result("Total Annual Saving (4 drugs)",   f"${total_saving:,.0f}")
    result("ROI vs Baseline",                 kpi_roi["label"])
    result("Revenue Alignment",               kpi_rev_aln["label"])
    print(f"\n  SCENARIO PLANNING PAGE")
    result("Decision Agility",                kpi_agility["label"])
    print(DIV)

    # Save KPI values to CSV for dashboard
    kpi_rows = [
        {"kpi": "model_accuracy",     "value": kpi_model_acc["value"],
         "label": kpi_model_acc["label"],
         "detail": kpi_model_acc.get("detail",""),
         "benchmark": kpi_model_acc.get("benchmark","")},
        {"kpi": "forecast_accuracy",  "value": kpi_fc_acc["value"],
         "label": kpi_fc_acc["label"],
         "detail": kpi_fc_acc.get("detail",""),
         "benchmark": kpi_fc_acc.get("benchmark","")},
        {"kpi": "error_reduction",    "value": kpi_err_red["value"],
         "label": kpi_err_red["label"],
         "detail": kpi_err_red.get("detail",""),
         "benchmark": kpi_err_red.get("benchmark","")},
        {"kpi": "overproduction",     "value": kpi_overprod["value"],
         "label": kpi_overprod["label"],
         "detail": kpi_overprod.get("detail",""),
         "benchmark": kpi_overprod.get("benchmark","")},
        {"kpi": "direction_accuracy", "value": kpi_dir_acc["value"],
         "label": kpi_dir_acc["label"],
         "detail": kpi_dir_acc.get("detail",""),
         "benchmark": kpi_dir_acc.get("benchmark","")},
        {"kpi": "revenue_alignment",  "value": kpi_rev_aln["value"],
         "label": kpi_rev_aln["label"],
         "detail": kpi_rev_aln.get("detail",""),
         "benchmark": kpi_rev_aln.get("benchmark","")},
        {"kpi": "total_saving",       "value": total_saving,
         "label": f"${total_saving:,.0f}", "detail": "4 dashboard drugs",
         "benchmark": ""},
        {"kpi": "roi",                "value": kpi_roi["value"],
         "label": kpi_roi["label"],
         "detail": kpi_roi.get("detail",""), "benchmark": ""},
        {"kpi": "decision_agility",   "value": kpi_agility["value"],
         "label": kpi_agility["label"],
         "detail": kpi_agility.get("detail",""), "benchmark": ""},
    ]
    pd.DataFrame(kpi_rows).to_csv(
        CSV_DIR / "dashboard_kpis.csv", index=False
    )
    ok("Dashboard KPIs saved to outputs/csv/dashboard_kpis.csv")

    # ── Save all artifacts ─────────────────────────────────────────────────────
    artifacts = {
        "lgb_global":        lgb_global,
        "xgb_models":        xgb_models,
        "lgb_local_models":  lgb_local_models,
        "label_encoder":     le,
        "best_models":       best_models,
        "mape_pivot":        mape_pivot,
        "df_global_meta": {
            "drug_means": (df_global.groupby("drug_name")["drug_mean"]
                           .first().to_dict()),
            "drug_stds":  (df_global.groupby("drug_name")["drug_std"]
                           .first().to_dict()),
            "drug_cvs":   (df_global.groupby("drug_name")["drug_cv"]
                           .first().to_dict()),
        },
    }
    art_path = MODELS_DIR / "pipeline_artifacts.pkl"
    with open(art_path, "wb") as f:
        pickle.dump(artifacts, f)
    ok(f"Artifacts saved to models/pipeline_artifacts.pkl")

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  Pipeline Complete  --  {elapsed:.0f}s")
    print(f"  Annual saving estimate : ${total_saving:,.0f}")
    print(f"  Model accuracy         : {kpi_model_acc['label']} (test window)")
    print(f"  Forecast accuracy      : {kpi_fc_acc['label']} (forecast window)")
    print(f"  Charts exported        : outputs/plots/  (13 charts)")
    print(f"  CSVs exported          : outputs/csv/    (5 files)")
    print(f"  MLflow UI              : mlflow ui "
          f"--backend-store-uri {_SQLITE_URI}")
    print(f"  Dashboard              : streamlit run dashboard/app.py")
    print(SEP)

    return artifacts


if __name__ == "__main__":
    run_pipeline()
