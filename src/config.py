"""
config.py
---------
Single source of truth for all constants, paths, and hyperparameters.
Changing a value here propagates through the entire pipeline.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
SRC_DIR     = ROOT_DIR / "src"
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
CSV_DIR     = OUTPUTS_DIR / "csv"
MLFLOW_DIR  = ROOT_DIR / ".mlflow"

for _d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, CSV_DIR, MLFLOW_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "salesweekly.csv"

# ---------------------------------------------------------------------------
# MLflow  -- SQLite backend avoids all Windows file-URI issues
# ---------------------------------------------------------------------------
MLFLOW_DB_PATH         = MLFLOW_DIR / "pharmacast.db"
_db_str                = str(MLFLOW_DB_PATH).replace("\\", "/")
MLFLOW_TRACKING_URI    = f"sqlite:///{_db_str}"
MLFLOW_EXPERIMENT_NAME = "PharmaCast"

# ---------------------------------------------------------------------------
# Drug mapping
# ---------------------------------------------------------------------------
DRUG_NAMES = {
    "M01AB": "Anti-inflammatory (COX)",
    "M01AE": "Anti-inflammatory (Propionic)",
    "N02BA": "Analgesics (Salicylic)",
    "N02BE": "Analgesics / Paracetamol",
    "N05B":  "Anxiolytics",
    "N05C":  "Hypnotics and Sedatives",
    "R03":   "Respiratory / Asthma",
    "R06":   "Antihistamines",
}
DRUG_CODES = list(DRUG_NAMES.keys())
ALL_DRUGS  = list(DRUG_NAMES.values())

# 4 drugs selected for dashboard
DASHBOARD_DRUGS = [
    "Anti-inflammatory (COX)",
    "Analgesics / Paracetamol",
    "Anxiolytics",
    "Antihistamines",
]

# ---------------------------------------------------------------------------
# Chronological splits  (never random-split time series)
# ---------------------------------------------------------------------------
TRAIN_END      = "2018-09-02"   # training ends here
VAL_END        = "2019-02-01"   # validation ends here
TEST_END       = "2019-07-14"   # clean test window end
FORECAST_START = "2019-07-14"   # 8-week forecast starts
FORECAST_END   = "2019-09-15"   # 8-week forecast ends (real actuals exist)

# ---------------------------------------------------------------------------
# Feature columns  (17 features -- matches notebook exactly)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "lag_1", "lag_4", "lag_8", "lag_12", "lag_52",
    "roll_mean_4", "roll_mean_12", "roll_std_4",
    "month", "quarter", "week", "year",
    "month_sin", "month_cos",
    "is_peak_season",
    "yoy_growth",
    "peak_lag52_interaction",
]

GLOBAL_FEATURES = FEATURE_COLS + [
    "drug_id", "drug_mean", "drug_std", "drug_cv",
]

TARGET_COL = "sales"

# Seasonal peak months per drug (from EDA heatmap)
SEASONAL_FLAGS = {
    "Antihistamines":          [3, 4, 5],
    "Respiratory / Asthma":    [10, 11, 12, 1],
    "Analgesics / Paracetamol":[10, 11, 12, 1],
    "Analgesics (Salicylic)":  [10, 11, 12, 1],
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, verbosity=0,
)

LGB_LOCAL_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
    random_state=42, verbose=-1,
)

LGB_GLOBAL_PARAMS = dict(
    n_estimators=700, max_depth=6, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    random_state=42, verbose=-1,
)

# ---------------------------------------------------------------------------
# Business / dashboard constants
# ---------------------------------------------------------------------------
FORECAST_WEEKS        = 9
DEFAULT_UNIT_COST     = 50.0
DEFAULT_SAFETY_BUFFER = 0.20
DEFAULT_IMPL_COST     = 10_000.0
WEEKS_PER_YEAR        = 52

MAPE_RETRAIN_THRESHOLD = 20.0
MAPE_WATCH_THRESHOLD   = 12.0

# Naive MAPE from notebook Phase 8a (exact values)
NAIVE_MAPE = {
    "Anti-inflammatory (COX)":   22.8,
    "Analgesics / Paracetamol":  20.5,
    "Anxiolytics":               18.5,
    "Antihistamines":            42.3,
}

# ---------------------------------------------------------------------------
# Dashboard theme  (white bg, dark text, WCAG AA compliant)
# ---------------------------------------------------------------------------
THEME = {
    "bg":         "#FFFFFF",
    "card_bg":    "#F5F7FA",
    "sidebar":    "#1B3A6B",
    "text":       "#1A1A2E",
    "text_sub":   "#4A5568",
    "text_light": "#FFFFFF",
    "border":     "#D1D5DB",
    "divider":    "#E5E7EB",
    "accent":     "#2E6DA4",
    "positive":   "#1A6B3A",
    "warning":    "#B7800A",
    "danger":     "#C0392B",
    "header":     "#1B3A6B",
}

DRUG_COLORS = {
    "Anti-inflammatory (COX)":   "#2E6DA4",
    "Analgesics / Paracetamol":  "#C9622F",
    "Anxiolytics":               "#6B3A8B",
    "Antihistamines":            "#1A6B5A",
    "Anti-inflammatory (Propionic)": "#F4A261",
    "Analgesics (Salicylic)":        "#2D6A4F",
    "Hypnotics and Sedatives":       "#774936",
    "Respiratory / Asthma":          "#1B7A8C",
}

CURRENCIES = {
    "USD ($)":  {"symbol": "$",  "rate": 1.0},
    "INR (Rs)": {"symbol": "Rs", "rate": 83.0},
    "EUR (EU)": {"symbol": "EU", "rate": 0.92},
    "GBP (L)":  {"symbol": "L",  "rate": 0.79},
}
