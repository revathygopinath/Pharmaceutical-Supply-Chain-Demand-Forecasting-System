# Pharmaceutical Supply Chain Demand Forecasting System

> Time Series Forecasting | LightGBM | XGBoost | MLflow | Streamlit | Python

An end-to-end machine learning pipeline that forecasts weekly pharmaceutical drug demand using five years of historical sales data (2014–2019). The system solves a critical supply chain problem — predicting drug demand 8 weeks ahead to prevent stockouts, reduce excess inventory, and support data-driven procurement decisions.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

> Replace the URL above after deploying to Streamlit Cloud

---

## Business Problem

Pharmaceutical procurement teams face two costly problems:

- **Stockouts** — running out of stock before the next delivery, causing patient harm and lost revenue
- **Overstock** — ordering too much based on gut feel, tying up capital in unused inventory

This system addresses both by replacing reactive procurement with 8-week forward-looking demand forecasts. The dashboard surfaces rising demand signals in advance, quantifies the financial cost of forecast errors, and lets procurement teams simulate surge/drop scenarios before they happen.

**Where this is implemented in the system:**
- Rising demand signals → Executive Dashboard forecast alerts (advance stockout warning)
- Overstock reduction → 72% estimated inventory reduction at 20% safety buffer (Business Impact page)
- Scenario planning → Demand surge/drop simulator with week-by-week cost impact (Scenario Simulator page)
- Model health monitoring → Walk-forward retraining simulation flags degraded accuracy before it affects procurement

---

## Key Results

| Metric | Value |
|---|---|
| Model Accuracy (test window Feb–Jul 2019) | 94.6% |
| Forecast Accuracy (holdout Jul–Sep 2019) | 86.6% |
| Error Reduction vs Naive Baseline | ~78% |
| Estimated Inventory Reduction | ~72% at 20% safety buffer |
| Projected Annual Saving | $132,191 across 4 drugs |
| ROI (at $10,000 implementation cost) | 13.2x |

---

## Model Comparison — MAPE (%) on Test Window Feb–Jul 2019

| Drug | Naive | XGBoost | LGB Local | LGB Global |
|---|---|---|---|---|
| Anti-inflammatory (COX) | 22.8% | 5.8% | 7.2% | **3.1%** |
| Analgesics / Paracetamol | 20.5% | 8.0% | 7.4% | **7.0%** |
| Anxiolytics | 18.5% | 4.6% | 5.4% | **4.5%** |
| Antihistamines | 42.3% | 12.2% | 11.5% | **7.1%** |

LightGBM Global wins on all 4 dashboard drugs and on 7 of 8 total drugs.

---

## Forecast Validation — True Holdout Jul–Sep 2019

| Drug | Forecast MAPE | Status |
|---|---|---|
| Anti-inflammatory (COX) | 9.5% | Good |
| Anxiolytics | 11.5% | Good |
| Analgesics / Paracetamol | 15.1% | Fair |
| Antihistamines | 17.5% | Fair |

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| **Overview** | Project summary, dataset details, model comparison table, drug selection rationale |
| **Executive Dashboard** | 4 KPI tiles, historical demand chart (2014–2019), current forecast signals |
| **Demand Performance** | 8-week forecast vs actual, residual analysis, per-drug accuracy breakdown |
| **Business Impact** | Projected annual saving, ROI, forecast alignment, saving breakdown by drug |
| **Scenario Simulator** | Demand surge/drop simulation, procurement risk assessment, week-by-week impact |

---

## Screenshots

### Dashboard

<!-- Save browser screenshots to docs/screenshots/ and update paths below -->

| Executive Dashboard | Demand Performance |
|---|---|
| ![Executive Dashboard](docs/screenshots/executive_dashboard.png) | ![Demand Performance](docs/screenshots/demand_performance.png) |

| Business Impact | Scenario Simulator |
|---|---|
| ![Business Impact](docs/screenshots/business_impact.png) | ![Scenario Simulator](docs/screenshots/scenario_simulator.png) |

### Model Outputs

**Model Comparison Heatmap**
![Model Comparison](outputs/plots/09_model_comparison_heatmap.png)

**8-Week Forecast vs Real Actuals**
![Forecast vs Actual](outputs/plots/11_forecast_vs_actual.png)

### MLflow Experiment Tracking

**All 4 model runs compared side by side — average MAPE and per-drug metrics**
![MLflow Comparison](docs/screenshots/mlflow_comparison.png)

**LightGBM Global run detail — parameters, per-drug MAPE metrics, artifacts**
![MLflow Run Detail](docs/screenshots/mlflow_run_detail.png)

**Model artifacts — feature importance JSON and saved model**
![MLflow Artifacts](docs/screenshots/mlflow_artifacts.png)

---

## MLOps — Experiment Tracking with MLflow

Every training run is automatically logged to MLflow (SQLite backend) with full reproducibility:

```
Experiment: PharmaCast
  Run: Naive_Baseline     avg_mape=32.99%
  Run: XGBoost_Local      avg_mape=9.18%   n_estimators=500, lr=0.03
  Run: LightGBM_Local     avg_mape=9.96%   n_estimators=500, lr=0.03
  Run: LightGBM_Global    avg_mape=8.40%   n_estimators=700, lr=0.02
    Metrics:   mape_Anti-inflammatory_COX=3.1,  mape_Antihistamines=7.1, ...
    Artifacts: feature_importance.json, lgb_global_model/
```

**Why MLflow matters here:**
- Every run is reproducible — same parameters always produce the same results
- Model comparison is objective — MAPE tracked per drug, not just overall average
- If the model is retrained on new data, the new run is compared against historical runs to detect performance drift
- Feature importance is stored as an artifact — auditable explanation of what drives each prediction

**View all runs locally:**
```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/pharmacast.db
```
Open: `http://localhost:5000`

---

## Walk-Forward Retraining Simulation

Simulates production MLOps by retraining every 4 weeks on expanding data — mimicking exactly how the model would behave as new sales data arrives in production.

```
Week 200 (Nov 2017): Train → Predict weeks 201-204 → MAPE 11.3% → OK
Week 204 (Dec 2017): Train → Predict weeks 205-208 → MAPE 15.9% → WATCH
Week 208 (Dec 2017): Train → Predict weeks 209-212 → MAPE 21.1% → RETRAIN
Week 212 (Jan 2018): Train → Predict weeks 213-216 → MAPE  4.9% → OK
...continues to end of data
```

Runs for all 4 dashboard drugs. Results saved to `outputs/csv/walk_forward_all.csv`.

| Status | MAPE Range | Action |
|---|---|---|
| OK | < 12% | Model healthy — continue monitoring |
| WATCH | 12–20% | Schedule retraining within 2 weeks |
| RETRAIN | > 20% | Retrain immediately on latest data |

---

## Project Structure

```
PharmaCast/
├── train.py                         # Master pipeline — run this first
├── requirements.txt
│
├── data/
│   └── salesweekly.csv              # Raw weekly pharmaceutical sales (2014–2019)
│
├── src/
│   ├── config.py                    # All constants, paths, hyperparameters
│   ├── data_loader.py               # Load CSV, parse dates, quality checks
│   ├── features.py                  # 17-feature time series engineering
│   ├── models.py                    # Naive, XGBoost, LGB Local, LGB Global + MLflow
│   ├── evaluate.py                  # MAPE, RMSE, R² metric functions
│   ├── forecast.py                  # Recursive 8-week forecast engine
│   ├── walk_forward.py              # Walk-forward retraining simulation (MLOps)
│   ├── kpis.py                      # All dashboard KPI calculations
│   └── plots.py                     # 13 matplotlib chart exports
│
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   ├── page_modules/                # 5 dashboard pages
│   │   ├── 01_summary.py
│   │   ├── 02_executive.py
│   │   ├── 03_drug_performance.py
│   │   ├── 04_financial.py
│   │   └── 05_scenario.py
│   └── components/
│       ├── charts.py                # All Plotly interactive charts
│       └── utils.py                 # CSS theme, KPI cards, cached data loaders
│
├── outputs/
│   ├── csv/                         # model_comparison, forecast_table,
│   │                                # business_impact, dashboard_kpis,
│   │                                # walk_forward_all
│   └── plots/                       # 13 exported PNG charts
│
├── models/
│   └── pipeline_artifacts.pkl       # Trained models + metadata
│
├── docs/
│   └── screenshots/                 # Dashboard and MLflow screenshots
│
└── .mlflow/
    └── pharmacast.db                # MLflow SQLite experiment tracking
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pharma-demand-forecasting.git
cd pharma-demand-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train all models

```bash
python train.py
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open: `http://localhost:8501`

### 5. View MLflow experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///.mlflow/pharmacast.db
```

Open: `http://localhost:5000`

---

## Feature Engineering

17 time series features per drug:

| Feature | Type | Description |
|---|---|---|
| `lag_1`, `lag_4`, `lag_8`, `lag_12` | Lag | Recent demand momentum |
| `lag_52` | Lag | Same week last year — strongest single feature |
| `roll_mean_4`, `roll_mean_12` | Rolling | Short and medium-term demand averages |
| `roll_std_4` | Rolling | Recent demand volatility |
| `month_sin`, `month_cos` | Cyclical | Cyclical month encoding — no discontinuity at year boundary |
| `month`, `quarter`, `week`, `year` | Calendar | Standard calendar features |
| `yoy_growth` | Derived | Year-over-year growth rate |
| `is_peak_season` | Derived | Drug-specific seasonal flag |
| `peak_lag52_interaction` | Interaction | lag_52 × peak season |

Global model adds: `drug_id`, `drug_mean`, `drug_std`, `drug_cv`

---

## Why LightGBM Global Outperforms Local Models

| Aspect | Local (1 per drug) | Global (1 for all drugs) |
|---|---|---|
| Training rows | 244 per drug | 1,952 (8 × 244) |
| Seasonal learning | Drug-specific only | Shared patterns across all drugs |
| Volatile drugs | Overfits noise | Stabilised by stable drugs |
| Deployment | 8 separate models | 1 model artifact |
| Best MAPE | 4.6% | **3.1%** |

---

## Business Impact Calculation

```
Projected Annual Saving =
  (Forecast Error Reduction × Annual Drug Volume × Unit Cost × Inventory Impact Factor)
  + Operational Efficiency Gain

Example — Analgesics / Paracetamol:
  208.63 units/week × 13.5% error reduction × $50/unit × 52 weeks = $73,228

Total across 4 drugs = $132,191
```

---

## Data Splits — Chronological Only

```
Jan 2014 ──────────── Sep 2018 | Sep 2018 ─ Feb 2019 | Feb 2019 ─ Jul 2019 | Jul ─ Sep 2019
         TRAIN (244 wks)              VAL (21 wks)         TEST (24 wks)      FORECAST (9 wks)
```

Random splitting is never used — chronological splits prevent data leakage.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python 3.10, Pandas, NumPy |
| Machine learning | LightGBM, XGBoost, Scikit-learn |
| Statistical analysis | Statsmodels (ADF test, decomposition) |
| Experiment tracking | MLflow (SQLite backend) |
| Static charts | Matplotlib, Seaborn |
| Interactive charts | Plotly |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## Dataset

Source: [Pharma Sales Data — Kaggle](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)

| Code | Drug | Demand Type |
|---|---|---|
| M01AB | Anti-inflammatory (COX) | Stable |
| M01AE | Anti-inflammatory (Propionic) | Stable |
| N02BA | Analgesics (Salicylic) | Mild seasonal |
| N02BE | Analgesics / Paracetamol | Winter seasonal |
| N05B | Anxiolytics | Stable |
| N05C | Hypnotics and Sedatives | Intermittent |
| R03 | Respiratory / Asthma | Winter seasonal |
| R06 | Antihistamines | Spring seasonal |

Dashboard drugs: **M01AB, N02BE, N05B, R06** — selected for MAPE < 20% and demand archetype diversity.

---

## License

MIT License — free to use and adapt with attribution.
