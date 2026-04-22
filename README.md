# EcoTracing

> Data Center Energy Consumption Prediction & Carbon Emission Estimation

---

## Overview

EcoTracing is a machine learning research project that predicts data center energy consumption from CPU/Memory usage patterns and estimates carbon emissions from cloud computing workloads.

Using **Google Cluster Trace 2019** data (~20M samples), we benchmark multiple ML models and progressively improve extrapolation performance — a key challenge when training data duration (5-min intervals) differs from real-world inference conditions (hourly input).

---

## Project Goals

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Energy consumption prediction from CPU/Memory usage | ✅ In Progress |
| Phase 2 | Energy → Carbon emission conversion | 🔜 Planned |
| Phase 3 | AI training/inference carbon footprint analysis | 🔜 Planned |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Environment | Google Colab (A100 GPU) |
| ML Framework | scikit-learn, PyTorch |
| Models | LightGBM, XGBoost, RandomForest, CatBoost, MLP |
| Data | Google Cluster Trace 2019 |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Storage | Google Drive |

---

## Project Structure

```
EcoTracing/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── config.yaml
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 08_deep_learning.ipynb          # MLP baseline
│   ├── 09_preprocessing_hourly.ipynb   # Hourly aggregation
│   ├── 10_modeling_hourly.ipynb        # RF (hourly)
│   ├── 12_modeling_hourly_log.ipynb    # RF (hourly + log transform)
│   ├── 13_mlp_hourly.ipynb             # MLP (hourly + augmentation)
│   ├── 14_stacking_mlp_rf.ipynb        # MLP + RF Stacking
│   └── 15_residual_mlp.ipynb           # Residual MLP (final)
├── streamlit/
│   ├── app.py
│   ├── config/
│   │   └── config.yaml
│   ├── models/
│   ├── pages/
│   │   ├── predictor.py
│   │   ├── eda.py
│   │   ├── model_compare.py
│   │   └── feature_importance.py
│   └── utils/
│       ├── __init__.py
│       ├── loader.py
│       └── predictor.py
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
└── data/
    ├── raw/
    └── processed/
```

---

## Data Source

**Google Cluster Trace 2019**
- Source: https://github.com/google/cluster-data
- Data: `instance_usage` (CPU/Memory usage per instance)
- Samples: ~19.5M rows (5 files)

---

## Energy Formula

```
Power (W)   = 200 + (CPU_usage × 300) + (Memory_usage × 50)
Energy (kWh) = Power (W) × Duration (h) / 1000
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Power | 200W | Idle server power (SPECpower benchmark) |
| CPU Max | 300W | Additional power at 100% CPU (Intel Xeon / AMD EPYC) |
| Memory Max | 50W | Additional power at 100% Memory (DDR4 DIMM) |

---

## Model Experiments (Phase 1)

### Core Challenge
Training data has `duration` fixed at ~300s (5-min intervals), while Streamlit inference uses 1-hour inputs → RandomForest cannot extrapolate beyond training range.

**Test Condition**: CPU=50%, Memory=30%, Duration=1h → Formula Reference: **0.365 kWh**

| # | Method | Train MAPE | Extrapolation Error | Note |
|---|--------|------------|---------------------|------|
| 1 | RF (baseline) | 0.07% | 74.6% | Cannot extrapolate |
| 2 | RF+LR Stacking (depth=10) | 0.63% | 32.0% | |
| 3 | RF+LR Stacking (depth=15) | 0.059% | 26.6% | |
| 4 | Augmentation + Stacking | 0.63% | 28.3% | Worse |
| 5 | Augmentation + Retuning | 0.059% | 26.5% | |
| 6 | power_w feature | — | 53.2% | Worse |
| 7 | MLP (basic) | 26.93% | ~26% | |
| 8 | Residual Learning (RF→LR) | 0.343% | — | |
| 9 | Dynamic Weighted Blending | 0.143% | > RF | |
| 10 | **MLP (5min original)** | **2.18%** | **13.2%** | Best so far |
| 11 | Hourly Aggregation + MLP | — | > MLP | Aggregation hurts |
| 12 | RF (hourly + log transform) | — | — | |
| 13 | MLP (hourly + augmentation) | 1.37% | > MLP | |
| 14 | MLP + RF Stacking | 0.063% | ~RF level | Meta LR → RF dominated |
| **15** | **Residual MLP** | — | **0.33%** | **✅ Final Model** |

### Final Model: Residual MLP

```
y_final = y_formula(cpu, memory, duration) + MLP_pred_residual
```

- MLP learns the residual between actual and formula value
- Since training data is formula-derived, residual ≈ 0
- At inference (1h input): y_formula dominates → near-zero extrapolation error
- **Extrapolation error improved from 74.6% → 0.33% (98% improvement)**

### Base Model Performance (Phase 1)

| Rank | Model | RMSE | R² | MAPE | Train Time |
|------|-------|------|-----|------|------------|
| 🥇 | RandomForest | 1.38e-05 | 0.99999 | 0.07% | 470s |
| 🥈 | LightGBM | 3.42e-05 | 0.99997 | 0.15% | 42s |
| 🥉 | CatBoost | 4.31e-05 | 0.99995 | 0.79% | 78s |
| 4 | XGBoost | 5.67e-05 | 0.99992 | 2.78% | 66s |

---

## Training Environment

| Item | Spec |
|------|------|
| Platform | Google Colab |
| GPU | NVIDIA A100 |
| Runtime | Python 3.12 |
| Storage | Google Drive (~TB) |
| Data Volume | ~19.5M rows |

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/elfinLily/EcoTracing.git
cd EcoTracing
```

### 2. Setup Google Colab
- Open notebook in VS Code with Colab extension
- Connect to A100 GPU runtime
- Mount Google Drive

### 3. Create config.yaml
```python
config_content = """
project_name: "EcoTracing"

paths:
  raw_data: "raw"
  processed_data: "data/processed"
  models: "models"
  outputs: "outputs"

data:
  source: "google_cluster_trace"
  sample_size: 100000

model:
  type: "lightgbm"
  random_state: 42
  test_size: 0.2
  model_names:
    lightgbm: "energy_model_lightgbm.pkl"
    xgboost: "energy_model_xgboost.pkl"
    randomforest: "energy_model_randomforest.pkl"
    catboost: "energy_model_catboost.pkl"
    mlp: "energy_model_mlp.pkl"
    mlp_residual: "energy_model_mlp_residual.pkl"
    scaler_x: "scaler_x.pkl"
    scaler_y: "scaler_y.pkl"
    scaler_x_residual: "scaler_x_residual.pkl"
  results:
    metrics_csv: "phase1_metrics.csv"
    results_json: "phase1_full_results.json"
    feature_importance_csv: "feature_importance_comparison.csv"

carbon:
  emission_factor: 0.5
"""
with open("/content/config.yaml", "w") as f:
    f.write(config_content)
```

### 4. Run Notebooks in Order
```
01 → 02 → 03 → 04 → 05 → 08 → 09 → 10 → 15
```

### 5. Run Streamlit Dashboard
```bash
cd streamlit
streamlit run app.py
```

---

## License

MIT License