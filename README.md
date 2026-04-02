# EcoTracing

> Data Center Energy Consumption & Carbon Emission Prediction

---

## Overview

EcoTracing is a machine learning project that predicts data center energy consumption and estimates carbon emissions from cloud computing workloads. Using Google Cluster Trace 2019 data, we build models to understand the environmental impact of large-scale computing infrastructure.

---

## Project Goals

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Energy consumption prediction from CPU/Memory usage | ✅ Complete |
| Phase 1.5 | Streamlit dashboard for Phase 1 visualization | ✅ Complete |
| Phase 2 | Energy (kWh) to Carbon emission (kg CO2) conversion | 🔜 Planned |
| Phase 3 | AI training/inference carbon footprint analysis | 🔜 Planned |

---

## Tech Stack

- **Language**: Python 3.10+
- **Environment**: Google Colab (L4 GPU) + VS Code Colab Extension
- **ML Models**: RandomForest, LightGBM, XGBoost, CatBoost, LinearRegression
- **Ensemble**: Stacking (RF + LR -> Meta LinearRegression)
- **Dashboard**: Streamlit + Plotly
- **Data**: Google Cluster Trace 2019
- **Storage**: Google Drive (data), GitHub (code)

---

## Project Structure

```
EcoTracing/
├── README.md
├── .gitignore
├── config/
│   └── config.yaml                    # Project config
├── docs/
│   └── methodology/
│       └── energy_formula.md          # Energy formula reference
├── notebooks/
│   ├── 01_data_download.ipynb         # Download Google Cluster Trace
│   ├── 02_eda.ipynb                   # Exploratory Data Analysis
│   ├── 03_preprocessing.ipynb         # Feature engineering (full dataset)
│   ├── 04_modeling.ipynb              # Multi-model training
│   ├── 05_evaluation.ipynb            # Evaluation & visualization
│   └── 06_ensemble.ipynb              # Stacking ensemble (RF + LR)
├── streamlit/
│   ├── app.py                         # Main entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Config/model/data loader
│   │   └── predictor.py               # Energy prediction logic
│   └── pages/
│       ├── predictor.py               # Real-time energy predictor
│       ├── eda.py                     # EDA visualization
│       ├── model_compare.py           # Model comparison charts
│       └── feature_importance.py      # Feature importance charts
├── models/                            # Trained models (Google Drive)
└── data/ (Google Drive)
    ├── raw/                           # Google Cluster Trace raw files
    └── processed/                     # Preprocessed parquet files
```

---

## Data Source

**Google Cluster Trace 2019**
- Source: https://github.com/google/cluster-data
- Data: `instance_usage` (CPU/Memory usage per instance)
- Samples: 19,523,808 rows (5 files)
- Train/Test: 15,619,046 / 3,904,762 (80/20 split)

---

## Energy Calculation Formula

```
Power (W)   = 200 + (CPU_usage * 300) + (Memory_usage * 50)
Energy (kWh) = Power (W) * Duration (h) / 1000
```

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Base Power | 200W | Barroso & Holzle (2009), Fan et al. (2007) |
| CPU Max | 300W | Intel Xeon / AMD EPYC TDP |
| Memory Max | 50W | Lefurgy et al. (2003) |

---

## Features

| Feature | Type | Description | Importance |
|---------|------|-------------|------------|
| **cpu** | float | CPU usage ratio (0~1) | 2nd (41%) |
| **memory** | float | Memory usage ratio (0~1) | 3rd (8%) |
| **duration** | float | Measurement duration (seconds) | 1st (51%) |

> Note: `hour` feature was removed (importance < 1%)

---

## Model Performance

### Phase 1 - Single Models

| Rank | Model | RMSE | R2 | MAPE | Train Time |
|------|-------|------|-----|------|------------|
| 1 | RandomForest | 1.38e-05 | 0.99999 | 0.07% | 470s |
| 2 | LightGBM | 3.42e-05 | 0.99997 | 0.15% | 42s |
| 3 | CatBoost | 4.31e-05 | 0.99995 | 0.79% | 78s |
| 4 | XGBoost | 5.67e-05 | 0.99992 | 2.78% | 66s |

### Phase 1 - Ensemble (Stacking)

| Model | RMSE | MAPE | Error vs Formula |
|-------|------|------|-----------------|
| RF only | 1.38e-05 | 0.07% | ~75% |
| **Stacking (RF+LR)** | **1.3e-05** | **0.63%** | **~32%** |

> Stacking reduced extrapolation error by 57% compared to RF alone.

**Why Stacking?**
- RandomForest: Cannot extrapolate beyond training range (max 300s duration)
- LinearRegression: Linear extrapolation for duration
- Meta model learns optimal combination of both

Reference: Zhang et al. (2019) - *Regression-Enhanced Random Forests*

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/elfinLily/EcoTracing.git
cd EcoTracing
```

### 2. Install Dependencies
```bash
pip install -r streamlit/requirements.txt
```

### 3. Run Streamlit Dashboard
```bash
streamlit run streamlit/app.py
```

---

## References

- Barroso & Holzle (2009) - *The Datacenter as a Computer*
- Fan et al. (2007) - *Power Provisioning for a Warehouse-sized Computer*
- Lefurgy et al. (2003) - *Energy Management for Commercial Servers*
- Zhang et al. (2019) - *Regression-Enhanced Random Forests*
- SPECpower Committee - *SPECpower_ssj2008*
- Google Cluster Trace 2019: https://github.com/google/cluster-data