# EcoTracing

Data Center Energy Consumption & Carbon Emission Prediction

---

## Overview

EcoTracing is a machine learning project that predicts data center energy consumption and estimates carbon emissions from cloud computing workloads. Using Google Cluster Trace data, we build models to understand the environmental impact of large-scale computing infrastructure.

---

## Project Goals

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Energy consumption prediction from CPU/Memory usage | In Progress |
| Phase 2 | Energy to Carbon emission conversion | Planned |
| Phase 3 | AI training/inference carbon footprint analysis | Planned |

---

## Tech Stack

- **Language**: Python 3.10+
- **Environment**: Google Colab (L4 GPU)
- **ML Framework**: LightGBM, scikit-learn
- **Data**: Google Cluster Trace 2019
- **Visualization**: Matplotlib, Seaborn
- **Storage**: Google Drive

---

## Project Structure

```
EcoTracing/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
└── tests/
```

---

## Data Source

**Google Cluster Trace 2019**
- Source: https://github.com/google/cluster-data
- Data: `instance_usage` (CPU/Memory usage per instance)
- Size: Multiple TB (sampled for this project)

---

## Energy Calculation

```
Power (W) = 200 + (CPU_usage * 300) + (Memory_usage * 50)
Energy (kWh) = Power (W) * Duration (h) / 1000
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Power | 200W | Idle server power |
| CPU Max | 300W | Additional power at 100% CPU |
| Memory Max | 50W | Additional power at 100% Memory |

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/EcoTracing.git
cd EcoTracing
```

### 2. Setup Google Colab
- Open notebook in VS Code with Colab extension
- Connect to L4 GPU runtime
- Mount Google Drive

### 3. Create config.yaml in Colab
```python
config_content = """
project_name: "EcoTracing"

paths:
  drive_data: "/content/drive/MyDrive/EcoTracing/data"
  raw_data: "raw"
  processed_data: "processed"
  models: "models"
  outputs: "outputs"

data:
  source: "google_cluster_trace"
  sample_size: 50000

model:
  type: "lightgbm"
  random_state: 42
  test_size: 0.2

carbon:
  emission_factor: 0.5
"""

with open("/content/config.yaml", "w") as f:
    f.write(config_content)
```

### 4. Run Notebooks in Order
```
01_data_download -> 02_eda -> 03_preprocessing -> 04_modeling -> 05_evaluation
```

---

## Model Performance (Phase 1)

**Dataset**: Google Cluster Trace 2019 (19,523,808 samples)

| Rank | Model | RMSE | R² | MAPE | Train Time |
|------|-------|------|-----|------|------------|
| 🥇 | **RandomForest** | 1.38e-05 | 0.99999 | 0.07% | 470s |
| 🥈 | LightGBM | 3.42e-05 | 0.99997 | 0.15% | 42s |
| 🥉 | CatBoost | 4.31e-05 | 0.99995 | 0.79% | 78s |
| 4 | XGBoost | 5.67e-05 | 0.99992 | 2.78% | 66s |

**Best Model**: RandomForest (highest R², lowest MAPE)

---

## Features

| Feature | Description |
|---------|-------------|
| cpu | CPU usage ratio (0-1) |
| memory | Memory usage ratio (0-1) |
| duration | Measurement duration (seconds) |
| hour | Hour of day (0-23) |

---

## License

MIT License