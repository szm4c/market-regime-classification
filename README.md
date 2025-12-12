# market-regime-classification

Comparison of classical and nonlinear methods for market regime classification on TTF front-month futures, using:

- Markov-switching models
- XGBoost 3-class classifier
- Symbolic representations with fuzzy discretization
- Image-based features (Recurrence Plots, Gramian Angular Fields) with clustering

All methods share the same input (daily close prices of the front-month futures strip) and the same target - for each trading day, predict the regime of the next 20 business days:
- `-1` – bearish trend  
- `0` – no trend / sideways  
- `1` – bullish trend  

## Target / label construction

Labels are built once from raw daily prices and then reused by all models. The labeling logic lives in `market_regime.labeling.make_regime_labels`.

## Data flow

- Raw data: `data/raw/daily_prices.csv`  
  Columns:
  - `trading_date`
  - `delivery_date`
  - `open`
  - `close`

- Label creation script:  
  `python -m scripts.create_daily_price_labels`
  
  This will:
  - load data/raw/daily_prices.csv
  - compute regime labels using make_regime_labels
  - save data/preprocessed/daily_prices_with_labels.csv (same data plus a target column)
    
  All model experiments read from the preprocessed file and use the same target.

## Repository structure (simplified)

```
market-regime-classification/
├─ data/
│  ├─ raw/
│  │  └─ daily_prices.csv
│  └─ preprocessed/
│     └─ daily_prices_with_labels.csv       # created by script
│
├─ src/
│  └─ market_regime/
│     ├─ __init__.py
│     └─ labeling.py                        # make_regime_labels, calculate_best_k
│
├─ notebooks/
│  └─ 01_regime_label_construction.ipynb    # label exploration
│
├─ scripts/
│  └─ create_daily_price_labels.py
│
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## Installation

Create and activate a virtual environment, then install dependencies and the package:
```
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Planned model pipelines

Each of the four methods will share the same target labels and differ only in feature construction and modeling approach:
- Markov-switching models
- XGBoost 3-class classifier on time-series features / LGBM
- Symbolic + fuzzy discretization with a classifier on symbolic features
- Image-based representations (Recurrence Plots / GAF) + clustering or classifiers
  
All pipelines will be evaluated in an expanding window / walk-forward fashion, training on data up to time *t* and predicting the regime for the next 20 business days.
