# market-regime-classification

Market regime classification for energy futures (TTF front-month), combining classical time-series models and modern machine-learning approaches under a common labeling and evaluation framework. Tested models are:
- XGBoost 3-class classifier
- ARCH model
- Image-based features (Recurrence Plots, Gramian Angular Fields) with CNN

All methods share the same input (daily close prices of the front-month futures strip) and the same target - for each trading day, predict the regime of the next 20 business days:
- `-1` – bearish trend  
- `0` – no trend / sideways  
- `1` – bullish trend  

## Target / label construction

Labels are built once from raw daily prices and then reused by all models. The labeling logic lives in `market_regime.labels`.

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

## Feature construction

Feature engineering is modular and lives in `src/market_regime/features/`.

Key components:

- base.py
  Abstract base class for all feature builders.

- tree_classifier_features.py
  Time-series features for tree-based models (lags, rolling statistics, returns, volatility proxies, calendar effects).

- image_features.py
  Image-based representations (e.g. Recurrence Plots, Gramian Angular Fields) intended for CNN-based models.

- sharpe_features.py
  Only log-return calculation.

- utils.py
  Shared helpers (validation, log-return calculation, etc.).

Each feature builder converts a DataFrame into (X, y, index, features) and is fully compatible with walk-forward evaluation.

## Models

Models are implemented in `src/market_regime/models/`.

List of all models:

- xgb.py
  XGBoost multiclass classifier for regimes {-1, 0, 1}.

- sharpe_arch.py
  ARCH / GARCH-based forecasting model.

- torch_cnn.py
  CNN-based classifier consuming image representations.

- base.py
  Common model interface (fit, predict, predict_proba).

## Pipelines

Pipelines combine feature builders and models into a single object.

Pipeline code lives in `src/market_regime/pipelines/`.

Key file:
- base.py
  Defines the RegimePipeline, which:
  - builds features
  - fits the estimator
  - produces predictions or probabilities

Example usage:
```
rp = RegimePipeline(
    feature_builder=TreeClassifierFeatureBuilder(),
    estimator=XGBTreeModel(**best_params),
)
```

This design cleanly separates:
- data and feature logic
- model logic
- evaluation logic

## Evaluation

All models are evaluated using a walk-forward / expanding window procedure:

- At each date $t$:
  - train on all data available up to $t-1$
  - predict the regime for date $t$
- Predictions are aggregated into a full out-of-sample series.

Evaluation metrics include:
- macro F1-score
- balanced accuracy
- confusion matrices
- class distribution diagnostics

This ensures no look-ahead bias and realistic performance estimates.

## Repository structure

```
market-regime-classification/
├─ data/
│  ├─ params/
│  ├─ raw/
│  │  └─ daily_prices.csv
│  └─ preprocessed/
│     └─ daily_prices_with_labels.csv       # created by script
│
├─ notebooks/
│  ├─ 00_data_exploration.ipynb
│  ├─ 01_regime_label_construction.ipynb
│  ├─ 02_recurrence_plot_and_gramian_angular_fields.ipynb
│  ├─ 03_arma_garch_exploration.ipynb
│  ├─ 05_tree_classifier_exploration.ipynb
│  └─ 06_experiment.ipynb
│
├─ scripts/
│  └─ create_daily_price_labels.py
|
├─ src/
│  └─ market_regime/
│     ├─ features/
│     ├─ labels/
│     ├─ models/
│     ├─ pipelines/
│     ├─ data.py
│     └─ experiment.py
│
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## Installation
1. Clone the repo locally.
2. Create a virtual environment:
    ```
    python -m venv .venv
    ```
3. Activate a virtual environment:
    - Windows
      ```
      .venv\Scripts\Activate.ps1
      ```
    - Linux/macOS
      ```
      source .venv/bin/activate
      ```
4. Install dependencies and the package inside the venv:
    ```
    pip install -r requirements.txt
    ```
    ```
    pip install -e .
    ```