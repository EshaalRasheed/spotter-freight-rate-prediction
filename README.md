# Freight Rate Prediction – Spotter ML Engineer Assessment

Predicts the posted rate ($) of trucking loads using a LightGBM model trained on
Jan–Oct 2025 loads, and produces predictions for 12,000 Nov–Dec 2025 validation loads.

## Project structure

```text
├── data/                     # Provided CSV files
├── notebooks/01_eda.ipynb    # Exploratory data analysis and findings
├── src/
│   ├── data_prep.py          # Loading and cleaning
│   ├── features.py           # Feature engineering and target transform
│   ├── train.py              # Time-based validation + final model training
│   └── predict.py            # Validation and December predictions
├── reports/validation_metrics.json
├── scorer_results/candidate_december.png
├── validation_predictions.csv
├── score.py                  # Provided scorer (unchanged)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python src/train.py      # validate, train final model, save to models/
python src/predict.py    # write validation_predictions.csv and fill December file
python score.py --predictions validation-predictions.csv --december-predictions data/december-chart-inputs.csv
```

## Approach

**Data cleaning**
- Negative weights (292 train / 145 validation rows) mirror the normal range, so the sign is flipped.
- Missing `market_index` is filled with that date's daily average (it is a daily market value).
- 663 loads (1.4%) with impossible rate per mile (outside 3×IQR, e.g. $14/mile) are removed from training only.

**Features**
- Pickup/delivery latitude and longitude instead of city names, because validation contains
  8 cities (12% of loads) never seen in training.
- Equipment, distance, log distance, weight, market index, quote signal, day of week.
- Target: log(rate per mile), converted back to dollars after prediction.

**Validation**
- Time-based expanding window: train on all months before a 2-month test window, for
  May–Jun, Jul–Aug and Sep–Oct, mirroring the real Nov–Dec forecast.
- In each window, 8 random cities are hidden from training to measure accuracy on unseen cities.
- Recent loads are weighted more (30-day half-life) because market conditions drift.

## Results (average across 3 time windows)

| Metric (MAPE) | LightGBM | Baseline (lane median) |
|---|---|---|
| Clean test loads | 2.71% | 5.66% |
| Loads with unseen cities | 2.69% | – |
| All test loads (incl. data errors) | 4.91% | – |