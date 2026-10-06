"""Validate with time-based splits, compare against a baseline, then train the final model."""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from data_prep import (MODELS_DIR, REPORTS_DIR, city_coordinates, daily_market_index,
                       fill_market_index, fix_weight, flag_rate_outliers, load_data)
from features import CATEGORICAL, build_features, from_target, to_target

# LightGBM settings: many small trees with a low learning rate for stable results
MODEL_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Each fold trains on all months before the test window and tests on the next 2 months,
# mimicking the real task: learn from the past, predict 2 future months (Nov-Dec)
FOLDS = [
    ("2025-05-01", "2025-06-30"),  # train Jan-Apr, test May-Jun
    ("2025-07-01", "2025-08-31"),  # train Jan-Jun, test Jul-Aug
    ("2025-09-01", "2025-10-31"),  # train Jan-Aug, test Sep-Oct
]

# Validation contains 8 cities never seen in training, so each fold also hides
# 8 random cities from training to measure how well the model prices new cities
N_HIDDEN_CITIES = 8

# Recency weighting: a load from 30 days before the forecast window counts half as much
# as a load from the last day. Market conditions drift, so recent loads are more relevant
HALF_LIFE_DAYS = 30


def recency_weights(dates, reference_date):
    """Weight = 0.5 ** (age in days / half-life), relative to the start of the forecast window."""
    age_days = (pd.Timestamp(reference_date) - dates).dt.days
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def metrics(actual, predicted):
    """MAE in dollars and MAPE in percent."""
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "mape_pct": round(float(np.mean(np.abs(predicted / actual - 1)) * 100), 2),
    }


def baseline_predict(train_rows, test_rows):
    """Simple baseline: median rate per mile for the same lane and equipment (falling back
    to the equipment median for unseen lanes) times distance."""
    rpm = train_rows["posted_rate"] / train_rows["distance"]
    keys = ["pickup", "delivery", "equipment"]
    lane_median = rpm.groupby([train_rows[k] for k in keys]).median()
    equip_median = rpm.groupby(train_rows["equipment"]).median()
    lane_rpm = pd.Series(
        lane_median.reindex(pd.MultiIndex.from_frame(test_rows[keys])).to_numpy(),
        index=test_rows.index,
    )
    lane_rpm = lane_rpm.fillna(test_rows["equipment"].map(equip_median))
    return lane_rpm * test_rows["distance"]


def train_model(rows, reference_date):
    """Fit LightGBM on log rate per mile with recency weights."""
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(build_features(rows), to_target(rows),
              sample_weight=recency_weights(rows["date"], reference_date),
              categorical_feature=CATEGORICAL)
    return model


def main():
    train, valid, december = load_data()

    # Cleaning (the same rules are applied to the prediction files in predict.py)
    daily_index = daily_market_index(train, valid)
    train = fill_market_index(fix_weight(train), daily_index)
    is_outlier = flag_rate_outliers(train)
    print(f"Rows: {len(train):,} | rate outliers removed from training: {is_outlier.sum():,}\n")

    all_cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    results = []

    # ---- Time-based validation ----
    for fold_number, (test_start, test_end) in enumerate(FOLDS):
        # Pick cities to hide from training (fixed seed so results are repeatable)
        rng = np.random.default_rng(fold_number)
        hidden = set(rng.choice(all_cities, N_HIDDEN_CITIES, replace=False))
        touches_hidden = train["pickup"].isin(hidden) | train["delivery"].isin(hidden)

        past = train["date"] < test_start
        future = (train["date"] >= test_start) & (train["date"] <= test_end)
        fit_rows = train[past & ~is_outlier & ~touches_hidden]  # clean past data, no hidden cities
        test_rows = train[future]                                # ALL future rows

        model = train_model(fit_rows, test_start)
        pred = from_target(model.predict(build_features(test_rows)), test_rows["distance"])
        base = baseline_predict(fit_rows, test_rows)

        actual = test_rows["posted_rate"]
        clean = ~is_outlier[test_rows.index]
        new_city = touches_hidden[test_rows.index]

        fold = {
            "test_window": f"{test_start} to {test_end}",
            "train_rows": len(fit_rows),
            "test_rows": len(test_rows),
            "lightgbm_all": metrics(actual, pred),
            "lightgbm_clean": metrics(actual[clean], pred[clean]),
            "lightgbm_clean_known_cities": metrics(actual[clean & ~new_city], pred[clean & ~new_city]),
            "lightgbm_clean_new_cities": metrics(actual[clean & new_city], pred[clean & new_city]),
            "baseline_clean": metrics(actual[clean], base[clean]),
        }
        results.append(fold)
        print(json.dumps(fold, indent=2))

    # Average error across folds: the numbers to quote in the report
    print()
    for key in ["lightgbm_all", "lightgbm_clean", "lightgbm_clean_known_cities",
                "lightgbm_clean_new_cities", "baseline_clean"]:
        avg = np.mean([r[key]["mape_pct"] for r in results])
        print(f"Average MAPE {key:30s} {avg:.2f}%")

    # ---- Final model on ALL clean development data ----
    # Weights relative to 2025-11-01, the start of the real prediction window
    final_model = train_model(train[~is_outlier], "2025-11-01")

    # Save the model plus everything predict.py needs to rebuild features identically
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": final_model,
        "daily_market_index": daily_index,
        "city_coordinates": city_coordinates(train, valid),
        "quote_signal_median": float(train["quote_signal"].median()),
    }, MODELS_DIR / "freight_rate_model.joblib")

    # Feature importance (share of total gain) for the report and Loom
    importance = pd.Series(
        final_model.booster_.feature_importance("gain"), index=final_model.feature_name_
    ).sort_values(ascending=False)
    print("\nFeature importance (gain share):")
    print((importance / importance.sum()).round(3))

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model to {MODELS_DIR / 'freight_rate_model.joblib'}")
    print(f"Saved metrics to {REPORTS_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()