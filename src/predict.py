"""Predict every validation load and every December chart row with the trained model."""
import joblib
import numpy as np
import pandas as pd

from data_prep import (DATA_DIR, MODELS_DIR, ROOT, add_coordinates, fill_market_index,
                       fix_weight, load_data)
from features import build_features, from_target


def main():
    # Load the saved model and the lookups saved with it during training
    artifacts = joblib.load(MODELS_DIR / "freight_rate_model.joblib")
    model = artifacts["model"]
    daily_index = artifacts["daily_market_index"]
    coords = artifacts["city_coordinates"]
    quote_median = artifacts["quote_signal_median"]

    _, valid, december = load_data()

    # ---- 1. Validation predictions ----
    # Apply exactly the same cleaning as training
    valid_clean = fill_market_index(fix_weight(valid), daily_index)
    valid_pred = from_target(model.predict(build_features(valid_clean)), valid_clean["distance"])
    predictions = pd.Series(valid_pred.to_numpy(), index=valid_clean["load_id"])

    # Fill the template in its own row order, matching by load_id
    template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")
    template["predicted_rate"] = template["load_id"].map(predictions).round(2)

    # Safety checks: every load has a positive prediction
    assert template["predicted_rate"].notna().all(), "some load_ids have no prediction"
    assert (template["predicted_rate"] > 0).all(), "non-positive predictions found"

    # Saved in the project root, where the README's score.py command expects it
    out_path = ROOT / "validation-predictions.csv"
    template[["load_id", "predicted_rate"]].to_csv(out_path, index=False)
    print(f"Saved {len(template):,} predictions to {out_path}")
    print(template["predicted_rate"].describe().round(2), "\n")

    # ---- 2. December chart predictions ----
    # The December file only has city names, distance, equipment, weight and date,
    # so fill the missing model inputs:
    dec = add_coordinates(december, coords)              # coordinates from the city lookup
    dec["market_index"] = dec["date"].map(daily_index)   # that day's average market index
    dec["quote_signal"] = quote_median                   # typical (median) quote signal
    dec = fix_weight(dec)

    assert dec[["pickup_lat", "delivery_lat", "market_index"]].notna().all().all(), \
        "December inputs could not be filled"

    dec_pred = from_target(model.predict(build_features(dec)), dec["distance"])

    # Write predictions into the original file, keeping its 7 columns and order
    december["predicted_rate"] = np.round(dec_pred.to_numpy(), 2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DATA_DIR / "december-chart-inputs.csv", index=False)
    print(f"Filled {len(december)} December predictions in {DATA_DIR / 'december-chart-inputs.csv'}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()