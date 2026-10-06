"""Turn cleaned rows into the feature table the model trains and predicts on."""
import numpy as np
import pandas as pd

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

# Final list of model inputs, in a fixed order
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",  # location
    "equipment",                                                 # trailer type
    "distance", "log_distance", "weight",                        # load size
    "market_index", "quote_signal",                              # market signals
    "day_of_week",                                               # weekly pattern
]
CATEGORICAL = ["equipment"]


def build_features(df):
    """Create the model's input table from a cleaned DataFrame."""
    X = pd.DataFrame(index=df.index)

    # Cities as coordinates instead of names: validation has 8 cities never seen in
    # training, and coordinates let the model price them from nearby known cities
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[col] = df[col]

    # Equipment as a category: LightGBM learns a separate effect for each type
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)

    # Distance in raw and log form (rate per mile falls quickly for short trips)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["weight"] = df["weight"]

    # Market signals
    X["market_index"] = df["market_index"]
    X["quote_signal"] = df["quote_signal"]

    # Day of week (0 = Monday). Month / day-of-year are NOT used: the model would have to
    # guess values for Nov-Dec it never saw, which hurt the time-based validation
    X["day_of_week"] = df["date"].dt.dayofweek

    return X[FEATURES]


def to_target(df):
    """Model target: log of rate per mile. Removes the distance effect on the price scale
    and makes errors proportional (being $50 off matters more on a $300 load)."""
    return np.log(df["posted_rate"] / df["distance"])


def from_target(pred, distance):
    """Convert a predicted log rate per mile back to a dollar rate for the load."""
    return np.exp(pred) * distance