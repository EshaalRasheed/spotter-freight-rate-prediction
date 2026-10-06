"""Load the raw CSV files and clean the data-quality issues found during EDA."""
from pathlib import Path

import pandas as pd

# Project folders, built from this file's location so scripts work from any directory
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def load_data():
    """Read the training, validation and December files, parsing dates."""
    train = pd.read_csv(DATA_DIR / "train-test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv", parse_dates=["date"])
    return train, valid, december


def fix_weight(df):
    """Negative weights mirror the normal range (flipped sign), so take the absolute value.
    Missing weights stay as NaN; LightGBM handles missing values natively."""
    df = df.copy()
    df["weight"] = df["weight"].abs()
    return df


def daily_market_index(*frames):
    """market_index is a daily market value (loads on the same day barely differ),
    so the average per date is a reliable estimate for that day."""
    combined = pd.concat([f[["date", "market_index"]] for f in frames])
    return combined.groupby("date")["market_index"].mean()


def fill_market_index(df, daily_index):
    """Fill missing market_index values with that date's daily average."""
    df = df.copy()
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily_index))
    return df


def city_coordinates(*frames):
    """Build a city -> (lat, lon) lookup from pickup and delivery columns.
    Used to add coordinates to the December file, which only has city names."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].mean()


def add_coordinates(df, coords):
    """Add pickup/delivery latitude and longitude columns from the city lookup."""
    df = df.copy()
    df["pickup_lat"] = df["pickup"].map(coords["lat"])
    df["pickup_lon"] = df["pickup"].map(coords["lon"])
    df["delivery_lat"] = df["delivery"].map(coords["lat"])
    df["delivery_lon"] = df["delivery"].map(coords["lon"])
    return df


def flag_rate_outliers(df, k=3.0):
    """Mark rows whose rate per mile is extreme (beyond k * IQR from the quartiles).
    These look like data errors and badly hurt the model if kept in training."""
    rate_per_mile = df["posted_rate"] / df["distance"]
    q1, q3 = rate_per_mile.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (rate_per_mile < low) | (rate_per_mile > high)