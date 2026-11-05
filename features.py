"""
Feature engineering and chronological splits (leak-free).

Key principles:
- All lag / rolling features are shifted by >=1 day so the model never
  sees the target of the day it is predicting.
- Rolling stats are computed per-SKU inside the group.
- Split by date: train -> val -> test (no shuffling).
"""
from __future__ import annotations

import pandas as pd

from config import (
    CALENDAR_FEATURES, EXOG_FEATURES, LAGS, ROLLING_WINDOWS,
    SALES_CSV, TARGET_COL, TRAIN_FRAC, VAL_FRAC,
)

LAG_FEATURES = [f"lag_{l}" for l in LAGS]
ROLLING_FEATURES = (
    [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
)
MODEL_FEATURES = CALENDAR_FEATURES + EXOG_FEATURES + LAG_FEATURES + ROLLING_FEATURES


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(SALES_CSV, parse_dates=["date"])
    return df.sort_values(["sku", "date"]).reset_index(drop=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()

    # --- Calendar ---
    df["day_of_week"] = df["date"].dt.day_of_week.astype(int)
    df["day_of_year"] = df["date"].dt.day_of_year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Lags (per SKU) ---
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("sku")[TARGET_COL].shift(lag)

    # --- Rolling stats: shift(1) first, then roll, so no leakage ---
    df["_sales_shifted"] = df.groupby("sku")[TARGET_COL].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = (
            df.groupby("sku")["_sales_shifted"]
              .transform(lambda s: s.rolling(w).mean())
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("sku")["_sales_shifted"]
              .transform(lambda s: s.rolling(w).std())
        )
    df = df.drop(columns=["_sales_shifted"])

    return df


def train_val_test_split(df: pd.DataFrame):
    """Chronological split by unique dates (same dates for all SKUs)."""
    df = df.sort_values("date")
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(dates)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_end = dates.iloc[n_train - 1]
    val_end = dates.iloc[n_train + n_val - 1]

    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    return train, val, test


def to_xy(df: pd.DataFrame):
    """Drop NaN lag rows, return X, y, cleaned_df."""
    clean = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy()
    X = clean[MODEL_FEATURES]
    y = clean[TARGET_COL]
    return X, y, clean


if __name__ == "__main__":
    raw = load_raw()
    feat = create_features(raw)
    train, val, test = train_val_test_split(feat)

    print(f"raw rows         : {len(raw):,}")
    print(f"with features    : {len(feat):,}")
    print(f"train  rows      : {len(train):,}  ({train['date'].min().date()} .. {train['date'].max().date()})")
    print(f"val    rows      : {len(val):,}  ({val['date'].min().date()} .. {val['date'].max().date()})")
    print(f"test   rows      : {len(test):,}  ({test['date'].min().date()} .. {test['date'].max().date()})")
    print(f"n features       : {len(MODEL_FEATURES)}")
    print(f"features         : {MODEL_FEATURES}")
    print("\nNaN check on feature columns (train):")
    print(train[MODEL_FEATURES].isna().sum()[lambda s: s > 0].to_string() or "  no NaN in train")
