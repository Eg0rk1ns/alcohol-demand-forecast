"""
Forecast on demand (polished).

Key fixes vs previous version:
  - holiday_flag for future dates is taken from the real RU holiday list
    (imported from generate_data.py) instead of a broken placeholder;
  - ad_budget for future days is estimated from non-promo history only,
    to avoid inflating the baseline scenario;
  - LightGBM is trained here with internal validation + early stopping,
    so the model adapts to any dataset, no hard-coded best_iteration.

Usage:
    python forecast.py
        Forecast horizon in days [30]: 30
        SKU to forecast (...) [all]: all
"""
from __future__ import annotations

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RANDOM_SEED, ROOT_DIR, TARGET_COL
from features import (LAG_FEATURES, MODEL_FEATURES, ROLLING_FEATURES,
                      create_features, load_raw)
from generate_data import HOLIDAY_MMDD

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "seed": RANDOM_SEED,
    "verbose": -1,
}
N_ROUNDS = 2000
EARLY_STOP = 60


def _is_holiday(d: pd.Timestamp) -> int:
    return int(any((d.month, d.day) == (m, dd) for m, dd in HOLIDAY_MMDD))


def train_on_all():
    """Train LightGBM global on ~90% of all data, keep last 10% as val for early stop."""
    feat = create_features(load_raw())
    feat = feat.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy()

    dates = feat["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n_val = max(30, int(len(dates) * 0.10))
    val_start = dates.iloc[-n_val]

    train_mask = feat["date"] < val_start
    val_mask = feat["date"] >= val_start

    X_tr = feat.loc[train_mask, ["sku"] + MODEL_FEATURES].copy()
    X_tr["sku"] = X_tr["sku"].astype("category")
    sku_cats = list(X_tr["sku"].cat.categories)
    y_tr = feat.loc[train_mask, TARGET_COL]

    X_val = feat.loc[val_mask, ["sku"] + MODEL_FEATURES].copy()
    X_val["sku"] = pd.Categorical(X_val["sku"], categories=sku_cats)
    y_val = feat.loc[val_mask, TARGET_COL]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=["sku"])
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        PARAMS, dtrain, num_boost_round=N_ROUNDS, valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )
    print(f"[OK] LightGBM global trained: best_iteration={model.best_iteration}, "
          f"{len(feat):,} rows, {len(sku_cats)} SKUs")
    return model, sku_cats, feat


def build_future_row(model, sku_cats, history_df, sku, next_date):
    hist = history_df[history_df["sku"] == sku].sort_values("date").reset_index(drop=True)
    last = hist.iloc[-1]

    non_promo = hist[hist["promo_flag"] == 0]
    ad_future = (
        float(non_promo["ad_budget"].tail(30).mean())
        if len(non_promo) > 0 else 0.0
    )

    row = {
        "date": next_date,
        "sku": sku,
        "price": float(last["price"]),
        "competitor_price": float(last["competitor_price"]),
        "discount": 0.0,
        "promo_flag": 0,
        "ad_budget": ad_future,
        "oos_flag": 0,
        "holiday_flag": _is_holiday(next_date),
        "day_of_week": int(next_date.dayofweek),
        "day_of_year": int(next_date.dayofyear),
        "month": int(next_date.month),
        "is_weekend": int(next_date.dayofweek >= 5),
    }

    for lag in [7, 14, 30]:
        row[f"lag_{lag}"] = float(hist["sales"].iloc[-lag])

    for w in [7, 28]:
        tail = hist["sales"].iloc[-w:]
        row[f"rolling_mean_{w}"] = float(tail.mean())
        row[f"rolling_std_{w}"] = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0

    X = pd.DataFrame([row])[["sku"] + MODEL_FEATURES]
    X["sku"] = pd.Categorical(X["sku"], categories=sku_cats)
    return X, row


def forecast_sku(model, sku_cats, history_df, sku, horizon):
    hist = history_df[history_df["sku"] == sku].sort_values("date").reset_index(drop=True).copy()
    future = []
    for _ in range(horizon):
        next_date = hist["date"].max() + pd.Timedelta(days=1)
        X, row = build_future_row(model, sku_cats, hist, sku, next_date)
        yhat = max(float(model.predict(X)[0]), 0.0)

        new_row = {c: row.get(c, np.nan) for c in hist.columns}
        new_row["date"] = next_date
        new_row["sales"] = yhat
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        future.append({"date": next_date, "sku": sku, "forecast": yhat})
    return pd.DataFrame(future)


def plot_forecast(sku, hist_df, future_df, horizon):
    h = hist_df[hist_df["sku"] == sku].sort_values("date").tail(90)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(h["date"], h["sales"], color="tab:blue", label="history (last 90d)")
    ax.plot(future_df["date"], future_df["forecast"], color="tab:red",
            marker="o", ms=3, label=f"forecast ({horizon}d)")
    ax.axvline(h["date"].max(), color="gray", ls="--", lw=1)
    ax.set_title(f"Demand forecast: {sku}, horizon={horizon} days")
    ax.set_ylabel("units sold")
    ax.legend()
    fig.tight_layout()
    out = REPORTS_DIR / f"forecast_{sku}_{horizon}d.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def prompt_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  '{raw}' is not an integer, using default {default}")
        return default


def prompt_sku(available):
    raw = input(f"SKU to forecast ({', '.join(available)} or 'all') [all]: ").strip()
    if not raw or raw.lower() == "all":
        return "all"
    if raw in available:
        return raw
    print(f"  unknown SKU '{raw}', falling back to 'all'")
    return "all"


if __name__ == "__main__":
    model, sku_cats, _ = train_on_all()
    history = load_raw()

    horizon = prompt_int("Forecast horizon in days", 30)
    sku = prompt_sku(sorted(history["sku"].unique()))

    targets = sku_cats if sku == "all" else [sku]
    all_fc = []
    for s in targets:
        fc = forecast_sku(model, sku_cats, history, s, horizon)
        all_fc.append(fc)
        plot_out = plot_forecast(s, history, fc, horizon)
        out_csv = REPORTS_DIR / f"forecast_{s}_{horizon}d.csv"
        fc.to_csv(out_csv, index=False)
        print(f"[OK] {s}: {horizon}d -> {out_csv.name}, plot: {plot_out.name}")

    combined = pd.concat(all_fc, ignore_index=True)
    combined.to_csv(REPORTS_DIR / f"forecast_all_{horizon}d.csv", index=False)
    print(f"\n[OK] combined forecast -> reports/forecast_all_{horizon}d.csv")
    print(combined.groupby("sku")["forecast"]
          .agg(["sum", "mean", "min", "max"]).round(2).to_string())
