"""
Baseline forecasts (the yardstick every model must beat).

Models:
  - naive_7         : sales[date - 7]            (same weekday last week)
  - seasonal_365    : sales[date - 365]          (same day last year)
  - rolling_28      : mean(sales[date-28 .. date-1])

Metrics on val and test (overall and per-SKU):
  MAE, RMSE, WAPE (%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT_DIR, TARGET_COL
from features import load_raw, train_val_test_split

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def add_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    g = df.groupby("sku")[TARGET_COL]

    df["pred_naive_7"] = g.shift(7)
    df["pred_seasonal_365"] = g.shift(365)
    df["pred_rolling_28"] = df.groupby("sku")[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    return df


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - p)) / denom * 100.0) if denom > 0 else float("nan")


def evaluate_split(df: pd.DataFrame, pred_cols, split_name: str) -> pd.DataFrame:
    rows = []
    for col in pred_cols:
        sub = df.dropna(subset=[col])
        model_name = col.replace("pred_", "")

        # aggregate
        y, p = sub[TARGET_COL].values, sub[col].values
        rows.append({
            "split": split_name, "model": model_name, "n": len(sub),
            "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE_%": wape(y, p),
        })

        # per-SKU
        for sku, g in sub.groupby("sku"):
            yy, pp = g[TARGET_COL].values, g[col].values
            rows.append({
                "split": f"{split_name}:{sku}", "model": model_name, "n": len(g),
                "MAE": mae(yy, pp), "RMSE": rmse(yy, pp), "WAPE_%": wape(yy, pp),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_raw()
    df = add_baseline_predictions(df)
    train, val, test = train_val_test_split(df)

    pred_cols = ["pred_naive_7", "pred_seasonal_365", "pred_rolling_28"]
    results = pd.concat(
        [evaluate_split(val, pred_cols, "val"),
         evaluate_split(test, pred_cols, "test")],
        ignore_index=True,
    )

    # aggregate view (no ':' in split => overall, not per-SKU)
    agg = results[~results["split"].str.contains(":")].reset_index(drop=True)
    print("=== AGGREGATE (all SKUs) ===")
    print(agg.round(3).to_string(index=False))

    print("\n=== PER-SKU on TEST ===")
    per = results[results["split"].str.startswith("test:")].copy()
    per["sku"] = per["split"].str.replace("test:", "", regex=False)
    print(per[["sku", "model", "MAE", "RMSE", "WAPE_%"]].round(3).to_string(index=False))

    out = REPORTS_DIR / "baseline_metrics.csv"
    results.to_csv(out, index=False)
    print(f"\n[OK] saved -> {out}")

    print("\nInterpretation:")
    best_test = (
        agg[agg["split"] == "test"].sort_values("WAPE_%").iloc[0]
    )
    print(f"  Best baseline on test by WAPE: {best_test['model']} "
          f"(WAPE={best_test['WAPE_%']:.2f}%, MAE={best_test['MAE']:.2f})")
    print("  Any real ML model should beat this by a clear margin.")
