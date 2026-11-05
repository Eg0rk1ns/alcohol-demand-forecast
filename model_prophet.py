"""
Section 3: Prophet per-SKU with external regressors.

For each SKU:
  - fit Prophet on train,
  - built-in RU holidays,
  - promo_flag, discount, price as extra regressors,
  - predict val and test (using known future exog values).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

from baseline import mae, rmse, wape
from config import TARGET_COL
from features import load_raw, train_val_test_split
from utils import save_predictions

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

EXOG = ["promo_flag", "discount", "price"]


def fit_predict_prophet(train_df: pd.DataFrame, predict_df: pd.DataFrame, sku: str):
    tr = (train_df[train_df["sku"] == sku]
          [["date", TARGET_COL] + EXOG]
          .rename(columns={"date": "ds", TARGET_COL: "y"})
          .sort_values("ds"))
    pr = (predict_df[predict_df["sku"] == sku]
          [["date"] + EXOG]
          .rename(columns={"date": "ds"})
          .sort_values("ds"))

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays(country_name="RU")
    for col in EXOG:
        m.add_regressor(col)
    m.fit(tr)
    fc = m.predict(pr)
    return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values


def evaluate(name, y, p, split):
    return {"split": split, "model": name, "n": len(y),
            "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE_%": wape(y, p)}


if __name__ == "__main__":
    df = load_raw()
    train, val, test = train_val_test_split(df)
    skus = sorted(df["sku"].unique())

    print(f"Training Prophet for {len(skus)} SKUs...")
    val_pred = np.full(len(val), np.nan)
    test_pred = np.full(len(test), np.nan)

    for sku in skus:
        yhat_val, lo_val, hi_val = fit_predict_prophet(train, val, sku)
        val_mask = (val["sku"] == sku).values
        val_pred[val_mask] = yhat_val

        yhat_test, lo_test, hi_test = fit_predict_prophet(train, test, sku)
        test_mask = (test["sku"] == sku).values
        test_pred[test_mask] = yhat_test
        print(f"  [{sku}] done")

    val = val.copy(); val["pred_prophet"] = val_pred
    test = test.copy(); test["pred_prophet"] = test_pred

    rows = [
        evaluate("prophet", val[TARGET_COL].values, val_pred, "val"),
        evaluate("prophet", test[TARGET_COL].values, test_pred, "test"),
    ]
    res = pd.DataFrame(rows)
    print("\n=== PROPHET ===")
    print(res.round(3).to_string(index=False))

    save_predictions("prophet", test[["date", "sku", TARGET_COL, "pred_prophet"]],
                     "pred_prophet")
