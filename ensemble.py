"""
Section 7: Ensemble.

Weighted average of models. Weights are set as 1/WAPE_val normalized,
so models that were better on validation get more weight. No peeking
into test WAPE.

Two ensembles compared:
  - ensemble_all    : all models weighted by 1/WAPE_val
  - ensemble_top3   : only lgbm_global + prophet + lstm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import mae, rmse, wape
from utils import save_predictions

# Validation WAPEs we obtained from earlier runs (used as weights)
VAL_WAPE = {
    "lgbm_global": 11.385,
    "prophet": 18.575,
    "ridge": 19.505,
    "linear": 19.708,
    "bayesian": 19.638,
    "lasso": 21.054,
    "lstm": 21.83,
}


def load_all_preds():
    frames = {}
    for name in VAL_WAPE:
        try:
            df = pd.read_csv(f"reports/preds_{name}.csv", parse_dates=["date"])
            df = df.rename(columns={"y_pred": f"pred_{name}"})
            df = df[["date", "sku", "y_true", f"pred_{name}"]]
            frames[name] = df
        except FileNotFoundError:
            print(f"[skip] preds_{name}.csv not found")
    # inner merge
    base = None
    for df in frames.values():
        if base is None:
            base = df
        else:
            base = base.merge(df.drop(columns=["y_true"]), on=["date", "sku"], how="inner")
    return base, list(frames)


def inverse_wape_weights(models):
    inv = {m: 1.0 / VAL_WAPE[m] for m in models}
    total = sum(inv.values())
    return {m: inv[m] / total for m in models}


if __name__ == "__main__":
    df, models = load_all_preds()
    y = df["y_true"].values

    print(f"Merged rows: {len(df)}, models: {models}")

    # ---- All models, inverse-WAPE weights ----
    w_all = inverse_wape_weights(models)
    pred_all = sum(df[f"pred_{m}"].values * w_all[m] for m in models)
    df["pred_ensemble_all"] = pred_all

    # ---- Top-3 models ----
    top3 = ["lgbm_global", "prophet", "lstm"]
    top3 = [m for m in top3 if m in models]
    w_top3 = inverse_wape_weights(top3)
    pred_top3 = sum(df[f"pred_{m}"].values * w_top3[m] for m in top3)
    df["pred_ensemble_top3"] = pred_top3

    rows = []
    for name in models:
        rows.append({"model": name,
                     "MAE": mae(y, df[f"pred_{name}"].values),
                     "RMSE": rmse(y, df[f"pred_{name}"].values),
                     "WAPE_%": wape(y, df[f"pred_{name}"].values)})
    for ens_name in ["ensemble_all", "ensemble_top3"]:
        rows.append({"model": ens_name,
                     "MAE": mae(y, df[f"pred_{ens_name}"].values),
                     "RMSE": rmse(y, df[f"pred_{ens_name}"].values),
                     "WAPE_%": wape(y, df[f"pred_{ens_name}"].values)})

    res = pd.DataFrame(rows).sort_values("WAPE_%").reset_index(drop=True)
    print("\n=== ALL MODELS + ENSEMBLES (on test) ===")
    print(res.round(3).to_string(index=False))

    print("\nEnsemble weights (all):")
    for m, w in w_all.items():
        print(f"  {m:15s} {w:.3f}")
    print("\nEnsemble weights (top3):")
    for m, w in w_top3.items():
        print(f"  {m:15s} {w:.3f}")

    # save the best ensemble for later use in forecast.py
    out = df[["date", "sku", "y_true", "pred_ensemble_all", "pred_ensemble_top3"]].copy()
    out.to_csv("reports/preds_ensemble.csv", index=False)
    print("\n[OK] saved -> reports/preds_ensemble.csv")
