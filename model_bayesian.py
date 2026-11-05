"""
Section 5: Bayesian Ridge with prediction intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import mae, rmse, wape
from config import TARGET_COL
from features import (LAG_FEATURES, MODEL_FEATURES, ROLLING_FEATURES,
                      create_features, load_raw, train_val_test_split)
from utils import save_predictions

Z90 = 1.645


def get_xy(df: pd.DataFrame):
    clean = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy()
    X = clean[["sku"] + MODEL_FEATURES].reset_index(drop=True)
    y = clean[TARGET_COL].reset_index(drop=True)
    meta = clean[["date", "sku"]].reset_index(drop=True)
    return X, y, meta


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sku"]),
        ("num", StandardScaler(), MODEL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", BayesianRidge())])


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean() * 100.0)


if __name__ == "__main__":
    df = create_features(load_raw())
    train, val, test = train_val_test_split(df)
    X_tr, y_tr, _ = get_xy(train)
    X_va, y_va, meta_va = get_xy(val)
    X_te, y_te, meta_te = get_xy(test)

    pipe = build_pipeline().fit(X_tr, y_tr)
    p_va, s_va = pipe.predict(X_va, return_std=True)
    p_te, s_te = pipe.predict(X_te, return_std=True)

    lo_va, hi_va = p_va - Z90 * s_va, p_va + Z90 * s_va
    lo_te, hi_te = p_te - Z90 * s_te, p_te + Z90 * s_te

    rows = [
        {"split": "val", "model": "bayesian_ridge", "n": len(y_va),
         "MAE": mae(y_va.values, p_va), "RMSE": rmse(y_va.values, p_va),
         "WAPE_%": wape(y_va.values, p_va),
         "P10-P90 coverage_%": coverage(y_va.values, lo_va, hi_va)},
        {"split": "test", "model": "bayesian_ridge", "n": len(y_te),
         "MAE": mae(y_te.values, p_te), "RMSE": rmse(y_te.values, p_te),
         "WAPE_%": wape(y_te.values, p_te),
         "P10-P90 coverage_%": coverage(y_te.values, lo_te, hi_te)},
    ]
    print("=== BAYESIAN RIDGE ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    out = meta_te.copy()
    out["sales"] = y_te.values
    out["pred_bayesian"] = p_te
    out["p10"] = lo_te
    out["p90"] = hi_te
    save_predictions("bayesian", out, "pred_bayesian")
    out.to_csv("reports/preds_bayesian_with_intervals.csv", index=False)
    print("[OK] intervals saved -> reports/preds_bayesian_with_intervals.csv")
