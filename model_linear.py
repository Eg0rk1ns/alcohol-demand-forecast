"""
Section 1: linear regression.
Section 2: regularization - Ridge, Lasso.

All three are trained globally (all SKUs together, sku one-hot encoded).
Numeric features are standardized. Ridge/Lasso use sklearn's defaults
for solver and small alpha to avoid killing the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import mae, rmse, wape
from config import TARGET_COL
from features import (MODEL_FEATURES, LAG_FEATURES, ROLLING_FEATURES,
                      create_features, load_raw, train_val_test_split)
from utils import save_predictions


def get_xy(df: pd.DataFrame):
    clean = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy()
    X = clean[["sku"] + MODEL_FEATURES]
    y = clean[TARGET_COL]
    meta = clean[["date", "sku"]].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta


def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sku"]),
        ("num", StandardScaler(), MODEL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def evaluate(name: str, y: np.ndarray, p: np.ndarray, split: str) -> dict:
    return {
        "split": split, "model": name, "n": len(y),
        "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE_%": wape(y, p),
    }


if __name__ == "__main__":
    df = create_features(load_raw())
    train, val, test = train_val_test_split(df)
    X_train, y_train, meta_train = get_xy(train)
    X_val, y_val, meta_val = get_xy(val)
    X_test, y_test, meta_test = get_xy(test)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0, max_iter=5000),
    }

    rows = []
    for name, est in models.items():
        pipe = build_pipeline(est)
        pipe.fit(X_train, y_train)

        p_val = pipe.predict(X_val)
        p_test = pipe.predict(X_test)

        rows.append(evaluate(name, y_val.values, p_val, "val"))
        rows.append(evaluate(name, y_test.values, p_test, "test"))

        # save test predictions in the standard format for later ensemble
        test_eval = meta_test.copy()
        test_eval[TARGET_COL] = y_test.values
        test_eval[f"pred_{name}"] = p_test
        save_predictions(name, test_eval, f"pred_{name}")

    res = pd.DataFrame(rows)
    print("\n=== LINEAR / REGULARIZED MODELS ===")
    print(res.round(3).to_string(index=False))

    print("\nTop-5 coefficients (absolute, Ridge):")
    ridge_pipe = build_pipeline(Ridge(alpha=1.0)).fit(X_train, y_train)
    pre = ridge_pipe.named_steps["pre"]
    feat_names = list(pre.named_transformers_["cat"].get_feature_names_out(["sku"])) + MODEL_FEATURES
    coefs = ridge_pipe.named_steps["model"].coef_
    top = sorted(zip(feat_names, coefs), key=lambda t: abs(t[1]), reverse=True)[:5]
    for n, c in top:
        print(f"  {n:30s} {c:+.4f}")
