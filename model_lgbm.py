"""
Section 1: Gradient Boosting (LightGBM).
Section 3*: feature-based ML (lag + rolling features).

Two flavors:
  - global  : one model for all SKUs, sku treated as a categorical feature
              (this also covers Section 6 cross-sectional approach qualitatively)
  - per_sku : six separate models (industrial pattern for "specific SKU")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from baseline import mae, rmse, wape
from config import RANDOM_SEED, TARGET_COL
from features import (LAG_FEATURES, MODEL_FEATURES, ROLLING_FEATURES,
                      create_features, load_raw, train_val_test_split)
from utils import save_predictions

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


def get_xy(df: pd.DataFrame):
    clean = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy()
    X = clean[["sku"] + MODEL_FEATURES].copy()
    X["sku"] = X["sku"].astype("category")
    y = clean[TARGET_COL]
    meta = clean[["date", "sku"]].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta


def evaluate(name, y, p, split):
    return {
        "split": split, "model": name, "n": len(y),
        "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE_%": wape(y, p),
    }


def train_global(X_train, y_train, X_val, y_val):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=["sku"])
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        PARAMS, dtrain, num_boost_round=N_ROUNDS, valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )
    return model


def train_per_sku(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Returns dict: sku -> booster."""
    models = {}
    for sku in sorted(train_df["sku"].unique()):
        tr = train_df[train_df["sku"] == sku].sort_values("date")
        va = val_df[val_df["sku"] == sku].sort_values("date")
        tr = tr.dropna(subset=LAG_FEATURES + ROLLING_FEATURES)
        va = va.dropna(subset=LAG_FEATURES + ROLLING_FEATURES)
        if len(tr) < 100 or len(va) < 10:
            continue
        X_tr = tr[MODEL_FEATURES]
        X_va = va[MODEL_FEATURES]
        dtrain = lgb.Dataset(X_tr, label=tr[TARGET_COL])
        dval = lgb.Dataset(X_va, label=va[TARGET_COL], reference=dtrain)
        m = lgb.train(
            PARAMS, dtrain, num_boost_round=N_ROUNDS, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
        )
        models[sku] = m
    return models


def predict_per_sku(models: dict, df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=float)
    for sku, m in models.items():
        mask = (df["sku"] == sku).values
        if mask.sum() == 0:
            continue
        X = df.loc[mask, MODEL_FEATURES]
        out[mask] = m.predict(X)
    return out


if __name__ == "__main__":
    feat = create_features(load_raw())
    train, val, test = train_val_test_split(feat)

    # ---- global ----
    X_train, y_train, _ = get_xy(train)
    X_val, y_val, _ = get_xy(val)
    X_test, y_test, meta_test = get_xy(test)

    gmodel = train_global(X_train, y_train, X_val, y_val)
    print(f"[global] best_iteration = {gmodel.best_iteration}")

    p_val_g = gmodel.predict(X_val, num_iteration=gmodel.best_iteration)
    p_test_g = gmodel.predict(X_test, num_iteration=gmodel.best_iteration)

    # ---- per-SKU ----
    print("[per-sku] training 6 models...")
    sku_models = train_per_sku(train, val)
    p_val_s = predict_per_sku(sku_models, val.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy())
    p_test_s = predict_per_sku(sku_models, test.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).copy())

    # drop NaN rows to align with y
    val_clean = val.dropna(subset=LAG_FEATURES + ROLLING_FEATURES)
    test_clean = test.dropna(subset=LAG_FEATURES + ROLLING_FEATURES)

    rows = [
        evaluate("lgbm_global", y_val.values, p_val_g, "val"),
        evaluate("lgbm_global", y_test.values, p_test_g, "test"),
        evaluate("lgbm_per_sku", val_clean[TARGET_COL].values, p_val_s, "val"),
        evaluate("lgbm_per_sku", test_clean[TARGET_COL].values, p_test_s, "test"),
    ]
    res = pd.DataFrame(rows)
    print("\n=== LIGHTGBM ===")
    print(res.round(3).to_string(index=False))

    # save test predictions
    test_eval = meta_test.copy()
    test_eval["sales"] = y_test.values
    test_eval["pred_lgbm_global"] = p_test_g
    save_predictions("lgbm_global", test_eval, "pred_lgbm_global")

    test_eval2 = test_clean[["date", "sku", TARGET_COL]].reset_index(drop=True).copy()
    test_eval2["pred_lgbm_per_sku"] = p_test_s
    save_predictions("lgbm_per_sku", test_eval2, "pred_lgbm_per_sku")

    # feature importance (global)
    print("\nTop-10 features by gain (global model):")
    imp = pd.DataFrame({
        "feature": gmodel.feature_name(),
        "gain": gmodel.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(10)
    print(imp.round(0).to_string(index=False))
