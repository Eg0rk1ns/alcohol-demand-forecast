"""
Section 4: LSTM per-SKU.

Design:
  - one LSTM per SKU (classic sequence modeling);
  - input window: SEQ_LEN = 28 previous days;
  - features per day: [sales_scaled, promo_flag, discount,
                      price_scaled, competitor_price_scaled,
                      dow_sin, dow_cos, month_sin, month_cos, holiday_flag];
  - LSTM(32) -> Dense(1);
  - training on train split, early stop on val split;
  - evaluation: one-step-ahead prediction over val and test
    (each window uses true historical values; this matches the lab
    task framing and keeps the comparison with the other models fair).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import mae, rmse, wape
from config import RANDOM_SEED, TARGET_COL
from features import load_raw, train_val_test_split
from utils import save_predictions

SEQ_LEN = 28
DEVICE = "cpu"
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    df["day_of_week"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_windows(df_sku: pd.DataFrame, feat_cols, target_col,
                 seq_len: int, start_idx: int = SEQ_LEN):
    """Yield (X_window, y) for every index >= start_idx."""
    X = df_sku[feat_cols].values
    y = df_sku[target_col].values
    n = len(df_sku)
    if n <= start_idx:
        return np.empty((0, seq_len, len(feat_cols))), np.empty(0)
    Xs, ys = [], []
    for i in range(start_idx, n):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_one_sku(df_sku: pd.DataFrame, feat_cols, y_scaler_mean, y_scaler_std):
    df_sku = df_sku.reset_index(drop=True).copy()
    # normalize target inside features
    df_sku["sales_scaled"] = (df_sku[TARGET_COL] - y_scaler_mean) / y_scaler_std
    cols = ["sales_scaled"] + feat_cols
    n = len(df_sku)

    # index boundaries correspond to the split of the original (unshifted) rows
    # we need the same chronological cut points as in train_val_test_split
    # so we recompute them from df_sku dates
    return df_sku, cols


def predict_windows(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()


def run():
    raw = load_raw().sort_values(["sku", "date"]).reset_index(drop=True)
    df = build_features(raw)
    train, val, test = train_val_test_split(df)

    # global target scaler across train
    y_mean = float(train[TARGET_COL].mean())
    y_std = float(train[TARGET_COL].std() + 1e-8)

    # feature scaling (simple per-column z-score on train)
    base_feat = ["promo_flag", "discount", "price", "competitor_price",
                 "dow_sin", "dow_cos", "month_sin", "month_cos", "holiday_flag"]
    feat_mean = train[base_feat].mean()
    feat_std = train[base_feat].std().replace(0, 1)

    def prep(frame):
        f = frame.copy()
        f["sales_scaled"] = (f[TARGET_COL] - y_mean) / y_std
        for c in base_feat:
            f[c] = (f[c] - feat_mean[c]) / feat_std[c]
        return f

    train_p, val_p, test_p = prep(train), prep(val), prep(test)
    feat_cols = ["sales_scaled"] + base_feat

    # per-SKU train / predict (one-step ahead on val and test)
    val_pred = np.full(len(val), np.nan)
    test_pred = np.full(len(test), np.nan)

    for sku in sorted(df["sku"].unique()):
        tr = train_p[train_p["sku"] == sku].reset_index(drop=True)
        va = val_p[val_p["sku"] == sku].reset_index(drop=True)
        te = test_p[test_p["sku"] == sku].reset_index(drop=True)

        # training windows: from train only
        X_tr, y_tr = make_windows(tr, feat_cols, "sales_scaled", SEQ_LEN)
        # val windows: use train tail + val (true) to allow one-step prediction
        tr_va = pd.concat([tr, va], ignore_index=True)
        X_va_all, _ = make_windows(tr_va, feat_cols, "sales_scaled", SEQ_LEN)
        X_va = X_va_all[len(X_tr):len(X_tr) + len(va)] if len(X_va_all) >= len(X_tr) + len(va) else None

        # test windows: train tail + val + test
        tr_va_te = pd.concat([tr, va, te], ignore_index=True)
        X_te_all, _ = make_windows(tr_va_te, feat_cols, "sales_scaled", SEQ_LEN)
        start_te = len(X_tr) + len(va)
        X_te = X_te_all[start_te:start_te + len(te)] if len(X_te_all) >= start_te + len(te) else None

        # validation split for early stopping: 15% tail of train windows
        n_val_es = max(30, int(0.15 * len(X_tr)))
        X_es_tr, y_es_tr = X_tr[:-n_val_es], y_tr[:-n_val_es]
        X_es_va, y_es_va = X_tr[-n_val_es:], y_tr[-n_val_es:]

        model = LSTMRegressor(n_features=len(feat_cols), hidden=32).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_es_tr), torch.from_numpy(y_es_tr))
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        best_val, best_state, patience, bad = np.inf, None, 5, 0
        for epoch in range(40):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb.to(DEVICE))
                loss = loss_fn(pred, yb.to(DEVICE))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vp = model(torch.from_numpy(X_es_va).to(DEVICE)).cpu().numpy()
            vloss = float(np.mean((vp - y_es_va) ** 2))
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # one-step-ahead predictions
        if X_va is not None:
            pv = predict_windows(model, X_va) * y_std + y_mean
            val_pred[(val["sku"] == sku).values] = pv
        if X_te is not None:
            pt = predict_windows(model, X_te) * y_std + y_mean
            test_pred[(test["sku"] == sku).values] = pt

        print(f"  [{sku}] best_val_mse={best_val:.4f}")

    # evaluate
    m_va = ~np.isnan(val_pred)
    m_te = ~np.isnan(test_pred)
    rows = [
        {"split": "val", "model": "lstm", "n": int(m_va.sum()),
         "MAE": mae(val[TARGET_COL].values[m_va], val_pred[m_va]),
         "RMSE": rmse(val[TARGET_COL].values[m_va], val_pred[m_va]),
         "WAPE_%": wape(val[TARGET_COL].values[m_va], val_pred[m_va])},
        {"split": "test", "model": "lstm", "n": int(m_te.sum()),
         "MAE": mae(test[TARGET_COL].values[m_te], test_pred[m_te]),
         "RMSE": rmse(test[TARGET_COL].values[m_te], test_pred[m_te]),
         "WAPE_%": wape(test[TARGET_COL].values[m_te], test_pred[m_te])},
    ]
    print("\n=== LSTM ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    test_out = test[["date", "sku", TARGET_COL]].copy()
    test_out["pred_lstm"] = test_pred
    save_predictions("lstm", test_out, "pred_lstm")


if __name__ == "__main__":
    run()
