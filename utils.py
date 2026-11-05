"""Small shared helpers."""
from __future__ import annotations

import pandas as pd
from config import ROOT_DIR, TARGET_COL

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def save_predictions(name: str, df: pd.DataFrame, pred_col: str) -> None:
    """Save date, sku, y_true, y_pred (rows without NaN) to reports/preds_<name>.csv."""
    sub = df.dropna(subset=[pred_col])[["date", "sku", TARGET_COL, pred_col]].copy()
    sub = sub.rename(columns={TARGET_COL: "y_true", pred_col: "y_pred"})
    out = REPORTS_DIR / f"preds_{name}.csv"
    sub.to_csv(out, index=False)
    print(f"[OK] saved predictions -> {out}  (rows: {len(sub)})")
