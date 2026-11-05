"""
Final summary across all models + baseline.

Produces:
  - reports/summary_all.csv  (MAE, RMSE, WAPE on val and test for every model)
  - reports/summary_bar.png  (bar chart of test WAPE by model)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import mae, rmse, wape
from config import ROOT_DIR, TARGET_COL

REPORTS_DIR = ROOT_DIR / "reports"


def add_row(rows, name, y, p, split):
    rows.append({"model": name, "split": split, "n": len(y),
                 "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE_%": wape(y, p)})


def main():
    # ---- baseline ----
    rows = []
    try:
        b = pd.read_csv(REPORTS_DIR / "baseline_metrics.csv")
        b = b[~b["split"].str.contains(":")]
        for _, r in b.iterrows():
            rows.append({"model": r["model"] + " (baseline)",
                         "split": r["split"], "n": int(r["n"]),
                         "MAE": r["MAE"], "RMSE": r["RMSE"], "WAPE_%": r["WAPE_%"]})
    except FileNotFoundError:
        pass

    # ---- prediction files ----
    pred_map = {
        "linear": "linear",
        "ridge": "ridge",
        "lasso": "lasso",
        "lgbm_global": "lgbm_global",
        "lgbm_per_sku": "lgbm_per_sku",
        "prophet": "prophet",
        "lstm": "lstm",
        "bayesian": "bayesian",
        "ensemble_all": "ensemble_all",
        "ensemble_top3": "ensemble_top3",
    }
    ens = pd.read_csv(REPORTS_DIR / "preds_ensemble.csv", parse_dates=["date"])
    for name, col in pred_map.items():
        if name.startswith("ensemble"):
            sub = ens.dropna(subset=[f"pred_{col}"])
            add_row(rows, name, sub["y_true"].values,
                    sub[f"pred_{col}"].values, "test")
            continue
        path = REPORTS_DIR / f"preds_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if "y_true" in df and "y_pred" in df:
            add_row(rows, name, df["y_true"].values, df["y_pred"].values, "test")

    out = pd.DataFrame(rows)
    out = out.sort_values(["split", "WAPE_%"]).reset_index(drop=True)
    out.to_csv(REPORTS_DIR / "summary_all.csv", index=False)

    print("=== SUMMARY (all models) ===")
    print(out.round(3).to_string(index=False))

    # ---- bar chart on test ----
    test_df = out[out["split"] == "test"].sort_values("WAPE_%", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["tab:gray" if "baseline" in m else
              ("tab:red" if "ensemble" in m else "tab:blue")
              for m in test_df["model"]]
    ax.barh(test_df["model"], test_df["WAPE_%"], color=colors)
    for i, v in enumerate(test_df["WAPE_%"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("WAPE on test, % (lower is better)")
    ax.set_title("Model comparison: test WAPE")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "summary_bar.png", dpi=130)
    print(f"\n[OK] saved -> reports/summary_all.csv, reports/summary_bar.png")


if __name__ == "__main__":
    main()
