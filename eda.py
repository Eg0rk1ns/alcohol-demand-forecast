"""
Exploratory data analysis.

Saves figures to reports/:
  - eda_total_sales.png
  - eda_one_sku.png
  - eda_by_weekday.png
  - eda_by_month.png
  - eda_promo_effect.png
  - eda_sales_distribution.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ROOT_DIR
from features import create_features, load_raw

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
FOCUS_SKU = "SKU_01"


def plot_total_sales(df: pd.DataFrame) -> None:
    total = df.groupby("date")["sales"].sum()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(total.index, total.values, color="tab:blue", lw=0.9)
    ax.set_title("Total daily sales across all SKUs")
    ax.set_xlabel("date")
    ax.set_ylabel("units sold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_total_sales.png", dpi=130)
    plt.close(fig)


def plot_one_sku(df: pd.DataFrame, sku: str = FOCUS_SKU) -> None:
    sub = df[df["sku"] == sku].sort_values("date")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sub["date"], sub["sales"], color="tab:gray", lw=0.7, label="sales")
    promo = sub[sub["promo_flag"] == 1]
    ax.scatter(promo["date"], promo["sales"], s=10, color="tab:red", label="promo")
    oos = sub[sub["oos_flag"] == 1]
    ax.scatter(oos["date"], oos["sales"], s=14, color="tab:orange", marker="x", label="OOS")
    ax.set_title(f"{sku}: daily sales (promo highlighted, OOS marked)")
    ax.set_xlabel("date")
    ax.set_ylabel("units sold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_one_sku.png", dpi=130)
    plt.close(fig)


def plot_by_weekday(df: pd.DataFrame) -> None:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grp = df.groupby("day_of_week")["sales"].mean().reindex(range(7))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, grp.values, color="tab:blue")
    ax.set_title("Average sales by day of week")
    ax.set_ylabel("avg units")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_by_weekday.png", dpi=130)
    plt.close(fig)


def plot_by_month(df: pd.DataFrame) -> None:
    grp = df.groupby("month")["sales"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grp.index, grp.values, color="tab:green")
    ax.set_xticks(range(1, 13))
    ax.set_title("Average sales by month")
    ax.set_xlabel("month")
    ax.set_ylabel("avg units")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_by_month.png", dpi=130)
    plt.close(fig)


def plot_promo_effect(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    df_copy = df.copy()
    df_copy["promo"] = df_copy["promo_flag"].map({0: "no promo", 1: "promo"})
    sns.barplot(data=df_copy, x="promo", y="sales", errorbar="se", ax=ax)
    ax.set_title("Mean sales: promo vs non-promo")
    ax.set_ylabel("units sold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_promo_effect.png", dpi=130)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["sales"], bins=60, ax=ax, color="tab:purple")
    ax.set_title("Distribution of daily sales")
    ax.set_xlabel("units sold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_sales_distribution.png", dpi=130)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print("=== DATASET SUMMARY ===")
    print(f"rows: {len(df):,}")
    print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"skus: {df['sku'].nunique()}, categories: {sorted(df['category'].unique())}\n")

    print("Per-SKU:")
    print(df.groupby(["sku", "category"])
            .agg(mean_sales=("sales", "mean"),
                 std_sales=("sales", "std"),
                 pct_zero=("sales", lambda s: float((s == 0).mean()) * 100),
                 oos_days=("oos_flag", "sum"),
                 promo_days=("promo_flag", "sum"))
            .round(2).to_string())

    print("\nShare of zero-sales days explained by OOS:")
    zero_mask = df["sales"] == 0
    zero_total = int(zero_mask.sum())
    zero_oos = int((zero_mask & (df["oos_flag"] == 1)).sum())
    print(f"  zero days total: {zero_total}")
    print(f"  of which OOS   : {zero_oos}  ({zero_oos / max(zero_total, 1) * 100:.1f}%)")
    print(f"  other zeros    : {zero_total - zero_oos}")

    print("\nSeasonality check (avg sales by month):")
    print(df.groupby("month")["sales"].mean().round(2).to_string())

    print("\nWeekday check (avg sales by DOW, 0=Mon):")
    print(df.groupby("day_of_week")["sales"].mean().round(2).to_string())


if __name__ == "__main__":
    raw = load_raw()
    feat = create_features(raw)

    print_summary(feat)

    plot_total_sales(feat)
    plot_one_sku(feat)
    plot_by_weekday(feat)
    plot_by_month(feat)
    plot_promo_effect(feat)
    plot_distribution(feat)

    print(f"\n[OK] figures saved to {REPORTS_DIR}")
