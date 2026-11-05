"""
Synthetic generator of alcohol sales data.

Business effects baked in:
- long-term trend;
- weekly seasonality (Fri/Sat peaks);
- yearly seasonality (December peak, summer dip);
- RF holidays (New Year, 23 Feb, 8 Mar, 1 May, 9 May, 12 Jun, 4 Nov);
- promo weeks (spike + post-promo dip);
- price elasticity vs competitor price;
- OOS days (sales=0, but demand exists);
- random noise.

Output: data/sales_data.csv
Columns: date, sku, category, sales, price, competitor_price,
         discount, promo_flag, ad_budget, oos_flag, holiday_flag
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATE_END, DATE_START, N_SKUS, RANDOM_SEED, SALES_CSV,
)

HOLIDAY_MMDD = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
]


def _build_holiday_set(years):
    holidays = set()
    for y in years:
        for m, d in HOLIDAY_MMDD:
            try:
                holidays.add(pd.Timestamp(year=y, month=m, day=d).date())
            except ValueError:
                continue
    return holidays


def _make_sku_meta(n_skus, rng):
    categories = ["wine", "spirits"]
    skus = []
    for i in range(n_skus):
        cat = categories[i % len(categories)]
        if cat == "wine":
            base_price = rng.uniform(400, 1500)
            base_sales = rng.uniform(25, 80)
        else:
            base_price = rng.uniform(800, 3000)
            base_sales = rng.uniform(10, 40)
        skus.append({
            "sku": f"SKU_{i + 1:02d}",
            "category": cat,
            "base_price": round(float(base_price), 2),
            "base_sales": round(float(base_sales), 2),
        })
    return skus


def _promo_index(n_days, rng, promo_len=7):
    n_years = max(n_days / 365.25, 1.0)
    n_promos = max(1, int(round(n_years * rng.integers(4, 7))))
    starts = rng.choice(n_days - promo_len, size=n_promos, replace=False)
    mask = np.zeros(n_days, dtype=int)
    for s in starts:
        mask[s:s + promo_len] = 1
    return mask, starts


def generate_sales_data(
    start=DATE_START,
    end=DATE_END,
    n_skus=N_SKUS,
    seed=RANDOM_SEED,
    out_path=SALES_CSV,
):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    years = range(dates[0].year, dates[-1].year + 1)
    holiday_set = _build_holiday_set(years)
    holiday_flag = np.array([1 if d.date() in holiday_set else 0 for d in dates])

    skus = _make_sku_meta(n_skus, rng)

    weekly_lookup = np.array([0.90, 0.90, 0.95, 1.00, 1.20, 1.35, 1.05])
    weekly = weekly_lookup[dates.dayofweek]

    doy = dates.dayofyear.values
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (doy - 355) / 365)
    dec_boost = np.where(dates.month == 12, 1.40, 1.0)
    mar_boost = np.where((dates.month == 3) & (dates.day.values <= 10), 1.25, 1.0)
    may_boost = np.where((dates.month == 5) & (dates.day.values <= 10), 1.15, 1.0)
    holiday_boost = np.where(holiday_flag == 1, 1.20, 1.0)

    frames = []
    for meta in skus:
        sku = meta["sku"]
        cat = meta["category"]
        base_price = meta["base_price"]
        base_sales = meta["base_sales"]

        t = np.arange(n_days)
        trend = 1 + 0.00005 * t

        promo_flag, promo_starts = _promo_index(n_days, rng)
        promo_mult = np.where(promo_flag == 1, rng.uniform(1.6, 2.0, n_days), 1.0)

        post_dip = np.ones(n_days)
        for s in promo_starts:
            end_idx = s + 7
            post_dip[end_idx:end_idx + 5] *= 0.85

        price = base_price * (1 + 0.00005 * t) * (1 + rng.normal(0, 0.02, n_days))
        discount = np.where(promo_flag == 1, rng.uniform(0.15, 0.25, n_days), 0.0)
        price = np.maximum(price * (1 - discount), 1.0)

        comp_price = np.maximum(base_price * (1 + rng.normal(0, 0.05, n_days)), 1.0)

        rel_price = price / comp_price
        elasticity = np.exp(-1.5 * (rel_price - 1))

        demand = (base_sales * trend * weekly * yearly * dec_boost * mar_boost
                  * may_boost * holiday_boost * promo_mult * post_dip * elasticity)
        demand = np.maximum(demand, 0)

        sales_float = demand * (1 + rng.normal(0, 0.08, n_days))
        sales_float = np.maximum(sales_float, 0)

        oos_flag = (rng.random(n_days) < 0.02).astype(int)
        for _ in range(rng.integers(1, 4)):
            s = int(rng.integers(0, n_days - 10))
            ln = int(rng.integers(3, 8))
            oos_flag[s:s + ln] = 1

        sales = np.round(sales_float).astype(int)
        sales = np.where(oos_flag == 1, 0, sales)

        ad_budget = np.where(
            promo_flag == 1,
            rng.uniform(2000, 5000, n_days),
            rng.uniform(0, 500, n_days),
        )

        frames.append(pd.DataFrame({
            "date": dates,
            "sku": sku,
            "category": cat,
            "sales": sales,
            "price": np.round(price, 2),
            "competitor_price": np.round(comp_price, 2),
            "discount": np.round(discount, 3),
            "promo_flag": promo_flag,
            "ad_budget": np.round(ad_budget, 2),
            "oos_flag": oos_flag,
            "holiday_flag": holiday_flag,
        }))

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["sku", "date"])
        .reset_index(drop=True)
    )
    df.to_csv(out_path, index=False)

    print(f"[OK] Saved {len(df):,} rows -> {out_path}")
    print(f"     date range : {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"     SKUs       : {sorted(df['sku'].unique())}")
    print(f"     categories : {sorted(df['category'].unique())}")
    print(f"     total sales: {df['sales'].sum():,}")
    print("\nPer-SKU summary:")
    print(
        df.groupby(["sku", "category"])
        .agg(days=("date", "count"),
             mean_sales=("sales", "mean"),
             total_sales=("sales", "sum"),
             oos_days=("oos_flag", "sum"),
             promo_days=("promo_flag", "sum"))
        .round(2)
        .to_string()
    )
    return df


if __name__ == "__main__":
    generate_sales_data()
