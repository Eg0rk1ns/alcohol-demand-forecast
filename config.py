"""Central configuration for the demand forecasting lab."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SALES_CSV = DATA_DIR / "sales_data.csv"

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Data generation ---
N_SKUS = 6
DATE_START = "2022-01-01"
DATE_END = "2024-12-31"

# --- Forecasting ---
HORIZON_DAYS_DEFAULT = 30
TARGET_COL = "sales"

# --- Chronological split (train / val / test) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test_frac = 1 - TRAIN_FRAC - VAL_FRAC

# --- Feature engineering ---
LAGS = [7, 14, 30]
ROLLING_WINDOWS = [7, 28]

CALENDAR_FEATURES = [
    "day_of_week", "month", "day_of_year", "is_weekend", "holiday_flag",
]
EXOG_FEATURES = [
    "price", "competitor_price", "discount",
    "promo_flag", "ad_budget", "oos_flag",
]
