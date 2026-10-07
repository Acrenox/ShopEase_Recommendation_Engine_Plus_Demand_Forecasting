from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "forecasting"
OUTPUT_PATH = ROOT / "data" / "processed" / "forecasting_base.parquet"


def build_forecasting_base(output_path: str | Path | None = None) -> pd.DataFrame:
    sales_path = RAW_DIR / "sales_train_validation.csv"
    calendar_path = RAW_DIR / "calendar.csv"
    prices_path = RAW_DIR / "sell_prices.csv"

    if not sales_path.exists() or not calendar_path.exists() or not prices_path.exists():
        raise FileNotFoundError("Forecasting raw data files are missing: sales_train_validation.csv, calendar.csv, and sell_prices.csv")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    day_columns = [col for col in sales.columns if col.startswith("d_")]
    id_columns = [col for col in sales.columns if col not in day_columns]
    long_sales = sales.melt(id_vars=id_columns, value_vars=day_columns, var_name="day", value_name="sales")

    calendar = calendar[["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"]].copy()
    calendar = calendar.rename(columns={"d": "day"})
    long_sales = long_sales.merge(calendar, on="day", how="left")
    long_sales["date"] = pd.to_datetime(long_sales["date"])

    if "state_id" in long_sales.columns:
        long_sales = long_sales.rename(columns={"state_id": "state"})
    if "cat_id" in long_sales.columns:
        long_sales = long_sales.rename(columns={"cat_id": "category"})
    if "dept_id" in long_sales.columns:
        long_sales = long_sales.rename(columns={"dept_id": "department"})

    long_sales = long_sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    output = long_sales[[
        "date",
        "item_id",
        "store_id",
        "department",
        "category",
        "state",
        "sales",
        "sell_price",
        "weekday",
        "wday",
        "month",
        "year",
        "wm_yr_wk",
    ]].copy()

    output = output.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    if output_path is None:
        output_path = OUTPUT_PATH
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


if __name__ == "__main__":
    build_forecasting_base()
