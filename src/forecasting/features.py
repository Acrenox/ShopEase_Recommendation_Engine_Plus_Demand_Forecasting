from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "data" / "processed" / "forecasting_base.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "forecasting_features.parquet"


def create_forecasting_features(base_path: str | Path | None = None, output_path: str | Path | None = None) -> pd.DataFrame:
    if base_path is None:
        base_path = BASE_PATH
    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError("Forecasting base table is missing. Run preprocessing first.")

    df = pd.read_parquet(base_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["quarter"] = df["date"].dt.quarter

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].transform(lambda s: s.shift(lag))

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df.groupby(["store_id", "item_id"])["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby(["store_id", "item_id"])["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
        )

    feature_table = df.dropna(subset=["sales"]).copy()
    if output_path is None:
        output_path = OUTPUT_PATH
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(output_path, index=False)
    return feature_table


if __name__ == "__main__":
    create_forecasting_features()
