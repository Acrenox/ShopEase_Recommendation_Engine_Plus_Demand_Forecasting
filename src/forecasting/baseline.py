from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "data" / "processed" / "forecasting_base.parquet"


def generate_baselines(base_path: str | Path | None = None, horizon: int = 28) -> pd.DataFrame:
    if base_path is None:
        base_path = BASE_PATH
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError("Forecasting base data is missing. Run preprocessing first.")

    df = pd.read_parquet(base_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    rows = []
    for (store_id, item_id), group in df.groupby(["store_id", "item_id"]):
        group = group.sort_values("date").reset_index(drop=True)
        last_date = group["date"].max()
        for offset in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=offset)
            naive_value = float(group["sales"].iloc[-1])
            seasonal_value = float(group["sales"].iloc[-7]) if len(group) >= 7 else naive_value
            rows.append(
                {
                    "date": future_date,
                    "store_id": store_id,
                    "item_id": item_id,
                    "naive_forecast": naive_value,
                    "seasonal_naive_forecast": seasonal_value,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    generate_baselines()
