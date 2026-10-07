from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_RECO = ROOT / "data" / "raw" / "recommendation"
RAW_FORECAST = ROOT / "data" / "raw" / "forecasting"


def ensure_demo_data() -> None:
    RAW_RECO.mkdir(parents=True, exist_ok=True)
    RAW_FORECAST.mkdir(parents=True, exist_ok=True)

    events_path = RAW_RECO / "events.csv"
    props_1_path = RAW_RECO / "item_properties_part1.csv"
    props_2_path = RAW_RECO / "item_properties_part2.csv"
    category_tree_path = RAW_RECO / "category_tree.csv"

    if not events_path.exists():
        rng = np.random.default_rng(42)
        users = list(range(1, 51))
        items = list(range(1, 31))
        event_types = ["view", "addtocart", "transaction"]

        times = pd.date_range("2024-01-01", periods=365, freq="h")
        event_rows = []
        for _ in range(900):
            user_id = int(rng.choice(users))
            item_id = int(rng.choice(items))
            event_type = rng.choice(event_types, p=[0.7, 0.2, 0.1])
            timestamp = int(pd.Timestamp(rng.choice(times)).value / 1_000_000)
            event_rows.append({"user_id": user_id, "item_id": item_id, "event_type": event_type, "timestamp": timestamp})
        pd.DataFrame(event_rows).to_csv(events_path, index=False)

        category_rows = [
            {"category_id": 1, "parent_id": 0, "category_name": "Electronics"},
            {"category_id": 2, "parent_id": 0, "category_name": "Home"},
            {"category_id": 3, "parent_id": 0, "category_name": "Fashion"},
        ]
        pd.DataFrame(category_rows).to_csv(category_tree_path, index=False)

        property_rows = []
        for item_id in items:
            category_id = 1 + ((item_id - 1) % 3)
            property_rows.append({"item_id": item_id, "property": "category_id", "value": category_id})
            property_rows.append({"item_id": item_id, "property": "brand", "value": f"brand_{(item_id % 7) + 1}"})
            property_rows.append({"item_id": item_id, "property": "color", "value": ["red", "blue", "green", "black"][item_id % 4]})
        pd.DataFrame(property_rows[:len(property_rows)//2]).to_csv(props_1_path, index=False)
        pd.DataFrame(property_rows[len(property_rows)//2:]).to_csv(props_2_path, index=False)

    sales_path = RAW_FORECAST / "sales_train_validation.csv"
    calendar_path = RAW_FORECAST / "calendar.csv"
    prices_path = RAW_FORECAST / "sell_prices.csv"

    if not sales_path.exists():
        rng = np.random.default_rng(7)
        item_ids = list(range(1, 25))
        stores = ["CA_1", "TX_1", "WI_1"]
        dept_map = {1: "FOODS", 2: "HOUSEHOLD", 3: "HOBBIES"}
        cat_map = {1: "FOODS", 2: "HOUSEHOLD", 3: "HOBBIES"}
        rows = []
        for item_id in item_ids:
            dept_id = dept_map[(item_id % 3) + 1]
            cat_id = cat_map[(item_id % 3) + 1]
            for store in stores:
                state = store.split("_")[0]
                item_row = {"item_id": f"ITEM_{item_id:03d}", "dept_id": dept_id, "cat_id": cat_id, "store_id": store, "state_id": state}
                for day_idx in range(1, 181):
                    trend = 25 + (item_id % 5) + (day_idx / 10)
                    seasonal = 8 * np.sin(day_idx / 7) + 3 * np.cos(day_idx / 14)
                    noise = rng.normal(0, 2)
                    item_row[f"d_{day_idx}"] = max(0, round(trend + seasonal + noise))
                rows.append(item_row)
        sales_df = pd.DataFrame(rows)
        sales_df.to_csv(sales_path, index=False)

        dates = pd.date_range("2023-01-01", periods=180, freq="D")
        calendar_df = pd.DataFrame({
            "date": dates,
            "wm_yr_wk": dates.isocalendar().year * 100 + dates.isocalendar().week,
            "weekday": dates.strftime("%A"),
            "wday": dates.dayofweek,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, len(dates) + 1)],
        })
        calendar_df.to_csv(calendar_path, index=False)

        price_rows = []
        for item_id in item_ids:
            item_name = f"ITEM_{item_id:03d}"
            for store in stores:
                for week in sorted(calendar_df["wm_yr_wk"].unique()[:12]):
                    price_rows.append({
                        "store_id": store,
                        "item_id": item_name,
                        "wm_yr_wk": week,
                        "sell_price": round(8 + (item_id % 6) * 0.7 + (0.1 * (week % 10)), 2),
                    })
        pd.DataFrame(price_rows).to_csv(prices_path, index=False)


if __name__ == "__main__":
    ensure_demo_data()
    print("Demo data generated in data/raw")
