from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model import train_random_forest_model

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "forecasting_features.parquet"
MODEL_PATH = ROOT / "models" / "forecasting" / "random_forest.joblib"
OUTPUT_PATH = ROOT / "reports" / "forecast_metrics.csv"


def _calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_forecasts(feature_path: str | Path | None = None, model_path: str | Path | None = None) -> pd.DataFrame:
    if feature_path is None:
        feature_path = FEATURES_PATH
    feature_path = Path(feature_path)
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)

    if not feature_path.exists():
        raise FileNotFoundError("Forecasting feature data is missing. Run feature engineering first.")
    if not model_path.exists():
        train_random_forest_model(feature_path, model_path)

    df = pd.read_parquet(feature_path).copy()
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE"])

    split_date = df["date"].max() - pd.Timedelta(days=28)
    test_df = df[df["date"] > split_date].copy()
    if test_df.empty:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE"])

    model = joblib.load(model_path)
    feature_columns = [
        col
        for col in test_df.columns
        if col not in {"date", "sales"}
    ]
    category_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_object_dtype(test_df[col]) or pd.api.types.is_string_dtype(test_df[col])
    ]
    X_test = pd.get_dummies(test_df[feature_columns], columns=category_columns, dtype=float)
    X_test = X_test.reindex(columns=getattr(model, "feature_names_in_", X_test.columns), fill_value=0)
    rf_pred = model.predict(X_test)

    test_actual = test_df["sales"].to_numpy()
    naive_pred = test_df["lag_1"].fillna(test_actual.mean()).to_numpy()
    seasonal_pred = test_df["lag_7"].fillna(test_actual.mean()).to_numpy()

    metric_rows = []
    for name, pred in {"Naive": naive_pred, "Seasonal Naive": seasonal_pred, "Random Forest": rf_pred}.items():
        errors = test_actual - pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mape = _calculate_mape(test_actual, pred)
        metric_rows.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)})

    metrics = pd.DataFrame(metric_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)
    return metrics


def generate_replenishment_recommendations(feature_path: str | Path | None = None, output_path: str | Path | None = None) -> pd.DataFrame:
    if feature_path is None:
        feature_path = FEATURES_PATH
    feature_path = Path(feature_path)
    if not feature_path.exists():
        raise FileNotFoundError("Forecasting feature data is missing. Run feature engineering first.")

    df = pd.read_parquet(feature_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    last_28_days = df["date"].max() - pd.Timedelta(days=27)
    recent = df[df["date"] >= last_28_days].copy()

    if recent.empty:
        recent = df.copy()

    aggregation = recent.groupby(["store_id", "item_id"], as_index=False).agg(
        forecasted_demand_7d=("sales", lambda s: float(s.iloc[: min(len(s), 7)].sum())),
        forecasted_demand_28d=("sales", lambda s: float(s.sum())),
        forecast_uncertainty=("sales", lambda s: float(s.std(ddof=0)))
    )

    aggregation["reorder_priority"] = np.where(
        aggregation["forecasted_demand_28d"] > aggregation["forecast_uncertainty"],
        "HIGH",
        np.where(aggregation["forecasted_demand_28d"] > 0, "MEDIUM", "LOW"),
    )
    aggregation = aggregation[["item_id", "store_id", "forecasted_demand_7d", "forecasted_demand_28d", "forecast_uncertainty", "reorder_priority"]]

    if output_path is None:
        output_path = ROOT / "reports" / "replenishment_recommendations.csv"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregation.to_csv(output_path, index=False)
    return aggregation


if __name__ == "__main__":
    evaluate_forecasts()
    generate_replenishment_recommendations()
