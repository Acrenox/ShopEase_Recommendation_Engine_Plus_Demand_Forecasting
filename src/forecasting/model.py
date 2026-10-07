from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "forecasting_features.parquet"
MODEL_PATH = ROOT / "models" / "forecasting" / "random_forest.joblib"


def train_random_forest_model(feature_path: str | Path | None = None, model_path: str | Path | None = None) -> RandomForestRegressor:
    if feature_path is None:
        feature_path = FEATURES_PATH
    feature_path = Path(feature_path)
    if not feature_path.exists():
        raise FileNotFoundError("Forecasting features are missing. Run forecast preprocessing and feature engineering first.")

    df = pd.read_parquet(feature_path).copy()
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    target = "sales"
    feature_columns = [
        col
        for col in df.columns
        if col not in {"date", target}
    ]

    categorical_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    if categorical_columns:
        training_frame = pd.get_dummies(df[feature_columns + [target]], columns=categorical_columns, dtype=float)
    else:
        training_frame = df[feature_columns + [target]].copy()

    X = training_frame.drop(columns=[target])
    y = training_frame[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(X, y)

    if model_path is None:
        model_path = MODEL_PATH
    else:
        model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


if __name__ == "__main__":
    train_random_forest_model()
