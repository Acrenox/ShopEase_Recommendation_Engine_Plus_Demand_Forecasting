from __future__ import annotations

from scripts.generate_demo_data import ensure_demo_data
from src.forecasting.baseline import generate_baselines
from src.forecasting.evaluation import evaluate_forecasts, generate_replenishment_recommendations
from src.forecasting.features import create_forecasting_features
from src.forecasting.model import train_random_forest_model
from src.forecasting.preprocessing import build_forecasting_base
from src.recommendation.evaluation import evaluate_recommendation_models
from src.recommendation.preprocessing import build_item_features, build_user_item_interactions


def main() -> None:
    ensure_demo_data()

    build_user_item_interactions()
    build_item_features()
    evaluate_recommendation_models()

    build_forecasting_base()
    create_forecasting_features()
    generate_baselines()
    train_random_forest_model()
    evaluate_forecasts()
    generate_replenishment_recommendations()

    print("ShopEase pipeline completed successfully.")
    print("Recommendation metrics: reports/recommendation_metrics.csv")
    print("Forecast metrics: reports/forecast_metrics.csv")
    print("Replenishment output: reports/replenishment_recommendations.csv")


if __name__ == "__main__":
    main()
