from __future__ import annotations

from pathlib import Path

import pandas as pd

from .collaborative import recommend_collaborative
from .content_based import recommend_similar_products
from .hybrid import recommend
from .popularity import recommend_popular

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "user_item_interactions.parquet"
OUTPUT_PATH = ROOT / "reports" / "recommendation_metrics.csv"


def _safe_metric(true_set: set[int], recs: list[dict], k: int) -> tuple[float, float]:
    rec_items = {int(item["item_id"]) for item in recs[:k]}
    relevant = len(true_set & rec_items)
    precision = relevant / k if k else 0.0
    recall = relevant / len(true_set) if true_set else 0.0
    return precision, recall


def _content_recommendations_for_user(train_df: pd.DataFrame, user_id: int, k: int = 10) -> list[dict]:
    history = train_df[train_df["user_id"] == user_id].sort_values("interaction_score", ascending=False)
    if history.empty:
        return []

    candidates: dict[int, float] = {}
    for seed_item in history["item_id"].head(5).astype(int).tolist():
        for item in recommend_similar_products(seed_item, k=max(10, k)):
            item_id = int(item["item_id"])
            candidates[item_id] = max(candidates.get(item_id, 0.0), float(item["score"]))

    return [{"item_id": item_id, "score": score} for item_id, score in sorted(candidates.items(), key=lambda pair: pair[1], reverse=True)[:k]]


def evaluate_recommendation_models() -> pd.DataFrame:
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError("Interaction data is missing. Run recommendation preprocessing first.")

    df = pd.read_parquet(INTERACTIONS_PATH)
    df = df.sort_values("last_interaction_at").reset_index(drop=True)
    split_idx = max(1, int(len(df) * 0.8))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    rows: list[dict] = []
    for model_name, model_func in {
        "Popularity": lambda user_id, k: recommend_popular(k=k),
        "Content-Based": lambda user_id, k: _content_recommendations_for_user(train_df, int(user_id), k=k),
        "Collaborative": lambda user_id, k: recommend_collaborative(int(user_id), k=k),
        "Hybrid": lambda user_id, k: recommend(int(user_id), k=k),
    }.items():
        precisions_5 = []
        recalls_5 = []
        precisions_10 = []
        recalls_10 = []

        for user_id in sorted(test_df["user_id"].unique()):
            historical = set(train_df[train_df["user_id"] == user_id]["item_id"].astype(int).tolist())
            relevant = set(test_df[(test_df["user_id"] == user_id) & (~test_df["item_id"].isin(list(historical)))]["item_id"].astype(int).tolist())
            if not relevant:
                continue
            recs = model_func(user_id, 10)
            p5, r5 = _safe_metric(relevant, recs, 5)
            p10, r10 = _safe_metric(relevant, recs, 10)
            precisions_5.append(p5)
            recalls_5.append(r5)
            precisions_10.append(p10)
            recalls_10.append(r10)

        rows.append(
            {
                "model": model_name,
                "precision@5": round(sum(precisions_5) / len(precisions_5), 4) if precisions_5 else 0.0,
                "recall@5": round(sum(recalls_5) / len(recalls_5), 4) if recalls_5 else 0.0,
                "precision@10": round(sum(precisions_10) / len(precisions_10), 4) if precisions_10 else 0.0,
                "recall@10": round(sum(recalls_10) / len(recalls_10), 4) if recalls_10 else 0.0,
            }
        )

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    return output
