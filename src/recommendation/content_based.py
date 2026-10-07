from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "item_features.parquet"
MATRIX_PATH = ROOT / "models" / "recommender" / "item_feature_matrix.joblib"


def recommend_similar_products(item_id: int, k: int = 10) -> list[dict]:
    """Return the top-k items most similar to the input item using TF-IDF cosine similarity."""
    if not PROCESSED_PATH.exists() or not MATRIX_PATH.exists():
        raise FileNotFoundError("Item feature matrix is missing. Run recommendation preprocessing first.")

    item_features = pd.read_parquet(PROCESSED_PATH)
    item_matrix = joblib.load(MATRIX_PATH)
    item_features["item_id"] = item_features["item_id"].astype(int)

    if item_id not in item_features["item_id"].values:
        return []

    item_index = int(item_features.index[item_features["item_id"] == item_id][0])
    similarities = cosine_similarity(item_matrix[item_index], item_matrix).ravel()
    candidate_indexes = similarities.argsort()[::-1]
    top_results = []
    for candidate in candidate_indexes:
        candidate_item_id = int(item_features.iloc[candidate]["item_id"])
        if candidate_item_id == item_id:
            continue
        top_results.append({"item_id": candidate_item_id, "score": float(similarities[candidate])})
        if len(top_results) >= k:
            break
    return top_results
