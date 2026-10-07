from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "user_item_interactions.parquet"


def recommend_collaborative(user_id: int, k: int = 10) -> list[dict]:
    """Return top-k recommendations for a user using item-item collaborative filtering."""
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError("Interaction data is missing. Run recommendation preprocessing first.")

    interactions = pd.read_parquet(INTERACTIONS_PATH)
    user_history = interactions[interactions["user_id"] == user_id]
    if user_history.empty:
        return []

    item_matrix = interactions.pivot_table(index="user_id", columns="item_id", values="interaction_score", aggfunc="sum").fillna(0)
    item_similarity = cosine_similarity(item_matrix.T.values)
    similarity_df = pd.DataFrame(item_similarity, index=item_matrix.columns, columns=item_matrix.columns)

    interacted_items = set(user_history["item_id"].astype(int).tolist())
    candidate_scores = similarity_df.loc[list(interacted_items)].sum(axis=0)
    candidate_scores = candidate_scores[~candidate_scores.index.isin(interacted_items)]
    ranked = candidate_scores.sort_values(ascending=False).head(k)

    return [{"item_id": int(item_id), "score": float(score)} for item_id, score in ranked.items()]
