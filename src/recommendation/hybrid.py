from __future__ import annotations

from pathlib import Path

import pandas as pd

from .collaborative import recommend_collaborative
from .content_based import recommend_similar_products
from .popularity import recommend_popular

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "user_item_interactions.parquet"


def recommend(user_id: int, k: int = 10) -> list[dict]:
    """Return top-k hybrid recommendations for a user."""
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError("Interaction data is missing. Run recommendation preprocessing first.")

    interactions = pd.read_parquet(INTERACTIONS_PATH)
    if user_id not in interactions["user_id"].unique():
        return recommend_popular(k=k)

    collaborative_candidates = recommend_collaborative(user_id, k=max(k * 5, 20))
    user_items = interactions[interactions["user_id"] == user_id].sort_values("interaction_score", ascending=False)["item_id"].tolist()
    content_candidates = []
    for item_id in user_items[:5]:
        content_candidates.extend(recommend_similar_products(int(item_id), k=max(k * 3, 10)))

    collaborative_map = {int(item["item_id"]): float(item["score"]) for item in collaborative_candidates}
    content_map = {}
    for item in content_candidates:
        content_map[int(item["item_id"])] = max(content_map.get(int(item["item_id"]), 0.0), float(item["score"]))

    popular_map = {int(item["item_id"]): float(item["score"]) for item in recommend_popular(k=max(k * 5, 20))}
    interacted = set(int(item_id) for item_id in user_items)
    all_candidates = set(collaborative_map) | set(content_map) | set(popular_map)
    all_candidates = {item for item in all_candidates if item not in interacted}

    if not all_candidates:
        return []

    max_coll = max(collaborative_map.values(), default=0.0) or 1.0
    max_content = max(content_map.values(), default=0.0) or 1.0
    max_pop = max(popular_map.values(), default=0.0) or 1.0

    blended = {}
    for item_id in all_candidates:
        coll_score = collaborative_map.get(item_id, 0.0) / max_coll
        content_score = content_map.get(item_id, 0.0) / max_content
        pop_score = popular_map.get(item_id, 0.0) / max_pop
        blended[item_id] = 0.6 * coll_score + 0.4 * content_score + 0.1 * pop_score

    ranked = sorted(blended.items(), key=lambda item: item[1], reverse=True)[:k]
    return [{"item_id": item_id, "score": float(score)} for item_id, score in ranked]
