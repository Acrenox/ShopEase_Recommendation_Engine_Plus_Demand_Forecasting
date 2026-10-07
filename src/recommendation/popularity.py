from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "user_item_interactions.parquet"


def recommend_popular(k: int = 10) -> list[dict]:
    """Return the top-k globally popular products."""
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError("Interaction data is missing. Run recommendation preprocessing first.")

    interactions = pd.read_parquet(INTERACTIONS_PATH)
    popularity = (
        interactions.groupby("item_id", as_index=False)["interaction_score"]
        .sum()
        .sort_values(["interaction_score", "item_id"], ascending=[False, True])
        .head(k)
        .rename(columns={"interaction_score": "score"})
    )
    return [{"item_id": int(row.item_id), "score": float(row.score)} for row in popularity.itertuples(index=False)]
