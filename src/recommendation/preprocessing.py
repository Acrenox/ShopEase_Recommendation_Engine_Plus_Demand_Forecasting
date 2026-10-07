from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models" / "recommender"

EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}


def _read_recommendation_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = DATA_DIR / "raw" / "recommendation"
    events_path = raw_dir / "events.csv"
    item_properties_1 = raw_dir / "item_properties_part1.csv"
    item_properties_2 = raw_dir / "item_properties_part2.csv"
    category_tree = raw_dir / "category_tree.csv"

    if not events_path.exists() or not item_properties_1.exists() or not item_properties_2.exists():
        raise FileNotFoundError(
            "Recommendation data files not found. Run the demo generator or download the RetailRocket dataset into data/raw/recommendation/."
        )

    events = pd.read_csv(events_path)
    properties_1 = pd.read_csv(item_properties_1)
    properties_2 = pd.read_csv(item_properties_2)
    categories = pd.read_csv(category_tree) if category_tree.exists() else pd.DataFrame(columns=["category_id", "parent_id", "category_name"])
    return events, properties_1, properties_2, categories


def build_user_item_interactions(output_path: str | Path | None = None) -> pd.DataFrame:
    events, _, _, _ = _read_recommendation_inputs()
    required_cols = {"user_id", "item_id", "event_type", "timestamp"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"Recommendation events are missing required columns: {sorted(missing)}")

    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.lower()
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce")
    events["event_time"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
    events["weight"] = events["event_type"].map(EVENT_WEIGHTS).fillna(0)
    interactions = (
        events.groupby(["user_id", "item_id"], as_index=False)
        .agg(interaction_score=("weight", "sum"), last_interaction_at=("event_time", "max"))
        .sort_values(["user_id", "interaction_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if output_path is None:
        output_path = PROCESSED_DIR / "user_item_interactions.parquet"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(output_path, index=False)
    return interactions


def build_item_features(
    output_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    _, props_1, props_2, category_tree = _read_recommendation_inputs()

    property_frames = [props_1, props_2]
    item_properties = pd.concat(property_frames, ignore_index=True, sort=False)
    item_properties = item_properties.copy()
    item_properties.columns = [str(column).lower() for column in item_properties.columns]
    if "item_id" not in item_properties.columns:
        raise ValueError("Item property files must include an 'item_id' column.")
    if "property" in item_properties.columns:
        item_properties["property_name"] = item_properties["property"]
        item_properties["property_value"] = item_properties.get("value", "")
    elif "property_name" not in item_properties.columns:
        item_properties["property_name"] = "attribute"
    if "property_value" not in item_properties.columns:
        item_properties["property_value"] = ""

    item_properties["property_value"] = item_properties["property_value"].fillna("").astype(str)
    item_properties["property_name"] = item_properties["property_name"].fillna("attribute").astype(str)

    if not category_tree.empty:
        category_tree = category_tree.copy()
        category_tree.columns = [str(column).lower() for column in category_tree.columns]
        category_tree["category_id"] = category_tree["category_id"].astype(str)
        category_tree["category_name"] = category_tree.get("category_name", category_tree.get("category", "")).fillna("").astype(str)
        if "category_id" in item_properties.columns:
            item_properties = item_properties.merge(category_tree[["category_id", "category_name"]], on="category_id", how="left")
            item_properties["category_name"] = item_properties["category_name"].fillna("")
        else:
            item_properties["category_name"] = ""
    else:
        item_properties["category_name"] = ""

    item_text = (
        item_properties.groupby("item_id", as_index=False)
        .agg(item_text=("property_value", lambda values: " ".join(sorted(set(values.astype(str).tolist())))))
    )
    if "category_name" in item_properties.columns:
        category_text = (
            item_properties.groupby("item_id", as_index=False)
            .agg(category_text=("category_name", lambda values: " ".join(sorted(set(values.astype(str).tolist())))))
        )
        item_text = item_text.merge(category_text, on="item_id", how="left")
        item_text["item_text"] = item_text["item_text"].fillna("") + " " + item_text["category_text"].fillna("")
    item_text = item_text[["item_id", "item_text"]]

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(item_text["item_text"].fillna(""))

    if output_path is None:
        output_path = PROCESSED_DIR / "item_features.parquet"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_text.to_parquet(output_path, index=False)

    if vectorizer_path is None:
        vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    else:
        vectorizer_path = Path(vectorizer_path)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)

    if matrix_path is None:
        matrix_path = MODEL_DIR / "item_feature_matrix.joblib"
    else:
        matrix_path = Path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matrix, matrix_path)

    return item_text


if __name__ == "__main__":
    build_user_item_interactions()
    build_item_features()
