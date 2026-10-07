# E-commerce Recommendation Engine and Demand Forecasting Implementation

This document defines the implementation plan for a beginner-friendly, interview-ready machine learning project with two independent use cases:

- Product Recommendation Engine using the RetailRocket e-commerce dataset
- Demand Forecasting for inventory and replenishment planning using the M5 Forecasting dataset

The two pipelines must stay separate in code, data preparation, modeling, evaluation, and reporting.

## Dataset Sources

Use the real public datasets and do not fabricate input data when the original files are available.

- RetailRocket e-commerce dataset: https://www.kaggle.com/retailrocket/ecommerce-dataset
- M5 Forecasting Accuracy dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Before sharing or publishing outputs, review each Kaggle dataset page for license and usage requirements.

## Target Project Structure

```text
ShopEase_Recommendation_Engine/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── recommendation/
│   │   │   ├── events.csv
│   │   │   ├── item_properties_part1.csv
│   │   │   ├── item_properties_part2.csv
│   │   │   └── category_tree.csv
│   │   └── forecasting/
│   │       ├── sales_train_validation.csv
│   │       ├── calendar.csv
│   │       └── sell_prices.csv
│   └── processed/
├── notebooks/
│   ├── 01_recommendation_eda.ipynb
│   ├── 02_recommendation_modeling.ipynb
│   ├── 03_forecasting_eda.ipynb
│   └── 04_forecasting_modeling.ipynb
├── src/
│   ├── recommendation/
│   │   ├── preprocessing.py
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid.py
│   │   ├── popularity.py
│   │   └── evaluation.py
│   └── forecasting/
│       ├── preprocessing.py
│       ├── features.py
│       ├── baseline.py
│       ├── model.py
│       └── evaluation.py
├── models/
│   ├── recommender/
│   └── forecasting/
└── reports/
    ├── recommendation_metrics.csv
    ├── forecast_metrics.csv
    └── replenishment_recommendations.csv
```

## Recommended Dependencies

Keep the dependency set small and practical:

```text
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
pyarrow
joblib
jupyter
```

Optional advanced forecasting models can be added after the baseline and Random Forest model work:

```text
xgboost
lightgbm
```

## Part 1: Product Recommendation Engine

### Business Goal

Improve product discovery and repeat purchases by recommending relevant items based on user behavior and product metadata.

The recommender should combine:

- Collaborative filtering from user-item interactions
- Content-based filtering from product attributes
- Popularity fallback for cold-start users

### Raw Input Files

Place RetailRocket files in `data/raw/recommendation/`:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

### Recommendation EDA

Create `notebooks/01_recommendation_eda.ipynb`.

The notebook should report:

- Number of users
- Number of products
- Number of events
- Event types
- Number of transactions
- Number of unique categories
- Missing values
- Duplicate records
- Event time range
- Most popular products
- Most active users

Implementation notes:

- Convert event timestamps from Unix milliseconds to datetime.
- Analyze `view`, `addtocart`, and `transaction` events separately.
- Create visualizations for event distribution, activity over time, popular items, and active users.
- Explain what each chart means for the business.

### Interaction Scoring

Treat RetailRocket as implicit feedback data. Assign weights:

| Event Type | Weight |
|---|---:|
| view | 1 |
| addtocart | 3 |
| transaction | 5 |

Aggregate repeated user-item events into one row:

```text
interaction_score = views * 1 + addtocart * 3 + transactions * 5
```

Output:

```text
data/processed/user_item_interactions.parquet
```

Required columns:

- `user_id`
- `item_id`
- `interaction_score`
- `last_interaction_at`

### Product Feature Engineering

Use `item_properties_part1.csv`, `item_properties_part2.csv`, and `category_tree.csv` to build item metadata.

Recommended approach:

1. Concatenate item property files.
2. Keep useful property names such as category, brand-like fields, and stable product attributes.
3. Convert values to strings because many item properties are encoded.
4. Aggregate item attributes into one document-like field per item.
5. Apply TF-IDF to create an item-feature matrix.

Output:

```text
data/processed/item_features.parquet
models/recommender/tfidf_vectorizer.joblib
models/recommender/item_feature_matrix.joblib
```

### Content-Based Recommender

Create `src/recommendation/content_based.py`.

Required function:

```python
def recommend_similar_products(item_id: int, k: int = 10) -> list[dict]:
    """Return the top-k items most similar to the input item using TF-IDF cosine similarity."""
```

Implementation:

- Load the saved item-feature matrix.
- Locate the requested `item_id`.
- Compute cosine similarity between the item vector and all other item vectors.
- Exclude the same item.
- Return top-k item IDs with similarity scores.

### Collaborative Filtering Recommender

Create `src/recommendation/collaborative.py`.

Required function:

```python
def recommend_collaborative(user_id: int, k: int = 10) -> list[dict]:
    """Return top-k recommendations for a user using item-item collaborative filtering."""
```

Implementation:

- Build a sparse user-item matrix from `user_item_interactions.parquet`.
- Use item-item cosine similarity as the first implementation.
- Score candidate items based on similarity to the user's interacted items.
- Remove items already interacted with by the user.
- Return top-k item IDs with scores.

Optional later improvement:

- `TruncatedSVD` matrix factorization
- ALS using the `implicit` library

### Popularity Fallback

Create `src/recommendation/popularity.py`.

Use weighted event popularity:

```text
popularity_score = views * 1 + addtocart * 3 + transactions * 5
```

Required function:

```python
def recommend_popular(k: int = 10) -> list[dict]:
    """Return the top-k globally popular products."""
```

This function is used when:

- `user_id` is unknown
- the user has no interaction history
- collaborative filtering cannot generate enough recommendations

### Hybrid Recommender

Create `src/recommendation/hybrid.py`.

Required function:

```python
def recommend(user_id: int, k: int = 10) -> list[dict]:
    """Return top-k hybrid recommendations for a user."""
```

Recommended score blend:

```text
hybrid_score = 0.6 * normalized_collaborative_score + 0.4 * normalized_content_score
```

Implementation:

1. Check whether the user has interaction history.
2. If not, return popularity recommendations.
3. Generate collaborative candidates.
4. Generate content-based candidates from the user's strongest or most recent interacted items.
5. Normalize collaborative and content scores before blending.
6. Remove items already interacted with.
7. Fill missing slots with popular products if fewer than `k` recommendations remain.

### Recommendation Evaluation

Create `src/recommendation/evaluation.py`.

Use a chronological train/test split. Do not randomly split recommendation events because that leaks future behavior into training.

Recommended split:

- Oldest 80 percent of interactions for training
- Latest 20 percent for testing

Evaluate each model with:

- Precision@5
- Recall@5
- Precision@10
- Recall@10

Definitions:

```text
Precision@K = relevant recommended items / K
Recall@K = relevant recommended items / total relevant test items
```

Output:

```text
reports/recommendation_metrics.csv
```

Example schema:

```text
model,precision@5,recall@5,precision@10,recall@10
Popularity,...
Content-Based,...
Collaborative,...
Hybrid,...
```

## Part 2: Demand Forecasting

### Business Goal

Predict future product demand so inventory teams can plan replenishment, reduce stockouts, avoid overstocking, and improve purchasing decisions.

### Raw Input Files

Place M5 files in `data/raw/forecasting/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

### Forecasting EDA

Create `notebooks/03_forecasting_eda.ipynb`.

Because M5 is large:

- Load only necessary columns during experiments.
- Use memory-efficient dtypes.
- Convert wide daily sales columns (`d_1`, `d_2`, ...) into long format only for the selected modeling scope if memory is limited.

Clean output schema:

```text
date
item_id
store_id
department
category
state
sales
```

Merge:

- Calendar fields from `calendar.csv`
- Price fields from `sell_prices.csv` where useful

EDA should include:

- Overall daily sales trend
- Weekly sales
- Monthly sales
- Sales by store
- Sales by category
- Sales by department
- Top-selling products
- Low-selling products
- Seasonal patterns

### Forecasting Feature Engineering

Create `src/forecasting/features.py`.

Time features:

- `year`
- `month`
- `week`
- `day_of_week`
- `day_of_month`
- `quarter`

Lag features:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

Rolling features:

- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_28`
- `rolling_std_7`
- `rolling_std_28`

Data leakage rule:

Rolling features must use only previous observations. For example, compute rolling statistics after shifting the target by one day within each `store_id` and `item_id` group.

Output:

```text
data/processed/forecasting_features.parquet
```

### Baseline Forecasts

Create `src/forecasting/baseline.py`.

Implement:

- Naive forecast: next demand equals previous day's demand
- Seasonal naive forecast: next demand equals demand from 7 days ago

Baselines must be evaluated before ML models. They create the minimum performance bar.

### Machine Learning Forecast

Create `src/forecasting/model.py`.

Start with:

```python
RandomForestRegressor
```

Recommended feature groups:

- Lag features
- Rolling features
- Calendar features
- Price features
- Store, state, category, and department features

Use simple encodings first:

- One-hot encoding for small categorical columns
- Ordinal or frequency encoding only if one-hot creates too many columns

Save trained model:

```text
models/forecasting/random_forest.joblib
```

Optional later comparisons:

- XGBoost
- LightGBM

### Forecast Train/Test Split

Use chronological splitting only.

Recommended split:

- Training: historical records before the last 28 days
- Test: most recent 28 days

This simulates the real business question: based on everything known today, can the model predict the next 28 days?

### Forecast Evaluation

Create `src/forecasting/evaluation.py`.

Metrics:

- MAE: average absolute prediction error
- RMSE: penalizes large errors more heavily
- MAPE: percentage error, calculated carefully when actual demand is zero

Recommended MAPE handling:

- Exclude zero-actual rows from MAPE, or
- Use a small epsilon denominator and clearly document the choice

Output:

```text
reports/forecast_metrics.csv
```

Example schema:

```text
model,MAE,RMSE,MAPE
Naive,...
Seasonal Naive,...
Random Forest,...
XGBoost,...
```

### Replenishment Planning Output

Create a business-facing replenishment table:

```text
reports/replenishment_recommendations.csv
```

Recommended columns:

- `item_id`
- `store_id`
- `forecasted_demand_7d`
- `forecasted_demand_28d`
- `forecast_uncertainty`
- `reorder_priority`

If current inventory data is unavailable, do not fabricate it. Clearly state that the replenishment layer is a demonstration and uses expected demand plus uncertainty to rank reorder priority.

Suggested priority logic:

```text
HIGH = high forecasted demand and high recent sales velocity
MEDIUM = moderate forecasted demand or volatile demand
LOW = low forecasted demand
```

If inventory becomes available later, use:

```text
if forecasted_demand_28d > current_inventory:
    recommendation = "REPLENISH"
else:
    recommendation = "SUFFICIENT"
```

## Final Deliverables Checklist

- Clean recommendation interaction dataset
- Product feature matrix
- Content-based recommender
- Collaborative filtering recommender
- Hybrid recommender
- Popularity fallback
- Recommendation evaluation with Precision@5, Recall@5, Precision@10, Recall@10
- Clean forecasting dataset
- Time-series feature table
- Baseline forecasts
- Random Forest forecast model
- Forecast evaluation with MAE, RMSE, MAPE
- Replenishment recommendation output
- Model comparison tables
- EDA visualizations
- README with setup, workflow, results, limitations, and interview explanation

## Interview Explanation

Recommendation Engine:

> An e-commerce client wanted to improve product discovery and personalization. I built a hybrid recommendation engine using implicit user interactions and product metadata. I converted views, cart additions, and purchases into weighted interaction scores, then combined collaborative filtering with content-based similarity. I evaluated the system with Precision@K and Recall@K and added a popularity fallback for cold-start users.

Demand Forecasting:

> The business also needed better inventory planning. I developed a demand forecasting pipeline using historical sales, trend, seasonality, lag, rolling-window, calendar, and price features. I used chronological train/test splits to avoid leakage and compared baseline forecasts against a Random Forest model using MAE, RMSE, and MAPE. The forecasts were converted into replenishment priorities for inventory planning.

## Limitations

- RetailRocket item properties are partially anonymized, so product metadata may be less interpretable than a real catalog.
- M5 does not provide current inventory, so replenishment recommendations are priority signals rather than final purchase orders.
- Item-item collaborative filtering can be memory-intensive for very large catalogs.
- MAPE can be unstable when demand is zero or close to zero.
- Offline recommendation metrics do not fully capture online business impact.

## Future Improvements

- Add approximate nearest neighbor search for scalable item similarity.
- Add ALS or matrix factorization for stronger collaborative filtering.
- Add LightGBM or XGBoost for demand forecasting.
- Add model tracking with MLflow.
- Add scheduled retraining.
- Add API endpoints for serving recommendations and forecasts.
- Add business metrics such as conversion lift, revenue lift, stockout reduction, and inventory holding cost reduction.
