# E-commerce ML Project Workflow

This workflow describes the end-to-end execution order for the ShopEase recommendation and demand forecasting project.

The project should be built sequentially:

```text
data acquisition -> EDA -> preprocessing -> feature engineering -> modeling -> evaluation -> business outputs
```

## 1. Environment Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected dependencies:

```bash
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

## 2. Data Acquisition

Download the real datasets from Kaggle:

- RetailRocket e-commerce dataset: https://www.kaggle.com/retailrocket/ecommerce-dataset
- M5 Forecasting Accuracy dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Place files in this layout:

```text
data/raw/recommendation/events.csv
data/raw/recommendation/item_properties_part1.csv
data/raw/recommendation/item_properties_part2.csv
data/raw/recommendation/category_tree.csv
data/raw/forecasting/sales_train_validation.csv
data/raw/forecasting/calendar.csv
data/raw/forecasting/sell_prices.csv
```

Do not commit raw Kaggle datasets to Git unless the repository policy explicitly allows it.

## 3. Recommendation Pipeline

### Step 3.1: Run Recommendation EDA

Notebook:

```text
notebooks/01_recommendation_eda.ipynb
```

Tasks:

- Load RetailRocket files.
- Convert event timestamps to datetime.
- Count users, products, events, event types, and transactions.
- Check missing values and duplicates.
- Identify most popular products and most active users.
- Plot event type distribution and activity over time.

Output:

- EDA charts
- Business observations
- Data quality notes

### Step 3.2: Build Interaction Dataset

Module:

```text
src/recommendation/preprocessing.py
```

Tasks:

- Map event weights:
  - `view = 1`
  - `addtocart = 3`
  - `transaction = 5`
- Aggregate repeated user-item interactions.
- Save the processed interaction table.

Output:

```text
data/processed/user_item_interactions.parquet
```

### Step 3.3: Build Product Features

Module:

```text
src/recommendation/preprocessing.py
```

Tasks:

- Combine item property files.
- Create item-level metadata.
- Add category hierarchy where possible.
- Build TF-IDF features from item attributes.
- Save vectorizer and item-feature matrix.

Outputs:

```text
data/processed/item_features.parquet
models/recommender/tfidf_vectorizer.joblib
models/recommender/item_feature_matrix.joblib
```

### Step 3.4: Train and Test Recommendation Models

Modules:

```text
src/recommendation/popularity.py
src/recommendation/content_based.py
src/recommendation/collaborative.py
src/recommendation/hybrid.py
```

Tasks:

- Implement popularity recommendations for cold start.
- Implement content-based recommendations with TF-IDF cosine similarity.
- Implement item-item collaborative filtering from user behavior.
- Combine content and collaborative scores in the hybrid recommender.
- Remove items already interacted with when generating final recommendations.

Expected callable functions:

```python
recommend_popular(k=10)
recommend_similar_products(item_id, k=10)
recommend_collaborative(user_id, k=10)
recommend(user_id, k=10)
```

### Step 3.5: Evaluate Recommendations

Module:

```text
src/recommendation/evaluation.py
```

Tasks:

- Split interactions chronologically.
- Train on historical interactions.
- Test on future interactions.
- Evaluate Popularity, Content-Based, Collaborative, and Hybrid approaches.

Metrics:

- Precision@5
- Recall@5
- Precision@10
- Recall@10

Output:

```text
reports/recommendation_metrics.csv
```

## 4. Forecasting Pipeline

### Step 4.1: Run Forecasting EDA

Notebook:

```text
notebooks/03_forecasting_eda.ipynb
```

Tasks:

- Load M5 files memory-efficiently.
- Convert sales from wide to long format.
- Merge calendar fields.
- Merge sell prices where useful.
- Plot overall, weekly, and monthly demand.
- Analyze demand by store, category, and department.

Output:

- EDA charts
- Seasonality observations
- Data quality notes

### Step 4.2: Build Forecasting Base Table

Module:

```text
src/forecasting/preprocessing.py
```

Tasks:

- Create a clean long-format table.
- Keep one row per `date`, `item_id`, and `store_id`.
- Include category, department, state, calendar, and price columns.

Output:

```text
data/processed/forecasting_base.parquet
```

### Step 4.3: Create Time-Series Features

Module:

```text
src/forecasting/features.py
```

Tasks:

- Create calendar features.
- Create lag features.
- Create rolling mean and rolling standard deviation features.
- Shift target values before rolling calculations to avoid leakage.

Output:

```text
data/processed/forecasting_features.parquet
```

### Step 4.4: Train Baseline Forecasts

Module:

```text
src/forecasting/baseline.py
```

Tasks:

- Create naive forecasts using previous-day demand.
- Create seasonal naive forecasts using demand from seven days ago.
- Use these results as comparison baselines.

### Step 4.5: Train ML Forecast

Module:

```text
src/forecasting/model.py
```

Tasks:

- Split data chronologically.
- Use the last 28 days as the test window.
- Train `RandomForestRegressor`.
- Save the trained model.

Output:

```text
models/forecasting/random_forest.joblib
```

### Step 4.6: Evaluate Forecasts

Module:

```text
src/forecasting/evaluation.py
```

Tasks:

- Compare baseline and ML forecasts.
- Calculate MAE, RMSE, and MAPE.
- Document MAPE handling for zero-demand rows.

Output:

```text
reports/forecast_metrics.csv
```

### Step 4.7: Generate Replenishment Recommendations

Tasks:

- Aggregate predictions by `store_id` and `item_id`.
- Calculate forecasted demand over 7 and 28 days.
- Estimate uncertainty from recent forecast errors or rolling demand variation.
- Assign reorder priority.

Output:

```text
reports/replenishment_recommendations.csv
```

Important note:

The M5 dataset does not provide current inventory. Without inventory, this output should be presented as a reorder-priority demonstration, not a final replenishment decision system.

## 5. Quality Checks

Before presenting results, verify:

- Raw datasets are not fabricated.
- Time-based splits are used for both use cases.
- Processed datasets are saved and reusable.
- Trained models are saved with `joblib`.
- Evaluation CSVs are generated.
- Recommendation outputs exclude already-interacted items where appropriate.
- Forecast rolling features do not use current or future target values.
- MAPE calculation handles zero-demand rows.
- Notebook charts have clear business explanations.

## 6. Suggested Execution Order

Follow this order when building the project:

1. Create folders and dependency files.
2. Download and place datasets.
3. Complete recommendation EDA.
4. Build recommendation preprocessing.
5. Implement popularity recommender.
6. Implement content-based recommender.
7. Implement collaborative recommender.
8. Implement hybrid recommender.
9. Evaluate recommendation models.
10. Complete forecasting EDA.
11. Build forecasting preprocessing.
12. Create forecasting features.
13. Implement naive and seasonal naive baselines.
14. Train Random Forest forecast model.
15. Evaluate forecast models.
16. Generate replenishment recommendations.
17. Update README with setup, usage, results, limitations, and interview story.

## 7. Expected Final Artifacts

```text
data/processed/user_item_interactions.parquet
data/processed/item_features.parquet
data/processed/forecasting_base.parquet
data/processed/forecasting_features.parquet
models/recommender/tfidf_vectorizer.joblib
models/recommender/item_feature_matrix.joblib
models/forecasting/random_forest.joblib
reports/recommendation_metrics.csv
reports/forecast_metrics.csv
reports/replenishment_recommendations.csv
```

## 8. Business Impact Summary

Recommendation Engine:

- Helps customers discover relevant products.
- Supports personalization based on behavior and product similarity.
- Provides a fallback for new or inactive users.
- Can be measured offline with Precision@K and Recall@K, then online with click-through rate, add-to-cart rate, conversion rate, and revenue lift.

Demand Forecasting:

- Helps inventory teams plan replenishment.
- Reduces stockout risk.
- Reduces excess inventory risk.
- Converts model predictions into business-facing reorder priorities.

## 9. Interview Talking Points

Be ready to explain:

- Why interactions are treated as implicit feedback.
- Why purchases receive more weight than views.
- Why chronological splitting is required.
- How collaborative filtering differs from content-based filtering.
- Why a hybrid recommender is useful.
- How cold-start users are handled.
- Why forecasting features must be shifted before rolling calculations.
- Why baseline models are necessary.
- How MAE, RMSE, and MAPE differ.
- Why replenishment output is a priority ranking when inventory data is unavailable.
