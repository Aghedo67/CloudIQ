# CloudIQ — E-commerce Analytics on Google BigQuery

## Business Problem

The analytics team at a mid-size e-commerce retailer needs to move from spreadsheet reporting to a cloud-scale analytics platform. The brief covers four questions:

1. **Revenue** — where is money coming from and is the business growing?
2. **Retention** — how quickly do customers churn after their first purchase, and which cohorts are healthiest?
3. **Conversion** — where is the website funnel losing customers, and which traffic sources convert best?
4. **Prediction** — which customers are most likely to make a second purchase within 90 days?

This project answers all four using Google BigQuery, standard SQL, and BigQuery ML — without moving data out of the warehouse.

---

## Live Dashboard

*(Link added after Streamlit deployment)*

---

## Key Findings

| Area | Finding |
|------|---------|
| Top revenue category | Outerwear & Coats |
| Highest-return category | Intimates (~20% return rate) |
| Month-1 retention drop | ~75% of first-time buyers do not return the following month |
| Cart-to-purchase rate | ~40% of sessions that add to cart complete the purchase |
| Highest-converting source | Email (highest cart-to-purchase rate) |
| RFM Champions share | ~8% of customers drive ~25% of revenue |
| BigQuery ML AUC | 0.816 — model correctly ranks repeat vs non-repeat buyer 81.6% of the time |
| Repeat purchase rate (90-day) | ~20% of customers place a second order within 90 days |

---

## Project Structure

```
CloudIQ/
├── data/
│   └── processed/                     # Query results exported from BigQuery
│       ├── kpis_summary.csv
│       ├── monthly_revenue.csv
│       ├── monthly_growth.csv
│       ├── revenue_by_category.csv
│       ├── revenue_by_country.csv
│       ├── top_products.csv
│       ├── order_status.csv
│       ├── return_rate_by_category.csv
│       ├── cohort_retention.csv
│       ├── new_vs_returning.csv
│       ├── conversion_funnel.csv
│       ├── funnel_by_source.csv
│       ├── time_to_purchase.csv
│       ├── rfm_scores.csv
│       ├── rfm_segment_summary.csv
│       ├── bqml_evaluation.csv
│       ├── bqml_feature_weights.csv
│       └── repeat_purchase_predictions.csv
├── notebooks/
│   ├── 01_setup_and_analytics.ipynb   # BigQuery connection, schema, SQL analytics
│   ├── 02_cohort_and_funnel.ipynb     # Cohort retention, conversion funnel
│   ├── 03_rfm_and_bqml.ipynb          # RFM segmentation, BigQuery ML model
│   └── 04_dashboard_prep.ipynb        # Final exports for Streamlit
├── app.py                             # Streamlit dashboard (5 tabs)
├── outputs/figures/                   # Saved charts
├── requirements.txt
└── README.md
```

---

## Methodology

### Notebook 01 — SQL Analytics
All queries run directly in BigQuery against the public `thelook_ecommerce` dataset. Key SQL patterns demonstrated:

- `DATE_TRUNC` for time-series aggregation by month
- `WINDOW` functions: `LAG()` for month-over-month growth, `SUM() OVER` for cumulative revenue, `AVG() OVER ROWS BETWEEN` for rolling averages
- Multi-table `JOIN` chains across `orders`, `order_items`, `products`, `users`
- Revenue share using `SUM(...) OVER ()` as a denominator in a single pass

### Notebook 02 — Cohort & Funnel Analysis
**Cohort retention** — built entirely in BigQuery SQL using a four-step CTE chain:
1. Find each customer's first purchase month with `MIN()` + `GROUP BY`
2. Calculate `DATE_DIFF` between each subsequent order and the first purchase month (the period number)
3. Count distinct users per cohort–period pair
4. Divide by cohort size (period 0) to get retention rate

No Python loops — the entire cohort matrix is returned in a single query and pivoted with pandas.

**Conversion funnel** — uses the `events` table with `MAX(CASE WHEN event_type = '...' THEN 1 ELSE 0 END)` per session to classify each session by the deepest funnel stage reached. Segmented by `traffic_source` to identify which acquisition channels convert most efficiently.

**Time to first purchase** — `DATE_DIFF` between `MIN(events.created_at)` and `MIN(orders.created_at)` per user, identifying the 25th, 50th, 75th, and 90th percentile conversion windows.

### Notebook 03 — RFM Segmentation & BigQuery ML

**RFM scoring** — uses `NTILE(5) OVER (ORDER BY ...)` to split customers into five equal buckets on each dimension. Recency is reversed (lower days = higher score). Combined RFM score (e.g. "555") maps to ten named segments: Champions, Loyal Customers, Potential Loyalists, Promising, Recent Customers, Needs Attention, At Risk, Can't Lose Them, Hibernating, Lost.

**BigQuery ML** — logistic regression model trained entirely with SQL:

```sql
CREATE OR REPLACE MODEL `cloudiq-analytics.cloudiq.repeat_buyer_model`
OPTIONS(
    model_type         = 'logistic_reg',
    input_label_cols   = ['repeat_buyer'],
    auto_class_weights = TRUE,
    max_iterations     = 20,
    l2_reg             = 0.1
) AS
SELECT age, gender, country, first_order_value, first_category,
       num_items, days_account_to_first_order, traffic_source, repeat_buyer
FROM `cloudiq-analytics.cloudiq.repeat_purchase_training`
```

Model is evaluated with `ML.EVALUATE`, scored with `ML.PREDICT`, and feature weights extracted with `ML.WEIGHTS` — all in SQL.

| Metric | Value |
|--------|-------|
| ROC AUC | **0.816** |
| Accuracy | 0.752 |
| F1 Score | 0.570 |
| Precision | 0.470 |
| Recall | 0.724 |

`auto_class_weights = TRUE` upweights the minority class (repeat buyers ~20% of data), intentionally trading precision for recall — the right trade-off when missing a genuine repeat buyer (lost marketing opportunity) is costlier than a false positive (one wasted outreach).

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| Executive Overview | Revenue trend, MoM growth, category/country breakdown, return rates, top products |
| Customer Analytics | Cohort retention heatmap, new vs returning customers, time-to-purchase distribution |
| Conversion Funnel | Plotly funnel chart, step drop-off, breakdown by traffic source |
| RFM Segmentation | Segment bar chart, revenue share pie, frequency/recency bubble map, segment profiles |
| BigQuery ML | Model metrics, predicted probability distribution, feature weights, top-50 prospects table |

---

## Technical Stack

| Layer | Tools |
|-------|-------|
| Cloud data warehouse | Google BigQuery |
| Query language | Standard SQL — window functions, CTEs, UNNEST, DATE_TRUNC, NTILE |
| In-warehouse ML | BigQuery ML (logistic regression, `ML.EVALUATE`, `ML.PREDICT`, `ML.WEIGHTS`) |
| Notebook environment | Google Colab + `google-cloud-bigquery` Python client |
| Data processing | pandas, numpy |
| Visualisation | plotly, Streamlit |
| Dataset | `bigquery-public-data.thelook_ecommerce` |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GCP (Colab — run in notebook)
from google.colab import auth
auth.authenticate_user()

# 3. Run notebooks in order (Google Colab)
#    01 → 02 → 03 → 04
#    Each notebook saves results to /content/processed/

# 4. Copy processed CSVs to data/processed/

# 5. Launch dashboard
streamlit run app.py
```

---

## Data Source

[TheLook E-commerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) — a synthetic e-commerce dataset maintained by Google as a BigQuery public dataset. Contains orders, order items, users, products, events, and sessions tables (~125K orders). Freely accessible to any GCP account via `bigquery-public-data.thelook_ecommerce`.

---

## Interview Talking Points

**On BigQuery:**
BigQuery is a serverless, columnar data warehouse. It separates storage from compute so there are no idle server costs. For analytical workloads — large aggregations, window functions, GROUP BY across millions of rows — it outperforms row-based databases by orders of magnitude because it only scans the columns a query needs.

**On BigQuery ML:**
BigQuery ML trains models directly in the warehouse using SQL. The model lives alongside the data — no ETL, no separate ML platform, no data export. It is the right tool when data is already in BigQuery, models need to score at warehouse scale, and the team does not want to maintain a separate infrastructure. For more complex needs (deep learning, SHAP explainability, custom loss) you would export and use Python.

**On window functions:**
A window function computes a result across a set of rows related to the current row without collapsing them into one output row. Used for: running totals (`SUM OVER ORDER BY`), period-over-period comparisons (`LAG`/`LEAD`), ranking within groups (`ROW_NUMBER`/`RANK OVER PARTITION BY`), and moving averages (`AVG OVER ROWS BETWEEN`). Essential for any analytical SQL role.

**On RFM:**
RFM (Recency, Frequency, Monetary) scores customers on three behavioural dimensions using equal-width bucket splits (`NTILE`). It is interpretable by non-technical stakeholders, actionable (Champions get loyalty rewards, At Risk customers get win-back campaigns), and computable from any order history with a single SQL query. The limitation is that NTILE buckets are relative to the current customer base — a "5" today might be a "3" next quarter if the customer base grows.
