# CloudIQ — E-commerce Analytics on Google BigQuery

## Business Problem

The analytics team at a mid-size e-commerce retailer needs to move from spreadsheet reporting to a cloud-scale analytics platform. This project builds that platform end-to-end: ingesting data from BigQuery, writing production-grade SQL for business metrics, segmenting customers with RFM, training a churn prediction model directly in BigQuery ML, and surfacing results in a live Streamlit dashboard.

---

## Project Structure

```
CloudIQ/
├── data/
│   └── processed/              # Query results exported locally
├── notebooks/
│   ├── 01_setup_and_analytics.ipynb   # BigQuery connection, schema, SQL analytics
│   ├── 02_cohort_and_funnel.ipynb     # Cohort retention, conversion funnel
│   ├── 03_rfm_and_bqml.ipynb          # RFM segmentation, BigQuery ML
│   └── 04_dashboard_prep.ipynb        # Final exports for Streamlit
├── sql/                        # Saved SQL queries (reusable)
├── app.py                      # Streamlit dashboard
├── outputs/figures/
├── requirements.txt
└── README.md
```

---

## Key Findings

*(Populated after notebooks are run)*

| Metric | Finding |
|--------|---------|
| Total revenue | TBD |
| Best-performing category | TBD |
| 90-day cohort retention | TBD |
| Funnel drop-off (browse → purchase) | TBD |
| RFM Champions share | TBD |
| BigQuery ML model AUC | TBD |

---

## Live Dashboard

*(Link added after Streamlit deployment)*

---

## Technical Stack

| Layer | Tools |
|-------|-------|
| Cloud data warehouse | Google BigQuery |
| Query language | Standard SQL (window functions, CTEs, UNNEST) |
| In-warehouse ML | BigQuery ML (BQML) |
| Data processing | Python, pandas, google-cloud-bigquery |
| Visualisation | plotly, Streamlit |
| Dataset | `bigquery-public-data.thelook_ecommerce` |

---

## Data Source

[TheLook E-commerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) — a synthetic e-commerce dataset provided by Google as a BigQuery public dataset. Contains orders, users, products, events and sessions tables (~125K orders).
