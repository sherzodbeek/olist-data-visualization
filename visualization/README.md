# Olist Business Dashboard

Interactive dashboard for three Olist departments: **Sales and Marketplace Operations**,
**Logistics and Fulfillment**, and **Customer Experience**. It follows "overview first, zoom and filter,
then details on demand" (Shneiderman, 1996).

## Files

| File | Purpose |
|---|---|
| `data_preparation.ipynb` | Reads the raw Olist tables, checks data quality, cleans and joins them, saves `processed/` |
| `app.py` | The Streamlit dashboard |
| `processed/` | Analysis-ready data created by the notebook (`orders`, `order_items`, `states` as Parquet) |
| `.streamlit/config.toml` | Dashboard theme (follows the system light/dark setting) |
| `requirements.txt` | Python packages used |

## Run it

From the project root:

```bash
# 1. Raw data (skip if data/olist/ already has the CSV files)
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip

# 2. Packages
.venv/bin/python -m pip install -r visualization/requirements.txt

# 3. Prepare the data (or open the notebook and Run All)
cd visualization
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace data_preparation.ipynb

# 4. Start the dashboard (run from the visualization folder so the theme config is picked up)
../.venv/bin/streamlit run app.py
```

The dashboard opens at <http://localhost:8501>. The map needs an internet connection to load country outlines;
every other chart works offline.

## What is in the dashboard

**Filters** (top row, apply to every tab): purchase period, customer region, customer state, product category.
KPI tiles show the change versus the previous period of the same length.

| Tab | Contents |
|---|---|
| Overview | Revenue, orders, average order value, customers, on-time rate, review score; monthly revenue and orders; review score by delivery delay; region scorecard |
| Sales & Marketplace | Top categories, revenue by state, average order value trend, weekday × hour heatmap, seller concentration, top sellers |
| Logistics & Fulfillment | Promised vs actual delivery time, late rate by month and state, delivery-time map, time per fulfillment stage, delivery-time distribution, freight cost, busiest routes |
| Customer Experience | Score distribution and trend, score mix by delivery outcome, late rate vs score by state, lowest-rated categories |
| Order Explorer | Filterable order table, CSV download, single-order lookup with timeline and items |

Every chart has a **Table view** with the exact numbers.

## Definitions

- **Revenue**: product price + freight, excluding canceled and unavailable orders
- **Late**: delivered after the date promised to the customer at checkout
- **Main category / seller** of an order: taken from its most expensive item
- **Analysis window**: January 2017 – August 2018 (2016 and September–October 2018 have too few orders)
