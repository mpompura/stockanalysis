
# Fundamental Dashboard – Metric-row CSV Version (v3)

This app is tailored for CSVs like your Nu Holdings files:

- First column: **Metric** (e.g. "Net Income", "Total Assets", "Net Cash Provided By Operating Activities").
- Remaining columns: **TTM, 2024, 2023, 2022, ...**

It auto-maps metric names to canonical fields, builds a time series by period, and computes:

- Overview (with extra line + bar charts)
- Growth (growth bars + level lines)
- Profitability (margins, ROA/ROE, margin bars)
- Risk (leverage lines + capital structure bars, interest coverage)
- Valuation (multiples and historical lines)
- Simple DCF (with projected FCF bar chart)
- Raw data debug views

This v3 version:
- Prefers the "Net Income to Common Incl Extra Items" row as net_income instead of the EPS row.
- Drops TTM from time series (uses 2024, 2023, ... as periods; TTM is ignored for growth/ratios).
- Adds several additional charts in each category.
