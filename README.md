
# Fundamental Dashboard – Metric-row CSV Version (v2)

This app is tailored for CSVs like your Nu Holdings files:

- First column: **Metric** (e.g. "Net Income", "Total Assets", "Net Cash Provided By Operating Activities").
- Remaining columns: **TTM, 2024, 2023, 2022, ...**

It auto-maps metric names to canonical fields, builds a time series by period, and computes:

- Overview
- Growth
- Profitability (margins, ROA, ROE)
- Risk (leverage, interest coverage)
- Valuation (multiples, EV/EBIT)
- Simple DCF
- Raw data debug views

This v2 version has more robust handling for Series vs scalar math (safe_div, get_latest, interest coverage).
