
# Fundamental Dashboard – Metric-row CSV Version

This app is tailored to CSVs like your Nu Holdings files:

- First column: **Metric** (e.g. "Net Income", "Total Assets", "Net Cash Provided By Operating Activities").
- Remaining columns: **TTM, 2024, 2023, 2022, ...**

The app:

- Auto-maps metric names to canonical fields:
  - Income: `revenue`, `net_income`, `operating_income`, `gross_profit`, `interest_expense`, `shares_outstanding`
  - Balance: `total_assets`, `total_equity`, `total_debt`, `cash_and_equivalents`
  - Cash Flow: `cash_from_operations`, `capital_expenditures`
- Builds a time series by period.
- Calculates and visualizes:
  - Overview (revenue, net income, balance sheet size, basic multiples)
  - Growth (YoY revenue & net income)
  - Profitability (margins, ROA, ROE)
  - Risk (leverage ratios, interest coverage)
  - Valuation (multiples, EV/EBIT, etc.)
  - Simple DCF with configurable assumptions

## How to use

1. Put this repo into a GitHub repository.
2. Deploy it on Streamlit Community Cloud with `streamlit_app.py` as the main file.
3. In the app:
   - Upload your Income, Balance Sheet, and Cash Flow CSVs.
   - Enter the current share price and shares outstanding.
   - Explore the tabs.

If some metrics don't map (e.g. unusual label text), you will still see the raw data in the "Raw Data" tab.
You can then adjust the alias lists in `INCOME_METRICS`, `BALANCE_METRICS`, or `CASHFLOW_METRICS` if needed.
