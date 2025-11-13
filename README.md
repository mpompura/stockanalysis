# Fundamental Stock Dashboard (Streamlit)

This app lets you upload **income statement**, **balance sheet**, and **cash flow statement** data (annual or quarterly)
and gives you an interactive dashboard with:

- Growth metrics (CAGR, YoY / QoQ)
- Profitability (margins, ROE, ROA)
- Risk (leverage, net debt, interest coverage)
- Valuation ratios (P/E, P/S, P/B, EV multiples)
- A simple DCF-based "fair value per share"

## Data format

Upload CSV or Excel files with one row per period.

### Income statement

Required columns:

- `period` (e.g. `2021-12-31` or `2021`)
- `revenue`
- `gross_profit`
- `operating_income`
- `net_income`

Optional columns:

- `interest_expense` (for interest coverage)
- `shares_outstanding` (per-share metrics; you can also enter this in the sidebar)

### Balance sheet

- `period`
- `total_assets`
- `total_equity`
- `total_debt`
- `cash_and_equivalents`

### Cash flow statement

- `period`
- `cash_from_operations`
- `capital_expenditures` (use negative numbers if it's an outflow; app handles both)

If your column names differ, edit the `COLUMN_MAP` dictionary at the top of `streamlit_app.py`.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On Streamlit Community Cloud, create a new app from that repo.
3. Set the main file to `streamlit_app.py`.
4. Deploy.

Then, in the web app:

1. Type the **company name** in the sidebar.
2. Choose **Annual** or **Quarterly** (for labeling).
3. Upload income, balance, and cash flow CSV/Excel files.
4. Optionally set **current price** and **shares outstanding**.
5. Adjust DCF parameters and explore the tabs.
