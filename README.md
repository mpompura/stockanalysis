
# Fundamental Dashboard – Metric-row CSV Version (v4)

This app is tailored for CSVs like your Nu Holdings files:

- First column: **Metric** (e.g. "Net Income", "Total Assets", "Net Cash Provided By Operating Activities").
- Remaining columns: **TTM, 2024, 2023, 2022, ...**

Key points:

- TTM column is **ignored** for time-series metrics (only yearly columns used).
- `net_income` maps to the "Net Income to Common Incl Extra Items" row first (so EPS rows are not used as net income).
- Many extra charts per section.
- All Plotly charts use unique `key` values to avoid `StreamlitDuplicateElementId` errors.

Tabs:

- Overview (metrics + line + bar for Revenue & Net Income)
- Growth (YoY growth bars + level lines)
- Profitability (margins, ROA/ROE, margin bars)
- Risk (leverage ratios, cap structure, interest coverage)
- Valuation (multiples + P/S and EV/EBIT history)
- DCF (simple projection + FCF bar chart)
- Raw Data (inspect raw & canonical tables)
