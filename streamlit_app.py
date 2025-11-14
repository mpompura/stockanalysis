
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(
    page_title="Fundamental Dashboard – Metric-row CSV",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================
# Helpers for metric-row CSVs
# =====================

def _norm(s: str) -> str:
    """Normalize strings for fuzzy matching."""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_metric_col(df: pd.DataFrame) -> str | None:
    """Find the metric column (expects something like 'Metric')."""
    for c in df.columns:
        if _norm(c) == "metric":
            return c
    # fallback: first column
    return df.columns[0] if len(df.columns) > 0 else None

def find_best_metric_row(df: pd.DataFrame, aliases) -> pd.Series | None:
    """Return row whose metric label best matches any alias."""
    if df is None or df.empty:
        return None

    metric_col = find_metric_col(df)
    if metric_col is None:
        return None

    metrics_norm = df[metric_col].astype(str).map(_norm)

    # exact normalized match
    for cand in aliases:
        cand_norm = _norm(cand)
        match = df.loc[metrics_norm == cand_norm]
        if not match.empty:
            return match.iloc[0]

    # contains / startswith
    for cand in aliases:
        cand_norm = _norm(cand)
        mask = metrics_norm.apply(lambda x: cand_norm in x or x.startswith(cand_norm))
        match = df.loc[mask]
        if not match.empty:
            return match.iloc[0]

    return None

def parse_period_label(label: str):
    """Parse period labels like '2024', 'Dec 2024' into datetime where possible."""
    label_str = str(label).strip()
    # pure year
    if re.fullmatch(r"\d{4}", label_str):
        try:
            return pd.to_datetime(label_str + "-12-31")
        except Exception:
            return label_str
    # 'Dec 2024'
    m = re.match(r"([A-Za-z]{3})\s+(\d{4})", label_str)
    if m:
        dt = pd.to_datetime(label_str, errors="coerce")
        return dt if not pd.isna(dt) else label_str
    # otherwise keep as string (e.g. TTM)
    return label_str

def metric_table_to_timeseries(df: pd.DataFrame, metric_aliases: dict, drop_ttm: bool = True) -> pd.DataFrame | None:
    """
    Convert a metric-row CSV into a time series.

    Input format:
        - One column 'Metric' (or similar) with metric names.
        - Remaining columns are periods (TTM, 2024, 2023, ...).

    metric_aliases: { canonical_name: [alias1, alias2, ...] }

    Output:
        DataFrame indexed by period, columns = canonical_name.
    """
    if df is None or df.empty:
        return None

    metric_col = find_metric_col(df)
    if metric_col is None:
        st.error(f"Could not find a metric column. Columns: {list(df.columns)}")
        return None

    period_cols = [c for c in df.columns if c != metric_col]

    # Optional: drop TTM-like columns
    period_cols_clean = []
    for c in period_cols:
        if drop_ttm and _norm(c) in ("ttm", "trailing twelve months"):
            continue
        period_cols_clean.append(c)

    if not period_cols_clean:
        st.error("No usable period columns found (after dropping TTM).")
        return None

    # Build result: index = period labels, columns = canonical metrics
    result = pd.DataFrame(index=period_cols_clean)

    for canon, aliases in metric_aliases.items():
        row = find_best_metric_row(df, aliases)
        if row is None:
            # optional, skip if not found
            continue
        vals = pd.to_numeric(row[period_cols_clean], errors="coerce")
        result[canon] = vals

    # Parse periods
    parsed_index = [parse_period_label(lbl) for lbl in result.index]
    result.index = parsed_index

    # Sort index if possible
    try:
        result = result.sort_index()
    except Exception:
        pass

    return result

# =====================
# Canonical metrics & alias lists tuned for your CSVs
# =====================

INCOME_METRICS = {
    # revenue-like line; your CSV has both 'Revenue' and 'Total Revenues'
    "revenue": [
        "Total Revenues",
        "Total Revenue",
        "Revenue",
        "Revenue Before Loan Losses",
    ],
    # IMPORTANT: prefer the "Net Income to Common Incl Extra Items" line over EPS-like rows
    "net_income": [
        "Net Income to Common Incl Extra Items",
        "Net Income to Company",
        "Net Income to Common Excl. Extra Items",
        "Net Income",
    ],
    "operating_income": [
        "EBT, Excl. Unusual Items",
        "EBT, Incl. Unusual Items",
        "Earnings From Continuing Operations",
        "Earnings from Continuing Operations",
        "Operating Income",
        "Income From Operations",
    ],
    # For a bank, "Net Interest Income" is a rough proxy for gross profit
    "gross_profit": [
        "Net Interest Income",
        "Gross Profit",
    ],
    "interest_expense": [
        "Total Interest Expense",
        "Interest On Deposits",
        "Interest Expense",
    ],
    "shares_outstanding": [
        "Basic Weighted Average Shares Outst.",
        "Basic Weighted Average Shares Outst",
        "Diluted Weighted Average Shares Outst.",
        "Diluted Weighted Average Shares Outst",
    ],
}

BALANCE_METRICS = {
    "total_assets": [
        "Total Assets",
        "Assets",
        "Total Assets , End Of Period",
    ],
    "total_equity": [
        "Total Stockholders' Equity",
        "Total Shareholders' Equity",
        "Total Equity",
        "Total Equity Attributable To Shareholders",
    ],
    "total_debt": [
        "Total Debt",
        "Total Interest-Bearing Debt",
        "Total Liabilities And Debt",
        "Short-Term Debt",
        "Long-Term Debt",
    ],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash And Due From Banks",
        "Cash And Short-Term Investments",
        "Cash And Cash Equivalents , End Of Period",
    ],
}

CASHFLOW_METRICS = {
    "cash_from_operations": [
        "Net Cash Provided By Operating Activities",
        "Net Cash From Operating Activities",
        "Net Cash Provided By (Used In) Operating Activities",
        "Cash Flow From Operating Activities",
        "Operating Cash Flow",
    ],
    "capital_expenditures": [
        "Capital Expenditures",
        "Purchase Of Property, Plant And Equipment",
        "Purchase Of Fixed Assets",
        "Acquisition Of Property And Equipment",
    ],
}

# =====================
# Utility funcs for ratios / display
# =====================

def safe_div(a, b):
    """
    Safe division that works for scalars AND pandas Series.
    For Series, performs elementwise division and treats 0 denominators as NaN.
    """
    import pandas as _pd
    import numpy as _np

    # If either is a pandas Series/DataFrame, do elementwise division
    if isinstance(a, (_pd.Series, _pd.DataFrame)) or isinstance(b, (_pd.Series, _pd.DataFrame)):
        try:
            if isinstance(a, _pd.Series):
                a_ser = a
            else:
                if isinstance(b, _pd.Series):
                    a_ser = _pd.Series([a] * len(b), index=b.index)
                else:
                    a_ser = _pd.Series([a])

            if isinstance(b, _pd.Series):
                b_ser = b.replace(0, _np.nan)
            else:
                if isinstance(a, _pd.Series):
                    b_ser = _pd.Series([b] * len(a), index=a.index)
                else:
                    b_ser = _pd.Series([b])

            return a_ser / b_ser
        except Exception:
            if hasattr(a, "index"):
                return _pd.Series(_np.nan, index=a.index)
            if hasattr(b, "index"):
                return _pd.Series(_np.nan, index=b.index)
            return _np.nan

    # Scalar case
    try:
        if b == 0 or _pd.isna(b):
            return _np.nan
        return a / b
    except Exception:
        return _np.nan

def get_latest(series):
    """Return the last non-NaN element of a pandas Series, or handle scalars gracefully."""
    import pandas as _pd
    import numpy as _np
    if series is None:
        return _np.nan
    if not hasattr(series, "dropna"):
        try:
            return float(series)
        except Exception:
            return _np.nan
    s = series.dropna()
    return s.iloc[-1] if not s.empty else _np.nan

def with_period_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "period"})
    return df

def tail_periods(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return last n rows by index (period)."""
    try:
        return df.tail(n)
    except Exception:
        return df

# =====================
# Sidebar
# =====================

st.sidebar.title("Inputs")

company_name = st.sidebar.text_input("Company name", value="Unknown Company")
freq = st.sidebar.radio("Frequency", ["Annual", "Quarterly"], index=0)

st.sidebar.markdown("### Upload metric-row CSVs (Metric + TTM/years)")
income_file = st.sidebar.file_uploader("Income Statement CSV", type=["csv"])
balance_file = st.sidebar.file_uploader("Balance Sheet CSV", type=["csv"])
cashflow_file = st.sidebar.file_uploader("Cash Flow CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Market data")

current_price = st.sidebar.number_input("Current share price", min_value=0.0, value=0.0, step=0.1)
shares_out = st.sidebar.number_input("Shares outstanding", min_value=0.0, value=0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### DCF assumptions")

dcf_growth_years = st.sidebar.slider("Projection years", 3, 15, 5)
dcf_growth_rate = st.sidebar.slider("FCF growth rate (%)", -10.0, 30.0, 10.0, step=0.5)
dcf_discount_rate = st.sidebar.slider("Discount rate (%)", 4.0, 20.0, 10.0, step=0.5)
dcf_terminal_growth = st.sidebar.slider("Terminal growth (%)", 0.0, 6.0, 2.0, step=0.25)

@st.cache_data
def load_csv(file):
    if file is None:
        return None
    return pd.read_csv(file)

income_raw = load_csv(income_file)
balance_raw = load_csv(balance_file)
cashflow_raw = load_csv(cashflow_file)

income = metric_table_to_timeseries(income_raw, INCOME_METRICS) if income_raw is not None else None
balance = metric_table_to_timeseries(balance_raw, BALANCE_METRICS) if balance_raw is not None else None
cashflow = metric_table_to_timeseries(cashflow_raw, CASHFLOW_METRICS) if cashflow_raw is not None else None

data_ok = (income is not None) and (balance is not None) and (cashflow is not None)

tabs = st.tabs(["Overview", "Growth", "Profitability", "Risk", "Valuation", "DCF", "Raw Data"])

# =====================
# Overview
# =====================

with tabs[0]:
    st.header(f"Overview – {company_name} ({freq})")

    if not data_ok:
        st.warning("Upload all three CSVs to see the dashboard.")
    else:
        rev_latest = get_latest(income.get("revenue"))
        ni_latest = get_latest(income.get("net_income"))
        assets_latest = get_latest(balance.get("total_assets") if balance is not None else None)
        equity_latest = get_latest(balance.get("total_equity") if balance is not None else None)
        debt_latest = get_latest(balance.get("total_debt") if balance is not None else None)
        cash_latest = get_latest(balance.get("cash_and_equivalents") if balance is not None else None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Revenue", f"{rev_latest:,.0f}" if not pd.isna(rev_latest) else "N/A")
        c2.metric("Latest Net Income", f"{ni_latest:,.0f}" if not pd.isna(ni_latest) else "N/A")
        c3.metric(
            "Net Margin",
            f"{safe_div(ni_latest, rev_latest)*100:,.1f}%"
            if (not pd.isna(rev_latest) and not pd.isna(ni_latest) and rev_latest != 0)
            else "N/A",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("Total Assets", f"{assets_latest:,.0f}" if not pd.isna(assets_latest) else "N/A")
        c5.metric("Total Equity", f"{equity_latest:,.0f}" if not pd.isna(equity_latest) else "N/A")
        c6.metric("Total Debt", f"{debt_latest:,.0f}" if not pd.isna(debt_latest) else "N/A")

        market_cap = current_price * shares_out if current_price > 0 and shares_out > 0 else np.nan
        c7, c8, c9 = st.columns(3)
        c7.metric("Cash & Equivalents", f"{cash_latest:,.0f}" if not pd.isna(cash_latest) else "N/A")
        c8.metric("Market Cap", f"{market_cap:,.0f}" if not pd.isna(market_cap) else "N/A")
        pe = safe_div(market_cap, ni_latest)
        c9.metric("P/E", f"{pe:,.1f}" if not pd.isna(pe) else "N/A")

        # Charts: Revenue & Net Income lines + bar of last periods
        if income is not None and "revenue" in income.columns and "net_income" in income.columns:
            df_plot = pd.concat(
                [
                    income["revenue"].rename("Revenue"),
                    income["net_income"].rename("Net Income"),
                ],
                axis=1,
            )
            df_plot = with_period_col(df_plot)
            fig = px.line(df_plot, x="period", y=["Revenue", "Net Income"], markers=True)
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

            df_bar = tail_periods(df_plot.set_index("period"), 5).reset_index()
            fig2 = px.bar(df_bar, x="period", y=["Revenue", "Net Income"], barmode="group")
            fig2.update_layout(legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

# =====================
# Growth
# =====================

with tabs[1]:
    st.header("Growth")

    if not data_ok or income is None or "revenue" not in income.columns or "net_income" not in income.columns:
        st.info("Need revenue and net income mapped from income statement.")
    else:
        rev = income["revenue"]
        ni = income["net_income"]
        rev_growth = rev.pct_change() * 100
        ni_growth = ni.pct_change() * 100

        df_g = pd.concat(
            [
                rev_growth.rename("Revenue growth %"),
                ni_growth.rename("Net income growth %"),
            ],
            axis=1,
        ).dropna(how="all")

        if not df_g.empty:
            df_g = with_period_col(df_g)
            fig = px.bar(df_g, x="period", y=df_g.columns, barmode="group")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        # Also show revenue & net income levels here
        df_levels = pd.concat(
            [rev.rename("Revenue"), ni.rename("Net Income")],
            axis=1,
        )
        df_levels = with_period_col(df_levels)
        fig2 = px.line(df_levels, x="period", y=["Revenue", "Net Income"], markers=True)
        fig2.update_layout(legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

# =====================
# Profitability
# =====================

with tabs[2]:
    st.header("Profitability")

    if not data_ok or income is None:
        st.warning("Upload all three CSVs first.")
    else:
        df = pd.DataFrame(index=income.index)

        if "revenue" in income.columns:
            df["revenue"] = income["revenue"]
        if "gross_profit" in income.columns:
            df["gross_profit"] = income["gross_profit"]
            df["Gross margin %"] = safe_div(df["gross_profit"], df["revenue"]) * 100
        if "operating_income" in income.columns:
            df["operating_income"] = income["operating_income"]
            df["Operating margin %"] = safe_div(df["operating_income"], df["revenue"]) * 100
        if "net_income" in income.columns:
            df["net_income"] = income["net_income"]
            df["Net margin %"] = safe_div(df["net_income"], df["revenue"]) * 100

        # ROA / ROE
        if (
            balance is not None
            and "total_assets" in balance.columns
            and "total_equity" in balance.columns
            and "net_income" in income.columns
        ):
            assets = balance["total_assets"]
            equity = balance["total_equity"]
            ni_aligned = income["net_income"].reindex(assets.index).ffill()

            avg_assets = (assets + assets.shift(1)) / 2
            avg_equity = (equity + equity.shift(1)) / 2

            roa = safe_div(ni_aligned, avg_assets) * 100
            roe = safe_div(ni_aligned, avg_equity) * 100

            latest_roa = get_latest(roa)
            latest_roe = get_latest(roe)

            c1, c2 = st.columns(2)
            c1.metric("Latest ROA", f"{latest_roa:,.1f}%" if not pd.isna(latest_roa) else "N/A")
            c2.metric("Latest ROE", f"{latest_roe:,.1f}%" if not pd.isna(latest_roe) else "N/A")

            roa_roe_df = pd.concat(
                [roa.rename("ROA %"), roe.rename("ROE %")],
                axis=1,
            ).dropna(how="all")

            if not roa_roe_df.empty:
                roa_roe_df = with_period_col(roa_roe_df)
                fig = px.line(roa_roe_df, x="period", y=["ROA %", "ROE %"], markers=True)
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

        margin_cols = [c for c in ["Gross margin %", "Operating margin %", "Net margin %"] if c in df.columns]
        if margin_cols:
            margin_df = df[margin_cols].dropna(how="all")
            if not margin_df.empty:
                margin_df = with_period_col(margin_df)
                fig2 = px.line(margin_df, x="period", y=margin_cols, markers=True)
                fig2.update_layout(legend_title_text="")
                st.plotly_chart(fig2, use_container_width=True)

                # bar of last few periods margins
                md_tail = tail_periods(margin_df.set_index("period"), 5).reset_index()
                fig3 = px.bar(md_tail, x="period", y=margin_cols, barmode="group")
                fig3.update_layout(legend_title_text="")
                st.plotly_chart(fig3, use_container_width=True)

# =====================
# Risk
# =====================

with tabs[3]:
    st.header("Risk")

    if not data_ok:
        st.warning("Upload all three CSVs first.")
    else:
        if balance is None or not all(c in balance.columns for c in ["total_debt", "total_equity", "total_assets", "cash_and_equivalents"]):
            st.info("Need total_debt, total_equity, total_assets, cash_and_equivalents metrics from balance sheet.")
        else:
            debt = balance["total_debt"]
            equity = balance["total_equity"]
            assets = balance["total_assets"]
            cash_eq = balance["cash_and_equivalents"]

            debt_eq = safe_div(debt, equity)
            debt_assets = safe_div(debt, assets)
            net_debt = debt - cash_eq
            net_debt_eq = safe_div(net_debt, equity)

            c1, c2, c3 = st.columns(3)
            c1.metric("Debt/Equity (latest)", f"{get_latest(debt_eq):.2f}")
            c2.metric("Debt/Assets (latest)", f"{get_latest(debt_assets):.2f}")
            c3.metric("Net Debt/Equity (latest)", f"{get_latest(net_debt_eq):.2f}")

            risk_df = pd.concat(
                [
                    debt_eq.rename("Debt/Equity"),
                    debt_assets.rename("Debt/Assets"),
                    net_debt_eq.rename("Net Debt/Equity"),
                ],
                axis=1,
            ).dropna(how="all")

            if not risk_df.empty:
                risk_df = with_period_col(risk_df)
                fig = px.line(risk_df, x="period", y=risk_df.columns, markers=True)
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

                # stacked bar of debt vs equity for last few periods
                cap_df = pd.concat(
                    [debt.rename("Debt"), equity.rename("Equity")],
                    axis=1,
                ).dropna(how="all")
                if not cap_df.empty:
                    cap_tail = tail_periods(cap_df, 5)
                    cap_tail = with_period_col(cap_tail)
                    fig2 = px.bar(cap_tail, x="period", y=["Debt", "Equity"], barmode="stack")
                    fig2.update_layout(legend_title_text="")
                    st.plotly_chart(fig2, use_container_width=True)

        # Interest coverage
        if income is not None and "operating_income" in income.columns and "interest_expense" in income.columns:
            ebit = income["operating_income"]
            int_exp = income["interest_expense"].replace(0, np.nan)
            cov = safe_div(ebit, int_exp)
            st.subheader("Interest coverage")
            if hasattr(cov, "rename"):
                cov_series = cov.rename("Interest coverage")
                cov_latest = get_latest(cov_series)
                st.metric("EBIT / Interest (latest)", f"{cov_latest:,.1f}x" if not pd.isna(cov_latest) else "N/A")
                cov_df = cov_series.dropna()
                if not cov_df.empty:
                    cov_df = with_period_col(cov_df)
                    fig = px.line(cov_df, x="period", y="Interest coverage", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                cov_latest = get_latest(cov)
                st.metric("EBIT / Interest (latest)", f"{cov_latest:,.1f}x" if not pd.isna(cov_latest) else "N/A")
        else:
            st.info("Need operating_income and interest_expense mapped from income statement for interest coverage.")

# =====================
# Valuation
# =====================

with tabs[4]:
    st.header("Valuation")

    if not data_ok:
        st.warning("Upload all three CSVs first.")
    else:
        if current_price <= 0 or shares_out <= 0:
            st.info("Enter current price and shares outstanding in the sidebar.")
        else:
            market_cap = current_price * shares_out

            rev_latest = get_latest(income.get("revenue") if income is not None else None)
            ni_latest = get_latest(income.get("net_income") if income is not None else None)
            equity_latest = get_latest(balance.get("total_equity") if balance is not None else None)
            debt_latest = get_latest(balance.get("total_debt") if balance is not None else None)
            cash_latest = get_latest(balance.get("cash_and_equivalents") if balance is not None else None)
            ev = market_cap
            if not pd.isna(debt_latest):
                ev += debt_latest
            if not pd.isna(cash_latest):
                ev -= cash_latest
            op_inc_latest = get_latest(income.get("operating_income") if income is not None else None)

            pe = safe_div(market_cap, ni_latest)
            ps = safe_div(market_cap, rev_latest)
            pb = safe_div(market_cap, equity_latest)
            ev_sales = safe_div(ev, rev_latest)
            ev_ebit = safe_div(ev, op_inc_latest)

            c1, c2, c3 = st.columns(3)
            c1.metric("Market Cap", f"{market_cap:,.0f}")
            c2.metric("Enterprise Value", f"{ev:,.0f}" if not pd.isna(ev) else "N/A")
            c3.metric(
                "EPS (latest)",
                f"{safe_div(ni_latest, shares_out):.2f}"
                if (shares_out > 0 and not pd.isna(ni_latest))
                else "N/A",
            )

            c4, c5, c6 = st.columns(3)
            c4.metric("P/E", f"{pe:,.1f}" if not pd.isna(pe) else "N/A")
            c5.metric("P/S", f"{ps:,.2f}" if not pd.isna(ps) else "N/A")
            c6.metric("P/B", f"{pb:,.2f}" if not pd.isna(pb) else "N/A")

            c7, c8 = st.columns(2)
            c7.metric("EV/Sales", f"{ev_sales:,.2f}" if not pd.isna(ev_sales) else "N/A")
            c8.metric("EV/EBIT", f"{ev_ebit:,.1f}" if not pd.isna(ev_ebit) else "N/A")

            # History of P/S and optionally EV/EBIT using current price
            if income is not None and "revenue" in income.columns:
                ps_hist = market_cap / income["revenue"]
                ps_hist = ps_hist.rename("P/S (using current price)").dropna()
                if not ps_hist.empty:
                    ps_df = with_period_col(ps_hist.to_frame())
                    fig = px.line(ps_df, x="period", y="P/S (using current price)", markers=True)
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

            if income is not None and "operating_income" in income.columns:
                ev_ebit_hist = safe_div(ev, income["operating_income"])
                ev_ebit_hist = ev_ebit_hist.rename("EV/EBIT (using current price)").dropna()
                if not ev_ebit_hist.empty:
                    ev_df = with_period_col(ev_ebit_hist.to_frame())
                    fig2 = px.line(ev_df, x="period", y="EV/EBIT (using current price)", markers=True)
                    fig2.update_layout(legend_title_text="")
                    st.plotly_chart(fig2, use_container_width=True)

# =====================
# DCF
# =====================

with tabs[5]:
    st.header("DCF (very simple)")

    if not data_ok:
        st.warning("Upload all three CSVs first.")
    else:
        if cashflow is None or not all(c in cashflow.columns for c in ["cash_from_operations", "capital_expenditures"]):
            st.info("Need cash_from_operations and capital_expenditures from cash flow statement.")
        elif shares_out <= 0:
            st.info("Enter shares outstanding in the sidebar for per-share fair value.")
        else:
            fcf = cashflow["cash_from_operations"] + cashflow["capital_expenditures"]
            latest_fcf = get_latest(fcf)
            fcf_non_null = fcf.dropna()
            avg_fcf = fcf_non_null.tail(3).mean() if not fcf_non_null.empty else np.nan

            c1, c2 = st.columns(2)
            c1.metric("Latest FCF", f"{latest_fcf:,.0f}" if not pd.isna(latest_fcf) else "N/A")
            c2.metric("Avg FCF (last 3 periods)", f"{avg_fcf:,.0f}" if not pd.isna(avg_fcf) else "N/A")

            base_fcf = avg_fcf if not pd.isna(avg_fcf) else latest_fcf
            if pd.isna(base_fcf):
                st.error("No valid FCF data for DCF.")
            else:
                g = dcf_growth_rate / 100.0
                r = dcf_discount_rate / 100.0
                tg = dcf_terminal_growth / 100.0
                n = dcf_growth_years

                proj_fcfs = [base_fcf * (1 + g) ** t for t in range(1, n + 1)]
                disc_fcfs = [fcf_t / ((1 + r) ** t) for t, fcf_t in enumerate(proj_fcfs, start=1)]
                term_fcf = proj_fcfs[-1] * (1 + tg)
                term_value = term_fcf / (r - tg) if r > tg else np.nan
                disc_tv = term_value / ((1 + r) ** n) if not pd.isna(term_value) else np.nan

                equity_value = sum(disc_fcfs) + (disc_tv if not pd.isna(disc_tv) else 0)
                fair_value = equity_value / shares_out

                c3, c4 = st.columns(2)
                c3.metric("Intrinsic equity value", f"{equity_value:,.0f}")
                c4.metric("DCF fair value / share", f"{fair_value:,.2f}")

                if current_price > 0:
                    upside = (fair_value / current_price - 1) * 100
                    st.metric("Upside vs current price", f"{upside:,.1f}%")

                st.caption("Toy DCF only. Adjust growth/discount/terminal rates to stress-test.")

                # Show projected FCFs
                years = list(range(1, n + 1))
                proj_df = pd.DataFrame({"Year": years, "Projected FCF": proj_fcfs, "Discounted FCF": disc_fcfs})
                fig = px.bar(proj_df, x="Year", y=["Projected FCF", "Discounted FCF"], barmode="group")
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

# =====================
# Raw Data
# =====================

with tabs[6]:
    st.header("Raw Data / Debug")

    st.subheader("Income (raw)")
    if income_raw is not None:
        st.dataframe(income_raw)
    else:
        st.info("No income CSV uploaded.")

    st.subheader("Income (canonical timeseries)")
    if income is not None:
        st.dataframe(income)
    else:
        st.info("Income mapping failed or metrics missing.")

    st.subheader("Balance Sheet (raw)")
    if balance_raw is not None:
        st.dataframe(balance_raw)
    else:
        st.info("No balance CSV uploaded.")

    st.subheader("Balance Sheet (canonical)")
    if balance is not None:
        st.dataframe(balance)
    else:
        st.info("Balance mapping failed or metrics missing.")

    st.subheader("Cash Flow (raw)")
    if cashflow_raw is not None:
        st.dataframe(cashflow_raw)
    else:
        st.info("No cash flow CSV uploaded.")

    st.subheader("Cash Flow (canonical)")
    if cashflow is not None:
        st.dataframe(cashflow)
    else:
        st.info("Cash flow mapping failed or metrics missing.")
