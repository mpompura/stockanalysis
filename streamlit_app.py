
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import PyPDF2
import io
import re

st.set_page_config(
    page_title="Fundamental Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CONFIG: CANONICAL FIELDS & ALIASES (AUTO-MAPPING) ----------

CANONICAL = {
    "income": ["period", "revenue", "gross_profit", "operating_income", "net_income", "interest_expense", "shares_outstanding"],
    "balance": ["period", "total_assets", "total_equity", "total_debt", "cash_and_equivalents"],
    "cashflow": ["period", "cash_from_operations", "capital_expenditures"],
}

ALIASES = {
    "income": {
        "period": [
            "period", "date", "year", "fiscal year", "fiscal year ending",
            "year ending", "period ending", "year end", "fiscal year end"
        ],
        "revenue": [
            "revenue", "total revenue", "net revenue", "total net revenue",
            "total income", "sales", "total operating revenue"
        ],
        "gross_profit": [
            "gross profit", "gross income"
        ],
        "operating_income": [
            "operating income", "operating income (loss)", "income from operations",
            "operating profit", "operating earnings"
        ],
        "net_income": [
            "net income", "net income (loss)", "net earnings", "net profit",
            "net income attributable to common shareholders", "net income to common shareholders"
        ],
        "interest_expense": [
            "interest expense", "interest and similar expense", "interest expenses", "total interest expense"
        ],
        "shares_outstanding": [
            "shares outstanding", "diluted shares outstanding", "weighted average shares diluted",
            "weighted average shares outstanding diluted", "weighted average diluted shares outstanding"
        ],
    },
    "balance": {
        "period": [
            "period", "date", "year", "fiscal year", "fiscal year ending",
            "year ending", "period ending", "year end", "fiscal year end"
        ],
        "total_assets": [
            "total assets", "assets", "assets total", "total assets, end of period", "total assets (millions)"
        ],
        "total_equity": [
            "total equity", "total shareholders' equity", "shareholders equity",
            "stockholders equity", "total stockholders' equity", "total equity attributable to shareholders"
        ],
        "total_debt": [
            "total debt", "total liabilities and debt", "total interest-bearing debt",
            "short-term debt", "long-term debt", "total borrowings"
        ],
        "cash_and_equivalents": [
            "cash and cash equivalents", "cash and equivalents", "cash & equivalents",
            "cash and due from banks", "cash", "cash and short-term investments"
        ],
    },
    "cashflow": {
        "period": [
            "period", "date", "year", "fiscal year", "fiscal year ending",
            "year ending", "period ending", "year end", "fiscal year end"
        ],
        "cash_from_operations": [
            "cash from operations", "net cash from operating activities",
            "net cash provided by operating activities", "cash flow from operating activities",
            "operating cash flow", "net cash provided by (used in) operating activities"
        ],
        "capital_expenditures": [
            "capital expenditures", "capital expenditure", "capex",
            "purchase of property, plant and equipment", "purchase of fixed assets",
            "acquisition of property and equipment"
        ],
    },
}


def _normalize_name(name: str) -> str:
    """Normalize column names for fuzzy matching."""
    name = name.lower()
    # Replace non-alphanumeric with space
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def maybe_promote_first_row_to_header(df: pd.DataFrame) -> pd.DataFrame:
    """If columns look like generic 0,1,2,... treat first row as header."""
    if df is None or df.empty:
        return df
    # If all columns are numeric-like (0,1,2,...) it's probably from PDF helper with header=None
    try:
        col_strs = [str(c) for c in df.columns]
        if all(re.fullmatch(r"\d+", c) for c in col_strs):
            new_header = df.iloc[0].astype(str)
            df = df[1:].copy()
            df.columns = new_header
    except Exception:
        pass
    return df


def find_best_column(df: pd.DataFrame, candidates) -> str | None:
    """Find the first matching column in df among candidate names (using fuzzy matching)."""
    if df is None or df.empty:
        return None

    norm_cols = { _normalize_name(c): c for c in df.columns }

    for cand in candidates:
        cand_norm = _normalize_name(cand)
        if cand_norm in norm_cols:
            return norm_cols[cand_norm]

    # Also try startswith / contains matching for robustness
    for cand in candidates:
        cand_norm = _normalize_name(cand)
        for col_norm, col_orig in norm_cols.items():
            if col_norm.startswith(cand_norm) or cand_norm in col_norm:
                return col_orig

    return None


def standardize_df(df: pd.DataFrame, section: str) -> pd.DataFrame | None:
    """Return a new DataFrame with canonical column names for the given section.

    It will try to auto-map provider-specific column names using ALIASES.
    """
    if df is None:
        return None

    # Try to fix case where columns are 0,1,2,... from PDF helper CSV
    df = maybe_promote_first_row_to_header(df)

    df = df.copy()
    canonical_cols = CANONICAL[section]
    mapping = {}
    missing = []

    for canon in canonical_cols:
        alias_list = ALIASES.get(section, {}).get(canon, [])
        # Always consider canonical name itself as candidate
        candidates = [canon] + alias_list
        actual = find_best_column(df, candidates)
        if actual is not None:
            mapping[canon] = actual
        else:
            # period is mandatory; others can be missing (we'll just get NaNs)
            if canon == "period":
                st.error(
                    f"Could not find a suitable column for '{canon}' in {section} statement. "
                    f"Columns found: {list(df.columns)}"
                )
                return None
            missing.append(canon)

    if missing:
        st.info(
            f"{section.capitalize()} statement: could not find columns for {missing}. "
            "Those metrics will simply be skipped."
        )

    # Build standardized DataFrame with canonical columns
    std = pd.DataFrame()
    period_col = mapping["period"]
    std["period"] = df[period_col]

    for canon, actual in mapping.items():
        if canon == "period":
            continue
        std[canon] = df[actual]

    return std


# ---------- UTILS ----------

@st.cache_data
def load_table(file, sheet_name=None):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file, sheet_name=sheet_name)
    elif name.endswith(".pdf"):
        # Very simple PDF-to-table heuristic: used only in PDF Helper.
        try:
            reader = PyPDF2.PdfReader(file)
            text_pages = []
            for page in reader.pages:
                try:
                    text_pages.append(page.extract_text() or "")
                except Exception:
                    continue
            full_text = "\n".join(text_pages)

            # Heuristic: collapse multiple spaces into a comma; keep only lines containing digits
            lines = full_text.splitlines()
            processed_lines = []
            for ln in lines:
                if any(ch.isdigit() for ch in ln):
                    s = re.sub(r"\s+", ",", ln.strip())
                    if s.count(",") >= 1:
                        processed_lines.append(s)
            if not processed_lines:
                st.error("Could not find any table-like lines in this PDF.")
                return None
            csv_like = "\n".join(processed_lines)
            df = pd.read_csv(io.StringIO(csv_like), header=None, on_bad_lines="skip")
        except Exception:
            st.error("Failed to parse this PDF into a table. "
                     "Try opening it manually and copy-pasting into Excel, then upload the cleaned CSV.")
            return None
    else:
        st.error("Unsupported file type. Please use CSV or Excel for the main statements, "
                 "and use the PDF Helper tab for rough PDF extraction.")
        return None
    return df


def normalize_period(df, period_col: str):
    if df is None:
        return None
    df = df.copy()
    if period_col not in df.columns:
        st.error(f"Expected period column '{period_col}' not found in uploaded file. Columns: {list(df.columns)}")
        return None
    # Try parse to datetime; fall back to string
    try:
        df[period_col] = pd.to_datetime(df[period_col])
    except Exception:
        df[period_col] = df[period_col].astype(str)
    df = df.sort_values(period_col)
    df = df.set_index(period_col)
    # Convert numeric-like columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def get_latest_value(df, col):
    try:
        return df[col].dropna().iloc[-1]
    except Exception:
        return np.nan


def compute_cagr(series, years):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    end = series.iloc[-1]
    start_idx = 0
    if len(series) > years:
        start_idx = len(series) - years - 1
    start = series.iloc[start_idx]
    if start <= 0 or end <= 0:
        return np.nan
    n_periods = len(series) - 1 - start_idx
    if n_periods <= 0:
        return np.nan
    return (end / start) ** (1 / n_periods) - 1


def safe_div(numerator, denominator):
    try:
        if denominator == 0 or pd.isna(denominator):
            return np.nan
        return numerator / denominator
    except Exception:
        return np.nan


def with_period_column(df):
    df = df.reset_index()
    first_col = df.columns[0]
    if first_col != "period":
        df = df.rename(columns={first_col: "period"})
    return df


# ---------- SIDEBAR: INPUTS / FILES ----------

st.sidebar.title("Input Data")

company_name = st.sidebar.text_input("Company name", value="Unknown Company")

freq = st.sidebar.radio("Data frequency", ["Annual", "Quarterly"], index=0)

st.sidebar.markdown("### Upload financial statements (CSV or Excel)")

income_file = st.sidebar.file_uploader("Income Statement", type=["csv", "xlsx"])
balance_file = st.sidebar.file_uploader("Balance Sheet", type=["csv", "xlsx"])
cashflow_file = st.sidebar.file_uploader("Cash Flow Statement", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Market data (for valuation)")

current_price = st.sidebar.number_input("Current share price", min_value=0.0, value=0.0, step=0.1)
shares_out = st.sidebar.number_input("Shares outstanding (latest, in same units as statements)", min_value=0.0, value=0.0, step=1.0)

st.sidebar.markdown("If you leave these at 0, valuation ratios won’t be computed.")

st.sidebar.markdown("---")
st.sidebar.markdown("### DCF assumptions")

dcf_growth_years = st.sidebar.slider("Projection years", 3, 15, 5)
dcf_growth_rate = st.sidebar.slider("Growth rate for FCF (%)", -10.0, 30.0, 10.0, step=0.5)
dcf_discount_rate = st.sidebar.slider("Discount rate (%)", 4.0, 20.0, 10.0, step=0.5)
dcf_terminal_growth = st.sidebar.slider("Terminal growth (%)", 0.0, 6.0, 2.0, step=0.25)


# ---------- LOAD & PREPARE DATA ----------

income_raw = load_table(income_file)
balance_raw = load_table(balance_file)
cashflow_raw = load_table(cashflow_file)

# Standardize columns automatically using aliases
income_std = standardize_df(income_raw, "income") if income_raw is not None else None
balance_std = standardize_df(balance_raw, "balance") if balance_raw is not None else None
cashflow_std = standardize_df(cashflow_raw, "cashflow") if cashflow_raw is not None else None

if income_std is not None:
    income = normalize_period(income_std, "period")
else:
    income = None

if balance_std is not None:
    balance = normalize_period(balance_std, "period")
else:
    balance = None

if cashflow_std is not None:
    cashflow = normalize_period(cashflow_std, "period")
else:
    cashflow = None

data_ok = (income is not None) and (balance is not None) and (cashflow is not None)

# ---------- MAIN LAYOUT ----------

st.title(f"Fundamental Dashboard: {company_name}")
st.caption(f"Frequency: {freq}. Upload new statements to switch companies. Column names are auto-mapped where possible.")

tabs = st.tabs([
    "Overview",
    "Growth",
    "Profitability",
    "Risk",
    "Valuation",
    "DCF / Fair Value",
    "Raw Data",
    "PDF Helper",
])


# ---------- OVERVIEW TAB ----------

with tabs[0]:
    st.header("Overview")

    if not data_ok:
        st.warning("Upload income, balance sheet and cash flow statements (CSV/Excel) to see the dashboard.")
    else:
        rev_latest = get_latest_value(income, "revenue")
        ni_latest = get_latest_value(income, "net_income")
        assets_latest = get_latest_value(balance, "total_assets")
        equity_latest = get_latest_value(balance, "total_equity")
        debt_latest = get_latest_value(balance, "total_debt")
        cash_latest = get_latest_value(balance, "cash_and_equivalents")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Revenue", f"{rev_latest:,.0f}" if not pd.isna(rev_latest) else "N/A")
        col2.metric("Latest Net Income", f"{ni_latest:,.0f}" if not pd.isna(ni_latest) else "N/A")
        col3.metric("Latest Net Margin",
                    f"{safe_div(ni_latest, rev_latest)*100:,.1f}%"
                    if not pd.isna(rev_latest) and not pd.isna(ni_latest) else "N/A")

        col4, col5, col6 = st.columns(3)
        col4.metric("Total Assets", f"{assets_latest:,.0f}" if not pd.isna(assets_latest) else "N/A")
        col5.metric("Total Equity", f"{equity_latest:,.0f}" if not pd.isna(equity_latest) else "N/A")
        col6.metric("Total Debt", f"{debt_latest:,.0f}" if not pd.isna(debt_latest) else "N/A")

        if current_price > 0 and shares_out > 0:
            market_cap = current_price * shares_out
        else:
            market_cap = np.nan

        col7, col8, col9 = st.columns(3)
        col7.metric("Cash & Equivalents", f"{cash_latest:,.0f}" if not pd.isna(cash_latest) else "N/A")
        col8.metric("Market Cap", f"{market_cap:,.0f}" if not pd.isna(market_cap) else "N/A")
        if not pd.isna(market_cap) and not pd.isna(ni_latest) and ni_latest != 0:
            pe = market_cap / ni_latest
        else:
            pe = np.nan
        col9.metric("P/E (latest)", f"{pe:,.1f}" if not pd.isna(pe) else "N/A")

        st.subheader("Revenue and Net Income over time")
        if "revenue" in income.columns and "net_income" in income.columns:
            rev_series = income["revenue"].rename("Revenue")
            ni_series = income["net_income"].rename("Net Income")
            plot_df = pd.concat([rev_series, ni_series], axis=1)
            plot_df = with_period_column(plot_df)
            fig = px.line(plot_df, x="period", y=["Revenue", "Net Income"], markers=True)
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue or net income column is missing, cannot plot overview chart.")


# ---------- GROWTH TAB ----------

with tabs[1]:
    st.header("Growth")

    if not data_ok:
        st.warning("Upload all three statements first.")
    else:
        if "revenue" not in income.columns or "net_income" not in income.columns:
            st.info("Need revenue and net_income columns to compute growth. Check that they exist in your data.")
        else:
            revenue = income["revenue"]
            net_income_series = income["net_income"]

            growth_horizon = st.selectbox("Growth horizon (for CAGR)", [3, 4, 5, 7, 10], index=2)
            rev_cagr = compute_cagr(revenue, growth_horizon)
            ni_cagr = compute_cagr(net_income_series, growth_horizon)

            col1, col2 = st.columns(2)
            col1.metric(f"Revenue CAGR (approx, last {growth_horizon} periods)",
                        f"{rev_cagr*100:,.1f}%" if not pd.isna(rev_cagr) else "N/A")
            col2.metric(f"Net Income CAGR (approx, last {growth_horizon} periods)",
                        f"{ni_cagr*100:,.1f}%" if not pd.isna(ni_cagr) else "N/A")

            st.subheader("YoY / QoQ growth")
            growth_df = income[["revenue", "net_income"]].copy()
            growth_df["Revenue growth %"] = growth_df["revenue"].pct_change() * 100
            growth_df["Net income growth %"] = growth_df["net_income"].pct_change() * 100

            plot_df = growth_df[["Revenue growth %", "Net income growth %"]].dropna().copy()
            if not plot_df.empty:
                plot_df = with_period_column(plot_df)
                fig = px.bar(plot_df, x="period", y=["Revenue growth %", "Net income growth %"], barmode="group")
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two periods to calculate growth.")


# ---------- PROFITABILITY TAB ----------

with tabs[2]:
    st.header("Profitability")

    if not data_ok:
        st.warning("Upload all three statements first.")
    else:
        needed = ["revenue", "gross_profit", "operating_income", "net_income"]
        if not all(col in income.columns for col in needed):
            st.info(f"Missing one of required columns {needed} in income statement. "
                    "Some profitability metrics cannot be computed.")
        else:
            df = income[needed].copy()
            df["Gross margin %"] = safe_div(df["gross_profit"], df["revenue"]) * 100
            df["Operating margin %"] = safe_div(df["operating_income"], df["revenue"]) * 100
            df["Net margin %"] = safe_div(df["net_income"], df["revenue"]) * 100

            # ROE & ROA using average assets/equity
            if "total_assets" in balance.columns and "total_equity" in balance.columns:
                assets = balance["total_assets"]
                equity = balance["total_equity"]
                ni_aligned = income["net_income"].reindex(assets.index).fillna(method="ffill")
                avg_assets = (assets + assets.shift(1)) / 2
                avg_equity = (equity + equity.shift(1)) / 2

                roe = safe_div(ni_aligned, avg_equity) * 100
                roa = safe_div(ni_aligned, avg_assets) * 100

                latest_roe = roe.dropna().iloc[-1] if roe.dropna().size > 0 else np.nan
                latest_roa = roa.dropna().iloc[-1] if roa.dropna().size > 0 else np.nan

                col1, col2 = st.columns(2)
                col1.metric("Latest ROE", f"{latest_roe:,.1f}%" if not pd.isna(latest_roe) else "N/A")
                col2.metric("Latest ROA", f"{latest_roa:,.1f}%" if not pd.isna(latest_roa) else "N/A")

                st.subheader("Margins over time")
                plot_df = df[["Gross margin %", "Operating margin %", "Net margin %"]].dropna(how="all")
                if not plot_df.empty:
                    plot_df = with_period_column(plot_df)
                    fig = px.line(plot_df, x="period", y=["Gross margin %", "Operating margin %", "Net margin %"], markers=True)
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("ROE and ROA over time")
                roa_roe_df = pd.concat([roa.rename("ROA %"), roe.rename("ROE %")], axis=1).dropna(how="all")
                if not roa_roe_df.empty:
                    roa_roe_df = with_period_column(roa_roe_df)
                    fig2 = px.line(roa_roe_df, x="period", y=["ROA %", "ROE %"], markers=True)
                    fig2.update_layout(legend_title_text="")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Need at least two periods of assets/equity and income for ROA/ROE trends.")
            else:
                st.info("Balance sheet missing total_assets or total_equity; ROA/ROE cannot be computed.")


# ---------- RISK TAB ----------

with tabs[3]:
    st.header("Risk")

    if not data_ok:
        st.warning("Upload all three statements first.")
    else:
        if not all(col in balance.columns for col in ["total_debt", "total_equity", "total_assets", "cash_and_equivalents"]):
            st.info("Balance sheet missing one of total_debt, total_equity, total_assets, cash_and_equivalents.")
        else:
            debt = balance["total_debt"]
            equity = balance["total_equity"]
            assets = balance["total_assets"]
            cash = balance["cash_and_equivalents"]

            debt_to_equity = safe_div(debt, equity)
            debt_to_assets = safe_div(debt, assets)
            net_debt = debt - cash
            net_debt_to_equity = safe_div(net_debt, equity)

            latest_de = debt_to_equity.dropna().iloc[-1] if debt_to_equity.dropna().size > 0 else np.nan
            latest_da = debt_to_assets.dropna().iloc[-1] if debt_to_assets.dropna().size > 0 else np.nan
            latest_nde = net_debt_to_equity.dropna().iloc[-1] if net_debt_to_equity.dropna().size > 0 else np.nan

            col1, col2, col3 = st.columns(3)
            col1.metric("Debt / Equity (latest)", f"{latest_de:,.2f}" if not pd.isna(latest_de) else "N/A")
            col2.metric("Debt / Assets (latest)", f"{latest_da:,.2f}" if not pd.isna(latest_da) else "N/A")
            col3.metric("Net Debt / Equity (latest)", f"{latest_nde:,.2f}" if not pd.isna(latest_nde) else "N/A")

            st.subheader("Leverage ratios over time")
            risk_df = pd.concat(
                [
                    debt_to_equity.rename("Debt/Equity"),
                    debt_to_assets.rename("Debt/Assets"),
                    net_debt_to_equity.rename("Net Debt/Equity"),
                ],
                axis=1,
            ).dropna(how="all")
            if not risk_df.empty:
                risk_df = with_period_column(risk_df)
                fig = px.line(risk_df, x="period", y=["Debt/Equity", "Debt/Assets", "Net Debt/Equity"], markers=True)
                fig.update_layout(legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

        # Interest coverage
        if "interest_expense" in income.columns and "operating_income" in income.columns:
            try:
                op_inc = income["operating_income"]
                interest_exp = income["interest_expense"].replace(0, np.nan)
                int_cov = safe_div(op_inc, interest_exp)
                latest_cov = int_cov.dropna().iloc[-1]
                st.subheader("Interest coverage")
                st.metric("Latest interest coverage (EBIT / interest)", f"{latest_cov:,.1f}x")
                ic_df = int_cov.rename("Interest coverage").dropna()
                if not ic_df.empty:
                    ic_df = with_period_column(ic_df)
                    fig2 = px.line(ic_df, x="period", y="Interest coverage", markers=True)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("Could not compute interest coverage (check interest_expense data).")
        else:
            st.info("Income statement missing interest_expense or operating_income; interest coverage not computed.")


# ---------- VALUATION TAB ----------

with tabs[4]:
    st.header("Valuation")

    if not data_ok:
        st.warning("Upload all three statements first.")
    else:
        if current_price <= 0 or shares_out <= 0:
            st.warning("Enter current share price and shares outstanding in the sidebar to compute valuation ratios.")
        else:
            market_cap = current_price * shares_out

            rev_latest = get_latest_value(income, "revenue")
            ni_latest = get_latest_value(income, "net_income")
            equity_latest = get_latest_value(balance, "total_equity")
            debt_latest = get_latest_value(balance, "total_debt")
            cash_latest = get_latest_value(balance, "cash_and_equivalents")

            ev = market_cap
            if not pd.isna(debt_latest):
                ev += debt_latest
            if not pd.isna(cash_latest):
                ev -= cash_latest

            op_income_latest = get_latest_value(income, "operating_income") if "operating_income" in income.columns else np.nan

            pe = safe_div(market_cap, ni_latest)
            ps = safe_div(market_cap, rev_latest)
            pb = safe_div(market_cap, equity_latest)
            ev_sales = safe_div(ev, rev_latest)
            ev_ebit = safe_div(ev, op_income_latest)

            col1, col2, col3 = st.columns(3)
            col1.metric("Market Cap", f"{market_cap:,.0f}")
            col2.metric("Enterprise Value (approx)", f"{ev:,.0f}")
            col3.metric("Latest EPS (approx)",
                        f"{safe_div(ni_latest, shares_out):.2f}" if shares_out > 0 and not pd.isna(ni_latest) else "N/A")

            col4, col5, col6 = st.columns(3)
            col4.metric("P/E", f"{pe:,.1f}" if not pd.isna(pe) else "N/A")
            col5.metric("P/S", f"{ps:,.2f}" if not pd.isna(ps) else "N/A")
            col6.metric("P/B", f"{pb:,.2f}" if not pd.isna(pb) else "N/A")

            col7, col8 = st.columns(2)
            col7.metric("EV / Sales", f"{ev_sales:,.2f}" if not pd.isna(ev_sales) else "N/A")
            col8.metric("EV / EBIT (proxy)", f"{ev_ebit:,.1f}" if not pd.isna(ev_ebit) else "N/A")

            st.subheader("Historical P/S (using current price)")
            if "revenue" in income.columns:
                ps_hist = market_cap / income["revenue"]
                ps_df = ps_hist.rename("P/S (using current price)").dropna()
                if not ps_df.empty:
                    ps_df = with_period_column(ps_df)
                    fig = px.line(ps_df, x="period", y="P/S (using current price)", markers=True)
                    st.plotly_chart(fig, use_container_width=True)


# ---------- DCF TAB ----------

with tabs[5]:
    st.header("DCF / Fair Value (very simple model)")

    if not data_ok:
        st.warning("Upload all three statements first.")
    else:
        if not all(col in cashflow.columns for col in ["cash_from_operations", "capital_expenditures"]):
            st.info("Cash flow statement missing cash_from_operations or capital_expenditures; cannot compute FCF.")
        else:
            cfo = cashflow["cash_from_operations"]
            capex = cashflow["capital_expenditures"]
            fcf = cfo + capex  # if capex is negative, this is CFO - |Capex|

            st.subheader("Historical Free Cash Flow")
            fcf_df = fcf.rename("FCF").dropna()
            if not fcf_df.empty:
                fcf_df = with_period_column(fcf_df)
                fig = px.bar(fcf_df, x="period", y="FCF")
                st.plotly_chart(fig, use_container_width=True)

            latest_fcf = fcf.dropna().iloc[-1] if fcf.dropna().size > 0 else np.nan
            avg_fcf = fcf.dropna().tail(3).mean() if fcf.dropna().size > 0 else np.nan

            col1, col2 = st.columns(2)
            col1.metric("Latest FCF", f"{latest_fcf:,.0f}" if not pd.isna(latest_fcf) else "N/A")
            col2.metric("Average FCF (last 3 periods)", f"{avg_fcf:,.0f}" if not pd.isna(avg_fcf) else "N/A")

            if shares_out <= 0:
                st.warning("Enter shares outstanding in the sidebar to get per-share fair value.")
            else:
                base_fcf = avg_fcf if not pd.isna(avg_fcf) else latest_fcf
                if pd.isna(base_fcf):
                    st.error("Cannot compute DCF – no valid FCF data.")
                else:
                    g = dcf_growth_rate / 100.0
                    r = dcf_discount_rate / 100.0
                    tg = dcf_terminal_growth / 100.0
                    n = dcf_growth_years

                    projected_fcfs = []
                    for t in range(1, n + 1):
                        fcf_t = base_fcf * ((1 + g) ** t)
                        projected_fcfs.append(fcf_t)

                    discounted_fcfs = [fcf_t / ((1 + r) ** t) for t, fcf_t in enumerate(projected_fcfs, start=1)]

                    terminal_fcf = projected_fcfs[-1] * (1 + tg)
                    terminal_value = terminal_fcf / (r - tg) if r > tg else np.nan
                    discounted_tv = terminal_value / ((1 + r) ** n) if not pd.isna(terminal_value) else np.nan

                    intrinsic_equity_value = sum(discounted_fcfs) + (discounted_tv if not pd.isna(discounted_tv) else 0)
                    fair_value_per_share = intrinsic_equity_value / shares_out

                    col3, col4 = st.columns(2)
                    col3.metric("Intrinsic equity value", f"{intrinsic_equity_value:,.0f}")
                    col4.metric("DCF fair value / share", f"{fair_value_per_share:,.2f}")

                    if current_price > 0:
                        upside = (fair_value_per_share / current_price - 1) * 100
                        st.metric("Upside vs current price", f"{upside:,.1f}%")

                    st.caption("This is a very simplified DCF. Change growth/discount/terminal rates in the sidebar to stress-test scenarios.")


# ---------- RAW DATA TAB ----------

with tabs[6]:
    st.header("Raw Data")

    st.subheader("Income Statement (standardized)")
    if income is not None:
        st.dataframe(income)
    else:
        st.info("No income statement uploaded or mapping failed.")

    st.subheader("Balance Sheet (standardized)")
    if balance is not None:
        st.dataframe(balance)
    else:
        st.info("No balance sheet uploaded or mapping failed.")

    st.subheader("Cash Flow Statement (standardized)")
    if cashflow is not None:
        st.dataframe(cashflow)
    else:
        st.info("No cash flow statement uploaded or mapping failed.")


# ---------- PDF HELPER TAB ----------

with tabs[7]:
    st.header("PDF Helper (experimental)")
    st.write(
        "Upload a PDF financial statement here. The app will try a very simple text extraction and "
        "convert it into a rough table. You will usually still need to clean the result manually and then "
        "save it as CSV/Excel for the main dashboard. Column names in CSV/Excel are then auto-mapped to "
        "canonical fields (revenue, net income, etc.) where possible."
    )

    pdf_file = st.file_uploader("Upload a PDF file to preview parsed table", type=["pdf"], key="pdf_helper_uploader")
    if pdf_file is not None:
        tmp_df = load_table(pdf_file)
        if tmp_df is None or tmp_df.empty:
            st.error("Could not extract a usable table from this PDF with the simple parser. "
                     "You may need to open it manually and copy-paste into Excel/CSV.")
        else:
            st.subheader("Extracted table (best-effort)")
            st.dataframe(tmp_df)

            csv_bytes = tmp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV (optional: clean/rename locally, then upload in main sidebar)",
                data=csv_bytes,
                file_name="parsed_from_pdf.csv",
                mime="text/csv",
            )
