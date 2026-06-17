import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.portfolios import performance_metrics  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

st.set_page_config(page_title="Systematic Funds", layout="wide")
st.title("Systematic Multi-Asset Funds")
st.caption("Out-of-sample funds, current holdings, allocation blends, and sector news sentiment.")


@st.cache_data(show_spinner=False)
def load_outputs():
    fund_returns = pd.read_csv(DATA / "fund_returns.csv", parse_dates=["date"])
    weights = pd.read_csv(DATA / "fund_weights.csv", parse_dates=["rebalance_date"])
    sentiment = pd.read_csv(DATA / "sector_sentiment_index.csv", parse_dates=["date"])
    metrics = pd.read_csv(TABLES / "performance_metrics.csv")
    holdings = pd.read_csv(TABLES / "current_holdings.csv", parse_dates=["rebalance_date"])
    fusion = pd.read_csv(TABLES / "fusion_before_after.csv")
    design = pd.read_csv(TABLES / "backtest_design.csv")
    return fund_returns, weights, sentiment, metrics, holdings, fusion, design


fund_returns, weights, sentiment_df, metrics, holdings, fusion_df, design = load_outputs()
funds = metrics["fund"].tolist()


def fmt_pct(x):
    return f"{x:.2%}"


def metric_cards(row):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. return", fmt_pct(row["annual_return"]))
    c2.metric("Ann. vol", fmt_pct(row["annual_volatility"]))
    c3.metric("Sharpe", f"{row['sharpe']:.2f}")
    c4.metric("Max DD", fmt_pct(row["max_drawdown"]))
    c5.metric("Hit rate", fmt_pct(row["hit_rate"]))


def current_holdings(fund, top_n=15):
    subset = holdings[holdings["fund"].eq(fund)].sort_values("weight", ascending=False).head(top_n)
    return subset[["ticker", "weight", "rebalance_date"]]


tab_funds, tab_allocation, tab_sentiment, tab_method = st.tabs(["Funds", "Allocation", "Sentiment", "Method"])

with tab_funds:
    left, right = st.columns([0.62, 0.38])
    with left:
        selected = st.multiselect(
            "Compare funds",
            funds,
            default=["Combined Equal Weight", "Combined Min Variance", "Equity Sentiment Tilt"],
        )
        if selected:
            growth = (
                fund_returns[fund_returns["fund"].isin(selected)]
                .pivot(index="date", columns="fund", values="growth")
                .dropna(how="all")
            )
            st.line_chart(growth)
            st.dataframe(
                metrics[metrics["fund"].isin(selected)][
                    ["fund", "family", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "first_live_date"]
                ],
                width="stretch",
                hide_index=True,
            )
    with right:
        fund = st.selectbox("Fact sheet", funds, index=funds.index("Combined Min Variance") if "Combined Min Variance" in funds else 0)
        row = metrics[metrics["fund"].eq(fund)].iloc[0]
        metric_cards(row)
        dd = fund_returns[fund_returns["fund"].eq(fund)].set_index("date")["drawdown"]
        st.line_chart(dd)
        st.dataframe(current_holdings(fund), width="stretch", hide_index=True)

with tab_allocation:
    chosen = st.multiselect(
        "Funds",
        funds,
        default=["Combined Min Variance", "Equity Sentiment Tilt", "Crypto Risk Parity"],
        key="allocation_funds",
    )
    if chosen:
        cols = st.columns(len(chosen))
        raw_weights = {}
        for col, fund in zip(cols, chosen):
            raw_weights[fund] = col.slider(fund, 0, 100, int(100 / len(chosen)), 5)
        total = sum(raw_weights.values()) or 1
        alloc = {fund: weight / total for fund, weight in raw_weights.items()}
        pivot = fund_returns[fund_returns["fund"].isin(chosen)].pivot(index="date", columns="fund", values="return").dropna()
        blended = pivot.mul(pd.Series(alloc), axis=1).sum(axis=1)
        blend_growth = (1 + blended).cumprod()
        blend_metrics = performance_metrics(blended, periods_per_year=252)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ann. return", fmt_pct(blend_metrics["annual_return"]))
        c2.metric("Ann. vol", fmt_pct(blend_metrics["annual_volatility"]))
        c3.metric("Sharpe", f"{blend_metrics['sharpe']:.2f}")
        c4.metric("Max DD", fmt_pct(blend_metrics["max_drawdown"]))
        st.line_chart(blend_growth.rename("Blend growth"))
        st.dataframe(
            pd.DataFrame({"fund": list(alloc.keys()), "allocation": list(alloc.values())}),
            width="stretch",
            hide_index=True,
        )

with tab_sentiment:
    sectors = sorted(sentiment_df["sector"].unique())
    selected_sectors = st.multiselect("Sectors", sectors, default=sectors[:4])
    if selected_sectors:
        sector_chart = (
            sentiment_df[sentiment_df["sector"].isin(selected_sectors)]
            .pivot(index="date", columns="sector", values="lagged_signal")
            .dropna(how="all")
        )
        st.line_chart(sector_chart)
        counts = (
            sentiment_df[sentiment_df["sector"].isin(selected_sectors)]
            .groupby("sector", as_index=False)["article_count"]
            .sum()
            .sort_values("article_count", ascending=False)
        )
        st.dataframe(counts, width="stretch", hide_index=True)
    st.subheader("Fusion")
    st.dataframe(
        fusion_df[["fund", "annual_return", "annual_volatility", "sharpe", "max_drawdown"]],
        width="stretch",
        hide_index=True,
    )

with tab_method:
    st.dataframe(design, width="stretch", hide_index=True)
    st.dataframe(metrics.sort_values("sharpe", ascending=False), width="stretch", hide_index=True)
