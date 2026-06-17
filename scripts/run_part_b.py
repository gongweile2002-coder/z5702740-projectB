"""Reproduce Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
from __future__ import annotations

import pathlib
import sys
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl, features, fusion, portfolios, sentiment  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA = RESULTS / "data"
REPORT = ROOT / "report"

PALETTE = {
    "Combined": "#1f77b4",
    "Equity": "#2ca02c",
    "Crypto": "#d62728",
    "Fusion": "#9467bd",
}


def _ensure_dirs() -> None:
    for path in [FIGURES, TABLES, DATA, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def _currency_pct(x: float) -> str:
    return f"{x:.2%}"


def _savefig(path: pathlib.Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_growth(fund_returns: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.5))
    pivot = fund_returns.pivot(index="date", columns="fund", values="growth")
    for fund in pivot.columns:
        family = fund_returns.loc[fund_returns["fund"].eq(fund), "family"].iloc[0]
        plt.plot(pivot.index, pivot[fund], label=fund, linewidth=1.8, alpha=0.88, color=PALETTE.get(family))
    plt.axhline(1, color="#555555", linewidth=0.8)
    plt.title("Growth of $1: out-of-sample funds")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend(ncol=2, fontsize=8)
    _savefig(FIGURES / "growth_of_one.png")


def _plot_drawdown(fund_returns: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.2))
    focus = fund_returns[fund_returns["family"].isin(["Combined", "Equity"])]
    pivot = focus.pivot(index="date", columns="fund", values="drawdown")
    for fund in pivot.columns:
        plt.plot(pivot.index, pivot[fund], label=fund, linewidth=1.6)
    plt.title("Drawdown: combined and equity funds")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend(ncol=2, fontsize=8)
    _savefig(FIGURES / "drawdowns.png")


def _plot_weights(weights: pd.DataFrame, fund: str = "Combined Min Variance") -> None:
    latest = weights[weights["fund"].eq(fund)].copy()
    if latest.empty:
        return
    pivot = latest.pivot_table(index="rebalance_date", columns="ticker", values="weight", aggfunc="sum").fillna(0)
    top = pivot.mean().sort_values(ascending=False).head(12).index
    plt.figure(figsize=(10, 5.4))
    plt.stackplot(pivot.index, [pivot[t] for t in top], labels=top, alpha=0.92)
    plt.title(f"Weights over time: {fund} top average holdings")
    plt.ylabel("Portfolio weight")
    plt.xlabel("Rebalance date")
    plt.legend(loc="upper left", ncol=3, fontsize=8)
    _savefig(FIGURES / "weights_over_time.png")


def _plot_metrics(metrics: pd.DataFrame) -> None:
    m = metrics.sort_values("sharpe", ascending=True)
    plt.figure(figsize=(9.5, 5.4))
    colors_for_bars = [PALETTE.get(f, "#777777") for f in m["family"]]
    plt.barh(m["fund"], m["sharpe"], color=colors_for_bars)
    plt.title("Out-of-sample Sharpe ratio by fund")
    plt.xlabel("Annualised Sharpe")
    _savefig(FIGURES / "sharpe_barplot.png")

    plt.figure(figsize=(8, 5.6))
    for family, group in metrics.groupby("family"):
        plt.scatter(group["annual_volatility"], group["annual_return"], s=95, label=family, color=PALETTE.get(family))
        for _, row in group.iterrows():
            plt.annotate(row["fund"].replace(" ", "\n", 1), (row["annual_volatility"], row["annual_return"]), fontsize=7)
    plt.title("Return-risk map")
    plt.xlabel("Annualised volatility")
    plt.ylabel("Annualised return")
    plt.legend()
    _savefig(FIGURES / "return_risk_scatter.png")


def _plot_sentiment(sector_sent: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.3))
    pivot = sector_sent.pivot(index="date", columns="sector", values="lagged_signal")
    selected = pivot.var().sort_values(ascending=False).head(6).index
    for sector in selected:
        plt.plot(pivot.index, pivot[sector], label=sector, linewidth=1.4)
    plt.title("Lagged sector sentiment index")
    plt.ylabel("Five-day average score, lagged one trading day")
    plt.xlabel("Date")
    plt.legend(ncol=2, fontsize=8)
    _savefig(FIGURES / "sector_sentiment_index.png")


def _plot_fusion(fund_returns: pd.DataFrame) -> None:
    focus = fund_returns[fund_returns["fund"].isin(["Equity Max Sharpe", "Equity Sentiment Tilt"])]
    pivot = focus.pivot(index="date", columns="fund", values="growth")
    plt.figure(figsize=(9.5, 5.2))
    for fund in pivot.columns:
        plt.plot(pivot.index, pivot[fund], linewidth=2, label=fund)
    plt.title("Fusion test: base equity fund vs sentiment tilt")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    _savefig(FIGURES / "fusion_before_after.png")


def _table_rows(metrics: pd.DataFrame) -> list[list[str]]:
    cols = ["fund", "annual_return", "annual_volatility", "sharpe", "max_drawdown"]
    out = [["Fund", "Ann. return", "Ann. vol", "Sharpe", "Max DD"]]
    for _, row in metrics[cols].iterrows():
        out.append(
            [
                row["fund"],
                _currency_pct(row["annual_return"]),
                _currency_pct(row["annual_volatility"]),
                f"{row['sharpe']:.2f}",
                _currency_pct(row["max_drawdown"]),
            ]
        )
    return out


def _write_report(metrics: pd.DataFrame, fusion_table: pd.DataFrame, design: pd.DataFrame) -> None:
    """Create a concise PDF report draft with required exhibits."""
    pdf_path = REPORT / "report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    def h(text: str) -> None:
        story.append(Paragraph(text, styles["Heading2"]))

    def p(text: str) -> None:
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 0.18 * cm))

    story.append(Paragraph("FINS5545 Project Part B: Systematic Multi-Asset Funds with News Sentiment", styles["Title"]))
    p("This report draft documents a reproducible walk-forward experiment over the 2020-2023 course dataset. The investment universe combines 50 US equities and 10 cryptocurrencies, with equity-only and crypto-only funds included for benchmarking. All portfolio weights are estimated using only trailing observations available at each monthly rebalance.")
    h("1. Fund design and backtest controls")
    design_text = "; ".join(
        f"{r.fund}: first live date {r.first_live_date}, lookback {r.lookback_days} days, annualisation {r.periods_per_year}"
        for r in design.itertuples()
    )
    p("The methods are equal weight, minimum variance, maximum Sharpe and inverse-volatility risk parity. Long-only weights are capped at 20 percent to reduce single-name concentration. " + design_text + ".")
    p("Crypto returns are computed on the 365-day crypto calendar first. Combined funds then left-merge those already-computed crypto returns onto the equity trading calendar, matching the course calendar instruction.")

    h("2. Out-of-sample results")
    table = Table(_table_rows(metrics), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7c7c7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))
    best = metrics.sort_values("sharpe", ascending=False).iloc[0]
    p(f"The highest Sharpe in this run is {best['fund']} at {best['sharpe']:.2f}. This should be interpreted as an out-of-sample historical result, not a forecast. Differences across funds largely reflect the trade-off between volatility control and exposure to high-return crypto assets.")
    for image, caption in [
        ("growth_of_one.png", "Figure 1. Growth of $1 across all out-of-sample funds."),
        ("drawdowns.png", "Figure 2. Drawdown paths for the main combined and equity funds."),
        ("weights_over_time.png", "Figure 3. Top average holdings through time for the combined minimum-variance fund."),
    ]:
        story.append(Image(str(FIGURES / image), width=16.5 * cm, height=8.2 * cm))
        p(caption)

    story.append(PageBreak())
    h("3. News sentiment index")
    p("Headlines are de-duplicated by ticker, date and title, then aligned to the same or next equity trading day. A transparent finance lexicon scores the assembled ticker-day text. The sector index is equal-weighted across stocks, smoothed over five trading days, and lagged by one trading day before use in any portfolio rule.")
    story.append(Image(str(FIGURES / "sector_sentiment_index.png"), width=16.5 * cm, height=8.2 * cm))
    p("Figure 4. Lagged sector sentiment index for the six most variable sectors.")

    h("4. Fusion extension")
    p("The fusion test starts from the equity maximum-Sharpe weight schedule and tilts sector weights toward sectors with stronger lagged sentiment. Because the signal is lagged and measured before portfolio returns are realised, the test is look-ahead safe.")
    fusion_rows = [["Fund", "Ann. return", "Ann. vol", "Sharpe", "Max DD"]]
    for _, row in fusion_table.iterrows():
        fusion_rows.append([row["fund"], _currency_pct(row["annual_return"]), _currency_pct(row["annual_volatility"]), f"{row['sharpe']:.2f}", _currency_pct(row["max_drawdown"])])
    ftable = Table(fusion_rows, repeatRows=1)
    ftable.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(ftable)
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(FIGURES / "fusion_before_after.png"), width=16.5 * cm, height=8.2 * cm))
    p("Figure 5. Before-vs-after comparison for the sentiment fusion rule.")

    h("5. App and recommendations")
    p("The Streamlit app reads only precomputed CSV files from results/. It supports fund comparison, fund fact sheets, a simple allocation blend, current holdings and sector sentiment analytics. This design keeps the deployed app lightweight and avoids running NLP or portfolio optimisation on Streamlit Community Cloud.")
    p("Recommendations: first, review the cap and rebalance frequency as sensitivity checks; second, treat the lexicon sentiment as a noisy proxy and compare it with VADER or a transformer model if time permits; third, report both positive and negative fusion results honestly because a weak signal is still informative for allocation governance.")

    doc.build(story)
    md = REPORT / "report_draft.md"
    md.write_text(
        "# FINS5545 Project Part B Report Draft\n\n"
        "This PDF was generated from `scripts/run_part_b.py`. Review the wording, "
        "replace the placeholder zID folder name, and re-export from Word if required by the course.\n\n"
        + "\n".join("- " + " ".join(wrap(line, 110)) for line in [
            "The backtest is monthly, long-only and walk-forward with weights estimated from trailing data only.",
            "The sentiment model uses de-duplicated headlines aligned to the next trading day, smoothed five days and lagged one day.",
            "The deployed app reads precomputed CSV artifacts and does not import NLTK or recompute backtests.",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    _ensure_dirs()
    raw_news_rows = len(etl.data_access.load_news_headlines())
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()
    news = etl.load_clean_headlines()
    print("clean equities:", eq.shape, "crypto:", cr.shape, "headlines:", news.shape)

    pd.DataFrame(
        [
            etl.price_audit(eq, "Equity"),
            etl.price_audit(cr, "Crypto"),
            etl.headline_audit(news, raw_rows=raw_news_rows),
        ]
    ).to_csv(TABLES / "data_audit.csv", index=False)

    eq_ret = features.daily_returns(eq)
    cr_ret = features.daily_returns(cr)
    cr_on_equity_calendar = features.align_crypto_to_equity_calendar(cr_ret, eq_ret.index)
    combined_ret = pd.concat([eq_ret, cr_on_equity_calendar], axis=1).fillna(0.0)

    ticker_sector = eq[["ticker", "sector"]].drop_duplicates()
    headline_panel = features.assemble_headline_panel(news, trading_calendar=eq_ret.index)
    scores = sentiment.score_headlines(headline_panel)
    sector_index = sentiment.sector_sentiment_index(scores, calendar=eq_ret.index)
    sector_index.to_csv(DATA / "sector_sentiment_index.csv", index=False)
    headline_panel.to_csv(DATA / "headline_panel_sample.csv", index=False)

    specs = [
        ("Combined Equal Weight", combined_ret, "equal_weight", "Combined", 252, 252, 0.20),
        ("Combined Min Variance", combined_ret, "min_variance", "Combined", 252, 252, 0.20),
        ("Combined Max Sharpe", combined_ret, "max_sharpe", "Combined", 252, 252, 0.20),
        ("Equity Equal Weight", eq_ret, "equal_weight", "Equity", 252, 252, 0.20),
        ("Equity Min Variance", eq_ret, "min_variance", "Equity", 252, 252, 0.20),
        ("Equity Max Sharpe", eq_ret, "max_sharpe", "Equity", 252, 252, 0.20),
        ("Crypto Equal Weight", cr_ret, "equal_weight", "Crypto", 365, 365, 0.25),
        ("Crypto Risk Parity", cr_ret, "risk_parity", "Crypto", 365, 365, 0.25),
    ]
    results = []
    for fund, rets, method, family, lookback, periods, cap in specs:
        print("backtest:", fund)
        results.append(
            portfolios.oos_backtest(
                rets,
                method=method,
                lookback=lookback,
                rebalance="M",
                periods_per_year=periods,
                fund_name=fund,
                family=family,
                max_weight=cap,
            )
        )

    base_equity = next(r for r in results if r["metrics"]["fund"] == "Equity Max Sharpe")
    tilted_weights = fusion.apply_sentiment(base_equity["weights"], sector_index, ticker_sector)
    tilted_returns = portfolios.returns_from_weights(eq_ret, tilted_weights)
    tilted_metrics = portfolios.performance_metrics(tilted_returns, periods_per_year=252)
    tilted_metrics.update(
        {
            "fund": "Equity Sentiment Tilt",
            "family": "Equity",
            "method": "sentiment_tilt",
            "lookback_days": 252,
            "rebalance": "monthly",
            "periods_per_year": 252,
            "first_live_date": tilted_returns.index.min().date().isoformat(),
        }
    )
    results.append({"returns": tilted_returns, "weights": tilted_weights, "metrics": tilted_metrics})

    fund_returns = portfolios.returns_frame(results)
    weights = pd.concat([r["weights"] for r in results], ignore_index=True)
    metrics = pd.DataFrame([r["metrics"] for r in results]).sort_values(["family", "fund"])
    design = metrics[["fund", "family", "method", "lookback_days", "rebalance", "periods_per_year", "first_live_date"]]

    fund_returns.to_csv(DATA / "fund_returns.csv", index=False)
    weights.to_csv(DATA / "fund_weights.csv", index=False)
    metrics.to_csv(TABLES / "performance_metrics.csv", index=False)
    design.to_csv(TABLES / "backtest_design.csv", index=False)

    latest_holdings = (
        weights.sort_values("rebalance_date")
        .groupby(["fund", "ticker"], as_index=False)
        .tail(1)
        .sort_values(["fund", "weight"], ascending=[True, False])
    )
    latest_holdings.to_csv(TABLES / "current_holdings.csv", index=False)

    fusion_table = metrics[metrics["fund"].isin(["Equity Max Sharpe", "Equity Sentiment Tilt"])]
    fusion_table.to_csv(TABLES / "fusion_before_after.csv", index=False)

    _plot_growth(fund_returns)
    _plot_drawdown(fund_returns)
    _plot_weights(weights)
    _plot_metrics(metrics)
    _plot_sentiment(sector_index)
    _plot_fusion(fund_returns)
    _write_report(metrics, fusion_table, design)

    print("wrote results to", RESULTS)


if __name__ == "__main__":
    main()
