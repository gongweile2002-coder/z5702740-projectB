"""Station 1 ETL helpers for the project datasets.

The raw parquet files are loaded only through ``src.data_access``. These helpers
standardise dates, remove exact duplicate keys, and keep a compact audit trail
that the reproduction script writes to ``results/tables``.
"""
from __future__ import annotations

import pandas as pd

from src import data_access


PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]


def _clean_price_frame(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    clean = df.copy()
    clean["date"] = pd.to_datetime(clean["date"]).dt.tz_localize(None)
    clean = clean.sort_values(PRICE_KEY)
    clean = clean.drop_duplicates(PRICE_KEY, keep="last")
    clean = clean[clean["date"] <= pd.Timestamp("2023-12-31")]
    clean["asset_class"] = asset_class
    if "sector" not in clean.columns:
        clean["sector"] = "Crypto"
    clean = clean[clean["adjClose"].notna() & (clean["adjClose"] > 0)]
    return clean.reset_index(drop=True)


def load_clean_equities() -> pd.DataFrame:
    """Load, de-duplicate, and lightly validate the equity price panel."""
    return _clean_price_frame(data_access.load_equity_prices(), "Equity")


def load_clean_crypto() -> pd.DataFrame:
    """Load, de-duplicate, and lightly validate the crypto price panel."""
    return _clean_price_frame(data_access.load_crypto_prices(), "Crypto")


def load_clean_headlines() -> pd.DataFrame:
    """Load headline data and remove exact duplicate ticker-date-title rows."""
    news = data_access.load_news_headlines().copy()
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news = news[news["title"].ne("")]
    news = news.drop_duplicates(NEWS_KEY, keep="first")
    return news.sort_values(["date", "ticker", "title"]).reset_index(drop=True)


def price_audit(df: pd.DataFrame, asset_class: str) -> dict:
    """Return a small integrity summary for one cleaned price panel."""
    dates_per_ticker = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    returns = (
        df.sort_values(PRICE_KEY)
        .groupby("ticker")["adjClose"]
        .pct_change()
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )
    return {
        "dataset": f"{asset_class.lower()}_prices",
        "asset_class": asset_class,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "start_date": df["date"].min().date().isoformat(),
        "end_date": df["date"].max().date().isoformat(),
        "duplicate_ticker_dates_after_clean": int(df.duplicated(PRICE_KEY).sum()),
        "min_observations_per_ticker": int(dates_per_ticker["count"].min()),
        "max_abs_daily_return": float(returns.abs().max()) if not returns.empty else 0.0,
    }


def headline_audit(news: pd.DataFrame, raw_rows: int | None = None) -> dict:
    """Return a compact integrity summary for cleaned headlines."""
    return {
        "dataset": "news_headlines",
        "asset_class": "Equity news",
        "rows": int(len(news)),
        "tickers": int(news["ticker"].nunique()),
        "start_date": news["date"].min().date().isoformat(),
        "end_date": news["date"].max().date().isoformat(),
        "duplicate_rows_removed": int((raw_rows or len(news)) - len(news)),
        "duplicate_ticker_date_title_after_clean": int(news.duplicated(NEWS_KEY).sum()),
        "min_observations_per_ticker": int(news.groupby("ticker").size().min()),
        "max_abs_daily_return": "",
    }
