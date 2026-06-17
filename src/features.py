"""Return features and headline text assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    wide: bool = True,
) -> pd.DataFrame:
    """Compute simple daily returns within each ticker.

    Returns are computed before any cross-asset calendar alignment, which keeps
    the crypto return calculation on its native 365-day calendar.
    """
    df = prices[["date", "ticker", price_col]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker", group_keys=False)[price_col].pct_change()
    df["return"] = df["return"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["return"])
    if not wide:
        return df[["date", "ticker", "return"]].reset_index(drop=True)
    return (
        df.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .astype(float)
    )


def align_crypto_to_equity_calendar(
    crypto_returns: pd.DataFrame,
    equity_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Left-merge precomputed crypto returns onto the equity trading calendar."""
    calendar = pd.DatetimeIndex(pd.to_datetime(equity_calendar)).tz_localize(None)
    return crypto_returns.reindex(calendar)


def align_dates_to_trading_calendar(
    dates: pd.Series,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Map each timestamp to the same or next available equity trading day."""
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values()
    raw = pd.to_datetime(dates).dt.tz_localize(None)
    positions = np.searchsorted(calendar.values, raw.values, side="left")
    aligned = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    aligned.loc[valid] = calendar.values[positions[valid]]
    return aligned


def assemble_headline_panel(
    headlines: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Assemble headlines into a daily ticker-sector text panel.

    If a trading calendar is supplied, weekend/non-trading headlines are mapped
    to the next available equity trading day. This function does not score
    sentiment; it only prepares the text panel used by the Part B model.
    """
    news = headlines.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news = news[news["title"].ne("")]
    if trading_calendar is not None:
        news["trading_date"] = align_dates_to_trading_calendar(news["date"], trading_calendar)
        news = news.dropna(subset=["trading_date"])
    else:
        news["trading_date"] = news["date"]

    grouped = (
        news.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            article_count=("title", "size"),
            text=("title", lambda x: " | ".join(x.astype(str))),
        )
        .sort_values(["trading_date", "sector", "ticker"])
    )
    grouped["word_count"] = grouped["text"].str.split().str.len()
    return grouped.reset_index(drop=True)


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series from a daily return stream."""
    growth = (1 + returns.fillna(0)).cumprod()
    return growth / growth.cummax() - 1
