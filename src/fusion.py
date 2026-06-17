"""Station 3 extension: fuse lagged sector sentiment into equity weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    ticker_sector: pd.DataFrame,
    strength: float = 0.35,
    max_weight: float = 0.20,
    fund_name: str = "Equity Sentiment Tilt",
) -> pd.DataFrame:
    """Tilt equity weights toward sectors with higher lagged sentiment.

    The signal is already lagged by ``sector_sentiment_index``. At each rebalance
    date, the latest available sector value on or before that date is merged onto
    the existing weight schedule, transformed into a bounded multiplier, and
    renormalised to a long-only portfolio.
    """
    sched = weights.copy()
    sched["rebalance_date"] = pd.to_datetime(sched["rebalance_date"])
    sector_map = ticker_sector[["ticker", "sector"]].drop_duplicates()
    sched = sched.merge(sector_map, on="ticker", how="left", suffixes=("", "_map"))
    sched["sector"] = sched["sector"].fillna(sched.pop("sector_map") if "sector_map" in sched else "Crypto")

    sent = sentiment.copy()
    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.sort_values(["sector", "date"])

    tilted_rows = []
    for date, group in sched.groupby("rebalance_date", sort=True):
        sector_signal = (
            sent.loc[sent["date"].le(date)]
            .sort_values("date")
            .groupby("sector")
            .tail(1)
            .set_index("sector")["lagged_signal"]
        )
        g = group.copy()
        signal = g["sector"].map(sector_signal).fillna(0.0)
        if signal.std(ddof=0) > 0:
            signal = (signal - signal.mean()) / signal.std(ddof=0)
        multiplier = (1 + strength * signal.clip(-2, 2)).clip(0.35, 1.75)
        raw = g["weight"].astype(float) * multiplier
        if raw.sum() <= 0:
            raw = g["weight"].astype(float)
        g["weight"] = raw / raw.sum()
        if max_weight and len(g) * max_weight >= 1:
            for _ in range(8):
                over = g["weight"] > max_weight
                if not over.any():
                    break
                excess = (g.loc[over, "weight"] - max_weight).sum()
                g.loc[over, "weight"] = max_weight
                under = ~over
                if g.loc[under, "weight"].sum() <= 0:
                    break
                g.loc[under, "weight"] += excess * g.loc[under, "weight"] / g.loc[under, "weight"].sum()
        g["weight"] = g["weight"] / g["weight"].sum()
        g["sentiment_signal"] = signal
        g["fund"] = fund_name
        g["method"] = "sentiment_tilt"
        g["family"] = "Equity"
        tilted_rows.append(g)

    out = pd.concat(tilted_rows, ignore_index=True)
    return out.replace([np.inf, -np.inf], 0.0)
