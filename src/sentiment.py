"""Station 3 headline sentiment model and sector index.

The deployed app reads precomputed CSVs, so the scoring model lives here rather
than in ``streamlit_app.py``. I use a small transparent finance lexicon instead
of VADER so the reproduction step is deterministic and does not need an NLTK
data download.
"""
from __future__ import annotations

import math
import re

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

POSITIVE = {
    "advance", "advanced", "advances", "beat", "beats", "beating", "benefit",
    "boost", "boosts", "bullish", "buyback", "climb", "climbs", "deal",
    "dividend", "expand", "expands", "gain", "gains", "growth", "higher",
    "improve", "improves", "improved", "jump", "jumps", "leader", "outperform",
    "positive", "profit", "profits", "rally", "rallies", "record", "recover",
    "recovers", "recovery", "rise", "rises", "rose", "strong", "surge", "surges",
    "top", "tops", "upgrade", "upgrades", "upside", "win", "wins",
}

NEGATIVE = {
    "bearish", "cut", "cuts", "decline", "declines", "default", "downgrade",
    "downgrades", "drop", "drops", "fall", "falls", "fell", "fraud", "hurt",
    "inflation", "lawsuit", "layoff", "layoffs", "loss", "losses", "miss",
    "misses", "negative", "plunge", "plunges", "pressure", "probe", "recession",
    "risk", "risks", "slump", "slumps", "slowdown", "struggle", "struggles",
    "tumble", "tumbles", "underperform", "weak", "weaker", "warning",
}

NEGATORS = {"no", "not", "never", "without", "less", "hardly"}
INTENSIFIERS = {"very", "sharp", "sharply", "strong", "strongly", "major"}


def score_text(text: str) -> float:
    """Score one headline or aggregated text block in [-1, 1]."""
    tokens = [t.lower().strip("'") for t in TOKEN_RE.findall(str(text))]
    if not tokens:
        return 0.0
    score = 0.0
    hits = 0
    for i, token in enumerate(tokens):
        base = 0
        if token in POSITIVE:
            base = 1
        elif token in NEGATIVE:
            base = -1
        if base == 0:
            continue
        window = tokens[max(0, i - 3) : i]
        if any(w in NEGATORS for w in window):
            base *= -1
        if any(w in INTENSIFIERS for w in window):
            base *= 1.4
        score += base
        hits += 1
    if hits == 0:
        return 0.0
    return float(max(-1.0, min(1.0, score / math.sqrt(hits + 2))))


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score an assembled ticker-day text panel."""
    scores = panel.copy()
    text_col = "text" if "text" in scores.columns else "title"
    date_col = "trading_date" if "trading_date" in scores.columns else "date"
    scores["date"] = pd.to_datetime(scores[date_col])
    scores["sentiment_score"] = scores[text_col].map(score_text)
    if "article_count" not in scores.columns:
        scores["article_count"] = 1
    keep = ["date", "ticker", "sector", "article_count", "sentiment_score"]
    return scores[keep].sort_values(["date", "sector", "ticker"]).reset_index(drop=True)


def sector_sentiment_index(
    scores: pd.DataFrame,
    calendar: pd.DatetimeIndex | None = None,
    smoothing_days: int = 5,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build a lagged sector sentiment index equal-weighted across tickers."""
    daily_ticker = (
        scores.groupby(["date", "sector", "ticker"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), article_count=("article_count", "sum"))
    )
    sector = (
        daily_ticker.groupby(["date", "sector"], as_index=False)
        .agg(raw_sentiment=("sentiment_score", "mean"), article_count=("article_count", "sum"))
    )
    pivot = sector.pivot(index="date", columns="sector", values="raw_sentiment")
    counts = sector.pivot(index="date", columns="sector", values="article_count")
    if calendar is not None:
        calendar = pd.DatetimeIndex(pd.to_datetime(calendar))
        pivot = pivot.reindex(calendar)
        counts = counts.reindex(calendar)
    smoothed = pivot.fillna(0.0).rolling(smoothing_days, min_periods=1).mean()
    lagged = smoothed.shift(lag_days).fillna(0.0)
    pivot.index.name = "date"
    smoothed.index.name = "date"
    lagged.index.name = "date"
    counts.index.name = "date"
    raw_long = pivot.reset_index().melt(id_vars="date", var_name="sector", value_name="raw_sentiment")
    smooth_long = smoothed.reset_index().melt(id_vars="date", var_name="sector", value_name="sentiment_index")
    lagged_long = lagged.reset_index().melt(id_vars="date", var_name="sector", value_name="lagged_signal")
    count_long = counts.fillna(0).astype(int).reset_index().melt(
        id_vars="date", var_name="sector", value_name="article_count"
    )
    out = (
        raw_long.merge(smooth_long, on=["date", "sector"], how="left")
        .merge(lagged_long, on=["date", "sector"], how="left")
        .merge(count_long, on=["date", "sector"], how="left")
    )
    return out.sort_values(["date", "sector"]).reset_index(drop=True)
