"""Station 3 portfolio construction and walk-forward backtests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_returns(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all").fillna(0.0)
    return out.astype(float)


def _normalise_weights(weights: pd.Series, max_weight: float = 0.20) -> pd.Series:
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)
    if weights.sum() <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights = weights / weights.sum()

    if max_weight and len(weights) * max_weight >= 1:
        for _ in range(8):
            over = weights > max_weight
            if not over.any():
                break
            excess = (weights[over] - max_weight).sum()
            weights[over] = max_weight
            under = ~over
            if weights[under].sum() <= 0:
                break
            weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def estimate_weights(
    history: pd.DataFrame,
    method: str = "min_variance",
    max_weight: float = 0.20,
) -> pd.Series:
    """Estimate long-only weights from past returns only."""
    hist = _safe_returns(history)
    assets = hist.columns
    if len(assets) == 0:
        return pd.Series(dtype=float)
    if method == "equal_weight":
        return pd.Series(1.0 / len(assets), index=assets)

    vol = hist.std().replace(0, np.nan)
    if method in {"inverse_vol", "risk_parity"}:
        inv_vol = 1 / vol
        return _normalise_weights(inv_vol.fillna(inv_vol.median()), max_weight=max_weight)

    cov = hist.cov().values
    diag = np.diag(np.diag(cov))
    cov = 0.80 * cov + 0.20 * diag
    cov = cov + np.eye(len(assets)) * 1e-6

    if method == "min_variance":
        target = np.ones(len(assets))
    elif method == "max_sharpe":
        mu = hist.mean().clip(lower=0)
        target = mu.values
        if np.allclose(target, 0):
            target = (1 / vol.fillna(vol.median())).values
    else:
        raise ValueError(f"unknown portfolio method: {method}")

    try:
        raw = np.linalg.solve(cov, target)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov) @ target
    return _normalise_weights(pd.Series(raw, index=assets), max_weight=max_weight)


def rebalance_dates(index: pd.DatetimeIndex, lookback: int, frequency: str = "M") -> list[pd.Timestamp]:
    """Return period-end rebalance dates after the lookback window."""
    dates = pd.DatetimeIndex(index).sort_values()
    eligible = dates[lookback - 1 :]
    if frequency.upper().startswith("M"):
        return list(pd.Series(eligible, index=eligible).groupby(eligible.to_period("M")).max())
    if frequency.upper().startswith("Q"):
        return list(pd.Series(eligible, index=eligible).groupby(eligible.to_period("Q")).max())
    raise ValueError("frequency must be monthly or quarterly")


def returns_from_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    return_column: str = "return",
) -> pd.Series:
    """Apply a rebalance weight schedule out-of-sample."""
    panel = _safe_returns(returns)
    dates = panel.index
    out = pd.Series(index=dates, dtype=float, name=return_column)

    schedule = weights.copy()
    schedule["rebalance_date"] = pd.to_datetime(schedule["rebalance_date"])
    rebals = sorted(schedule["rebalance_date"].unique())
    for i, rebalance_date in enumerate(rebals):
        if rebalance_date not in dates:
            continue
        start_pos = dates.get_loc(rebalance_date) + 1
        if start_pos >= len(dates):
            continue
        if i + 1 < len(rebals) and rebals[i + 1] in dates:
            end_pos = dates.get_loc(rebals[i + 1]) + 1
        else:
            end_pos = len(dates)
        span = dates[start_pos:end_pos]
        w = schedule.loc[schedule["rebalance_date"].eq(rebalance_date), ["ticker", "weight"]]
        w = w.set_index("ticker")["weight"].reindex(panel.columns).fillna(0.0)
        out.loc[span] = panel.loc[span].mul(w, axis=1).sum(axis=1)
    return out.dropna()


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, volatility, Sharpe, max drawdown, and hit rate."""
    r = daily_returns.dropna().astype(float)
    if r.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "hit_rate": 0.0,
            "observations": 0,
        }
    growth = (1 + r).cumprod()
    ann_return = growth.iloc[-1] ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * math.sqrt(periods_per_year)
    drawdown = growth / growth.cummax() - 1
    return {
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(ann_return / ann_vol) if ann_vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "cumulative_return": float(growth.iloc[-1] - 1),
        "hit_rate": float((r > 0).mean()),
        "observations": int(len(r)),
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    lookback: int = 252,
    rebalance: str = "M",
    periods_per_year: int = 252,
    fund_name: str | None = None,
    family: str = "Combined",
    max_weight: float = 0.20,
) -> dict:
    """Walk-forward out-of-sample backtest with no look-ahead."""
    panel = _safe_returns(returns)
    rdates = rebalance_dates(panel.index, lookback=lookback, frequency=rebalance)
    weight_rows = []
    for date in rdates:
        hist = panel.loc[:date].tail(lookback)
        w = estimate_weights(hist, method=method, max_weight=max_weight)
        for ticker, weight in w.items():
            weight_rows.append(
                {
                    "rebalance_date": date,
                    "ticker": ticker,
                    "weight": float(weight),
                    "method": method,
                    "fund": fund_name or method,
                    "family": family,
                }
            )

    weights = pd.DataFrame(weight_rows)
    daily = returns_from_weights(panel, weights)
    daily.name = "return"
    metrics = performance_metrics(daily, periods_per_year=periods_per_year)
    metrics.update(
        {
            "fund": fund_name or method,
            "family": family,
            "method": method,
            "lookback_days": lookback,
            "rebalance": "monthly" if rebalance.upper().startswith("M") else "quarterly",
            "periods_per_year": periods_per_year,
            "first_live_date": daily.index.min().date().isoformat() if not daily.empty else "",
        }
    )
    return {"returns": daily, "weights": weights, "metrics": metrics}


def returns_frame(fund_results: list[dict]) -> pd.DataFrame:
    """Convert backtest dictionaries to a long app-ready return table."""
    rows = []
    for result in fund_results:
        meta = result["metrics"]
        r = result["returns"].dropna()
        growth = (1 + r).cumprod()
        dd = growth / growth.cummax() - 1
        for date, value in r.items():
            rows.append(
                {
                    "date": date,
                    "fund": meta["fund"],
                    "family": meta["family"],
                    "method": meta["method"],
                    "return": float(value),
                    "growth": float(growth.loc[date]),
                    "drawdown": float(dd.loc[date]),
                }
            )
    return pd.DataFrame(rows).sort_values(["fund", "date"]).reset_index(drop=True)
