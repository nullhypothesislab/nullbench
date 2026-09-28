"""Target-weight backtest engine.

Conventions
-----------
- ``returns.loc[t]`` is the return earned by a position held from the decision at ``t`` until the next row.
  Build it as ``prices.shift(-1) / prices - 1`` so a decision at ``t`` never sees ``t+1``.
- ``weights`` holds target weights only on rebalance dates; all-NaN rows mean "hold" (weights drift with returns).
- Long-only, no leverage: weights >= 0 and row sum <= 1; the remainder is cash earning 0.
- Costs: ``cost`` is the round-trip cost as a fraction (0.001 = 10 bp). Each side pays half of it on actual turnover,
  i.e. ``equity -= equity * 0.5 * sum(|new - drifted| * cost)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run(weights: pd.DataFrame, returns: pd.DataFrame, cost: float | pd.DataFrame = 0.001,
        start: str | pd.Timestamp | None = None, initial: float = 1.0) -> pd.DataFrame:
    """Simulate ``weights`` on ``returns``. Returns a DataFrame with columns equity, turnover, exposure."""
    dates = returns.index if start is None else returns.index[returns.index >= pd.Timestamp(start)]
    W = weights.reindex(index=dates, columns=returns.columns)
    r = returns.loc[dates].fillna(0.0).to_numpy()
    if isinstance(cost, pd.DataFrame):
        k = cost.reindex(index=dates, columns=returns.columns).ffill().fillna(0.0).to_numpy()
    else:
        k = np.full(r.shape, float(cost))
    target = W.to_numpy()
    w = np.zeros(r.shape[1])
    equity = float(initial)
    out = np.empty((len(dates), 3))
    for i in range(len(dates)):
        turnover = 0.0
        if not np.all(np.isnan(target[i])):
            new = np.nan_to_num(target[i])
            if (new < -1e-12).any() or new.sum() > 1 + 1e-9:
                raise ValueError(f"invalid weights on {dates[i]}: need >= 0 and sum <= 1")
            trade = np.abs(new - w)
            turnover = trade.sum()
            equity -= equity * 0.5 * (trade * k[i]).sum()
            w = new
        pr = float((w * r[i]).sum())
        equity *= 1 + pr
        out[i] = (equity, turnover, w.sum())
        w = w * (1 + r[i]) / (1 + pr) if 1 + pr > 0 else w * 0.0
    return pd.DataFrame(out, index=dates, columns=["equity", "turnover", "exposure"])


def equal_weight(mask: pd.DataFrame, max_weight: float | None = None) -> pd.DataFrame:
    """Equal weights across True cells of each row (rows with no True -> all cash)."""
    n = mask.sum(axis=1).replace(0, np.nan)
    w = mask.astype(float).div(n, axis=0)
    if max_weight is not None:
        w = w.clip(upper=max_weight)
    return w.fillna(0.0)


def on_dates(weights: pd.DataFrame, dates) -> pd.DataFrame:
    """Keep target rows only on ``dates`` (e.g. rebalance days); other rows become NaN = hold."""
    keep = weights.index.isin(pd.DatetimeIndex(dates))
    return weights.where(np.broadcast_to(keep[:, None], weights.shape))


def first_of_month(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).first().to_numpy())


def top_n(signal: pd.DataFrame, n: int, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Boolean mask of the ``n`` highest signal values per row (optionally only among ``eligible``)."""
    s = signal if eligible is None else signal.where(eligible)
    return s.rank(axis=1, ascending=False, method="first") <= n
