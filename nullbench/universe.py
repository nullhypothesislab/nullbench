"""Point-in-time universes: who was actually investable on each date."""
from __future__ import annotations

import numpy as np
import pandas as pd


def membership_mask(dates: pd.DatetimeIndex, history: pd.DataFrame, columns) -> pd.DataFrame:
    """Boolean (date x ticker) mask from a membership history.

    ``history`` has one row per change date: columns ``date`` and ``tickers`` (comma-separated list valid from that
    date on). This is the format of the public ``fja05680/sp500`` history file, but any index works.
    """
    h = history.sort_values("date")
    idx = np.searchsorted(pd.to_datetime(h["date"]).to_numpy(), dates.to_numpy(), side="right") - 1
    sets = [set(t.split(",")) for t in h["tickers"]]
    cols = pd.Index(columns)
    out = np.zeros((len(dates), len(cols)), dtype=bool)
    for i, j in enumerate(idx):
        if j >= 0:
            out[i] = cols.isin(sets[j])
    return pd.DataFrame(out, index=dates, columns=cols)


def tradable(returns: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """A name is eligible on ``t`` if it has a return for ``t`` (priced today and next row) and, optionally, is a member."""
    ok = returns.notna()
    return ok if members is None else ok & members.reindex_like(ok).fillna(False).astype(bool)


def survivors_only(prices: pd.DataFrame) -> pd.DataFrame:
    """The biased universe most backtests silently use: names that still trade on the last date, over all history.

    Provided so you can *measure* survivorship bias, not so you can use it.
    """
    alive = prices.iloc[-1].notna()
    return pd.DataFrame(np.broadcast_to(alive.to_numpy(), prices.shape), index=prices.index, columns=prices.columns)
