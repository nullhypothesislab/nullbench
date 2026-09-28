"""Null-hypothesis controls: what would the same strategy shape earn with no information?

A strategy "works" only if it beats its own controls, not just a benchmark:
- ``random_selection``: same universe, same number of names, same rebalance dates, random picks.
- ``shuffled_signal``: the real signal values, randomly re-assigned across names within each date
  (keeps the signal's distribution and the portfolio's turnover profile, destroys the information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import equal_weight, on_dates, run, top_n


def shuffle_rows(signal: pd.DataFrame, eligible: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute signal values among eligible names on each row."""
    x = signal.where(eligible).to_numpy(copy=True)
    for i in range(x.shape[0]):
        ok = ~np.isnan(x[i])
        x[i, ok] = rng.permutation(x[i, ok])
    return pd.DataFrame(x, index=signal.index, columns=signal.columns)


def random_selection(eligible: pd.DataFrame, n: int, rebalance, returns: pd.DataFrame, cost=0.001,
                     start=None, sims: int = 200, seed: int = 0) -> pd.Series:
    """Final equity of ``sims`` random top-``n`` portfolios (equal weight, same schedule)."""
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(sims):
        noise = pd.DataFrame(rng.random(eligible.shape), index=eligible.index, columns=eligible.columns)
        w = on_dates(equal_weight(top_n(noise, n, eligible)), rebalance)
        finals.append(run(w, returns, cost, start)["equity"].iloc[-1])
    return pd.Series(finals, name="final_equity")


def shuffled_signal(signal: pd.DataFrame, eligible: pd.DataFrame, n: int, rebalance, returns: pd.DataFrame,
                    cost=0.001, start=None, sims: int = 200, seed: int = 0) -> pd.Series:
    """Final equity of ``sims`` strategies built from row-shuffled copies of ``signal``."""
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(sims):
        s = shuffle_rows(signal, eligible, rng)
        w = on_dates(equal_weight(top_n(s, n, eligible)), rebalance)
        finals.append(run(w, returns, cost, start)["equity"].iloc[-1])
    return pd.Series(finals, name="final_equity")


def verdict(real_final: float, control_finals: pd.Series) -> dict:
    """Where the real result sits in the control distribution (one-sided empirical p-value)."""
    c = control_finals.to_numpy()
    p = (1 + (c >= real_final).sum()) / (1 + len(c))
    return {"real": round(float(real_final), 3), "control_median": round(float(np.median(c)), 3),
            "control_p95": round(float(np.quantile(c, 0.95)), 3), "p_value": round(float(p), 3),
            "beats_control_95": bool(real_final > np.quantile(c, 0.95))}
