"""Summary statistics for an engine result."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(result: pd.DataFrame, initial: float = 1.0, periods_per_year: int = 252) -> dict:
    eq = result["equity"]
    r = eq.pct_change().fillna(eq.iloc[0] / initial - 1)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / initial) ** (1 / years) - 1 if years > 0 else np.nan
    dd = 1 - eq / eq.cummax()
    by_year = eq.groupby(eq.index.year).last()
    yearly = by_year / by_year.shift(1).fillna(initial) - 1
    return {
        "final": float(eq.iloc[-1]),
        "cagr_%": round(100 * cagr, 1),
        "max_drawdown_%": round(100 * dd.max(), 1),
        "sharpe": round(r.mean() / r.std() * np.sqrt(periods_per_year), 2) if r.std() > 0 else np.nan,
        "worst_year_%": round(100 * yearly.min(), 1),
        "turnover_per_year": round(result["turnover"].sum() / years, 1) if years > 0 else np.nan,
        "avg_exposure_%": round(100 * result["exposure"].mean()),
    }
