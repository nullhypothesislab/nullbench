"""Survivorship bias on synthetic data (no third-party data needed).

Each "world" has 600 random-walk stocks (one shared market factor + stock-specific noise) with NO real momentum and
the same ~7%/yr expected return. Stocks that fall below 25% of their start are delisted after a final -40% day.
We run the same strategies on the point-in-time universe and on "today's survivors" (the usual mistake),
across 20 independent worlds, and check one world against a shuffled-signal control.

    python examples/survivorship_demo.py
"""
import numpy as np
import pandas as pd

import nullbench as nb


def world(seed: int, n: int = 600, years: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=252 * years)
    mvol = 0.16 / np.sqrt(252)
    market = mvol * rng.standard_normal((len(dates), 1))
    vol = rng.uniform(0.15, 0.60, n) / np.sqrt(252)
    logret = 0.07 / 252 - 0.5 * (vol ** 2 + mvol ** 2) + market + vol * rng.standard_normal((len(dates), n))
    prices = pd.DataFrame(np.exp(np.cumsum(logret, axis=0)), index=dates, columns=[f"S{i:03d}" for i in range(n)])
    for j, c in enumerate(prices.columns):
        hit = np.flatnonzero(prices[c].to_numpy() < 0.25)
        if hit.size:
            d = hit[0]
            prices.iloc[min(d + 1, len(dates) - 1), j] = prices[c].iloc[d] * 0.6
            prices.iloc[d + 2:, j] = np.nan
    return prices


def strategies(prices: pd.DataFrame):
    returns = prices.shift(-1) / prices - 1
    reb, start = nb.first_of_month(prices.index), prices.index[252]
    lagged = prices.shift(1)                                         # only yesterday's close is known
    mom = lagged.shift(21) / lagged.shift(252) - 1                   # 12-1 month momentum
    universes = {"point_in_time": nb.tradable(returns),
                 "survivors_only": nb.tradable(returns, nb.survivors_only(prices))}
    return returns, reb, start, mom, universes


rows = []
for seed in range(20):
    returns, reb, start, mom, universes = strategies(world(seed))
    row = {}
    for name, elig in universes.items():
        row[f"equal_weight_{name}"] = nb.summary(nb.run(nb.on_dates(nb.equal_weight(elig), reb), returns, 0.001, start))["cagr_%"]
        w = nb.on_dates(nb.equal_weight(nb.top_n(mom, 30, elig)), reb)
        row[f"momentum_{name}"] = nb.summary(nb.run(w, returns, 0.001, start))["cagr_%"]
    rows.append(row)
D = pd.DataFrame(rows)
D["equal_weight_bias"] = D.equal_weight_survivors_only - D.equal_weight_point_in_time
D["momentum_bias"] = D.momentum_survivors_only - D.momentum_point_in_time
print("CAGR % over 20 synthetic worlds (every stock has the same expected return, no real momentum):")
print(D.describe().loc[["mean", "min", "max"]].round(1).T.to_string(), "\n")

returns, reb, start, mom, universes = strategies(world(0))
print("World 0, momentum top-30 vs its shuffled-signal control (100 shuffles):")
for name, elig in universes.items():
    res = nb.run(nb.on_dates(nb.equal_weight(nb.top_n(mom, 30, elig)), reb), returns, 0.001, start)
    ctrl = nb.shuffled_signal(mom, elig, 30, reb, returns, 0.001, start, sims=100)
    print(f"  {name:15s}", nb.verdict(res.equity.iloc[-1], ctrl))
