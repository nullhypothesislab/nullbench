# nullbench

Small, honest backtesting for long-only portfolio strategies in Python. It covers three things most retail backtests skip:

1. **Point-in-time universes.** Trade only what was actually in the index on that day, including names that were later delisted.
2. **Realistic costs.** Costs are charged on real turnover, and weights drift between rebalances.
3. **Null-hypothesis controls.** Every strategy is compared with itself run on *no information*: random picks, or its own signal shuffled across names.

~160 lines, depends only on `numpy` and `pandas`.

```python
import nullbench as nb

returns = prices.shift(-1) / prices - 1            # decision at t earns t -> t+1
eligible = nb.tradable(returns, nb.membership_mask(prices.index, history, prices.columns))
signal = prices.shift(1).shift(21) / prices.shift(1).shift(252) - 1   # 12-1 momentum on yesterday's close
weights = nb.on_dates(nb.equal_weight(nb.top_n(signal, 30, eligible)), nb.first_of_month(prices.index))

result = nb.run(weights, returns, cost=0.001)      # 10 bp round trip
print(nb.summary(result))

control = nb.shuffled_signal(signal, eligible, 30, nb.first_of_month(prices.index), returns, 0.001)
print(nb.verdict(result.equity.iloc[-1], control))  # p-value vs. the same strategy with the information destroyed
```

`history` is a membership table with columns `date` and `tickers` (a comma-separated list valid from that date on), e.g. the public [fja05680/sp500](https://github.com/fja05680/sp500) file. nullbench ships **no market data**; bring your own.

## Demo: how big is survivorship bias?

`python examples/survivorship_demo.py` builds 20 synthetic markets. Each has 600 stocks with the *same* expected return and no real momentum, and the ones that collapse get delisted. It then runs the same strategies on the point-in-time universe and on "today's survivors":

| CAGR, mean of 20 worlds | point-in-time | survivors only | bias |
|---|---|---|---|
| Equal weight | 3.4% | 10.7% | **+7.3 pp/yr** (every world: +4.1 … +12.3) |
| Momentum 12-1, top 30 | 4.4% | 8.4% | **+4.0 pp/yr** (every world: +1.8 … +7.7) |

The demo also shows why controls matter, and what they *don't* tell you. In world 0, momentum beats its shuffled-signal control (p ≈ 0.01) even though no stock has real momentum. It isn't a bug: in that world a falling stock really does carry extra risk (a −40% delisting day), so avoiding falling stocks is genuine information. A control tells you *whether* a signal carries information, not *which* story explains it.

## Conventions
- `returns.loc[t]` is earned by a position chosen at `t`. Build signals from data known **before** `t` (e.g. `prices.shift(1)`).
- In `weights`, all-NaN rows mean "hold"; weights must be ≥ 0 and sum to ≤ 1. The rest is cash earning 0.
- `cost` is the round-trip cost; half of it is paid per side, on actual turnover.

## Tests
`python -m pytest tests`

## About
Made by **Null Hypothesis Lab**, a one-person-and-an-AI research project: an AI assistant (Claude) writes the code and articles, and a human owner supervises. The full research toolkit (a survivorship-free S&P 500 pipeline with delisted names, 50+ strategy implementations, walk-forward ML with shuffled-label controls, HTML reports) is a separate paid product.

MIT License.
