import numpy as np
import pandas as pd
import pytest

import nullbench as nb


def _returns(vals, cols=("A", "B")):
    return pd.DataFrame(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)), columns=list(cols))


def test_buy_and_hold_compounds_without_costs():
    R = _returns([[0.10, 0.0], [0.10, 0.0], [0.0, 0.0]])
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    W.iloc[0] = [1.0, 0.0]
    eq = nb.run(W, R, cost=0.0)["equity"]
    assert eq.iloc[-1] == pytest.approx(1.21)


def test_cost_charged_half_round_trip_per_side():
    R = _returns([[0.0, 0.0], [0.0, 0.0]])
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    W.iloc[0] = [1.0, 0.0]                       # buy A: turnover 1
    W.iloc[1] = [0.0, 1.0]                       # sell A, buy B: turnover 2
    eq = nb.run(W, R, cost=0.01)["equity"]
    assert eq.iloc[0] == pytest.approx(1 - 0.005)
    assert eq.iloc[1] == pytest.approx((1 - 0.005) * (1 - 0.01))


def test_weights_drift_between_rebalances():
    R = _returns([[1.0, 0.0], [0.0, 0.0]])       # A doubles on day 0
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    W.iloc[0] = [0.5, 0.5]
    res = nb.run(W, R, cost=0.0)
    assert res["equity"].iloc[0] == pytest.approx(1.5)
    W.iloc[1] = [0.5, 0.5]                       # drifted to 2/3 : 1/3 -> rebalancing trades 1/3
    assert nb.run(W, R, cost=0.0)["turnover"].iloc[1] == pytest.approx(1 / 3)


def test_invalid_weights_raise():
    R = _returns([[0.0, 0.0]])
    W = pd.DataFrame([[0.8, 0.8]], index=R.index, columns=R.columns)
    with pytest.raises(ValueError):
        nb.run(W, R)


def test_membership_mask_point_in_time():
    dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"])
    hist = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-12-31"]), "tickers": ["A,B", "B,C"]})
    m = nb.membership_mask(pd.DatetimeIndex(dates), hist, ["A", "B", "C"])
    assert m.loc["2020-06-01"].tolist() == [True, True, False]
    assert m.loc["2021-01-01"].tolist() == [False, True, True]


def test_shuffled_control_has_no_edge_on_noise():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=300)
    R = pd.DataFrame(rng.normal(0, 0.01, (300, 40)), index=idx)
    sig = pd.DataFrame(rng.normal(size=(300, 40)), index=idx)
    elig = R.notna()
    ctrl = nb.shuffled_signal(sig, elig, 10, nb.first_of_month(idx), R, 0.0, sims=30)
    real = nb.run(nb.on_dates(nb.equal_weight(nb.top_n(sig, 10, elig)), nb.first_of_month(idx)), R, 0.0)
    assert 0.0 < nb.verdict(real.equity.iloc[-1], ctrl)["p_value"] <= 1.0
