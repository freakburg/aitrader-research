import numpy as np
import pandas as pd

from aitrader.minervini import Params, build_signals, simulate


def _panel(n=500, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0008, 0.015, (n, k)), axis=0), index=idx,
                         columns=[f"S{i}" for i in range(k)])
    vol = pd.DataFrame(rng.integers(1_000, 5_000, (n, k)).astype(float), index=idx, columns=close.columns)
    spy = close.mean(axis=1)
    return close, vol, spy


def test_future_edit_does_not_change_past_equity():
    close, vol, spy = _panel()
    cash = np.zeros(len(close))
    for p in (Params("a"), Params("b", market_filter=True, breakout=False)):
        base, _ = simulate(close, build_signals(close, vol, spy, p), p, cash)
        c2, v2, s2 = close.copy(), vol.copy(), spy.copy()
        c2.iloc[400:] *= 1.5
        v2.iloc[400:] *= 3
        s2.iloc[400:] *= 0.5
        alt, _ = simulate(c2, build_signals(c2, v2, s2, p), p, cash)
        # decisions at bar 399 fill at 400, so bars <= 399 cannot depend on anything after bar 399
        assert np.allclose(base.iloc[:400], alt.iloc[:400])


def test_entry_fills_one_bar_after_signal_and_costs_are_charged():
    close, vol, spy = _panel(300, 3)
    sig = {"entry": np.zeros(close.shape, bool), "rs": np.ones(close.shape),
           "below50": np.zeros(close.shape, bool), "mkt_ok": np.ones(len(close), bool)}
    sig["entry"][100, 0] = True
    p = Params("t", max_pos=1, size=0.5, stop=0.99, sma50_exit=False)
    r0, tr0 = simulate(close, sig, p, np.zeros(len(close)), cost_bps=0)
    r1, _ = simulate(close, sig, p, np.zeros(len(close)), cost_bps=100)
    assert (r0.iloc[:102] == 0).all()  # signal at bar 100 -> filled at close of bar 101, first P&L on bar 102
    assert r0.iloc[102] != 0
    assert abs(r1.iloc[101] + 0.5 * 0.01) < 1e-9  # 1% cost on the half-equity position at the fill bar
    assert len(tr0) == 1 and tr0.entry.iloc[0] == 101
