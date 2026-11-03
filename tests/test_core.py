import numpy as np
import pandas as pd

from aitrader.backtest import run_backtest
from aitrader.data import synthetic_bars
from aitrader.gap import analyze_gaps, quotes_from_bars
from aitrader.ml_strategy import MLStrategy
from aitrader.strategy import Breakout, RSIReversion, SMACrossover, Strategy, TrendFilter, VolTarget
from aitrader.walkforward import walk_forward


class AlwaysLong(Strategy):
    name = "long"

    def signals(self, df):
        return pd.Series(1.0, index=df.index)


def test_always_long_matches_buy_and_hold_after_first_bar():
    df = synthetic_bars(300)
    res = run_backtest(df, AlwaysLong(), cost_bps=0)
    expected = (1 + df["close"].pct_change().fillna(0).iloc[2:]).prod() - 1
    got = (1 + res.returns.iloc[2:]).prod() - 1
    assert abs(got - expected) < 1e-9


def test_no_lookahead_future_edit_does_not_change_past_signals():
    df = synthetic_bars(400)
    for strat in (SMACrossover(10, 30), TrendFilter(50), VolTarget(0.1, 20), RSIReversion(2, 10, 60),
                  Breakout(30, 10), MLStrategy(min_train=150, retrain_every=21),
                  MLStrategy(min_train=150, retrain_every=21, horizon=5, window="rolling", model="hgb", depth=2)):
        base = strat.signals(df)
        df2 = df.copy()
        df2.loc[df2.index[300:], "close"] *= 1.5  # change only the future
        changed = strat.signals(df2)
        pd.testing.assert_series_equal(base.iloc[:300], changed.iloc[:300])


def test_rotation_has_no_lookahead_and_holds_cash_in_downtrend():
    from aitrader.rotation import rotation_returns
    idx = pd.date_range("2010-01-01", periods=900, freq="B")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame({
        "UP": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.005, 900))),
        "DOWN": 100 * np.exp(np.cumsum(rng.normal(-0.0008, 0.005, 900))),
        "SHY": 100 * np.exp(np.cumsum(np.full(900, 0.0001))),
    }, index=idx)
    base = rotation_returns(closes, lookback_m=6, top_k=1, skip_m=0)
    changed = closes.copy()
    changed.loc[idx[700:], ["UP", "DOWN"]] *= 0.5  # alter only the future
    pd.testing.assert_series_equal(base.iloc[:700], rotation_returns(changed, 6, 1, 0).iloc[:700])
    assert base.iloc[300:].mean() > 0  # it ends up in UP / cash, not DOWN


def test_crosssec_scores_ignore_the_future():
    from aitrader.crosssec import build_panel, walk_forward_scores
    rng = np.random.default_rng(3)
    idx = pd.date_range("2005-01-03", periods=1800, freq="B")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (1800, 25)), axis=0)), index=idx,
                         columns=[f"S{i}" for i in range(25)])
    volume = pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=idx, columns=close.columns)
    base = walk_forward_scores(build_panel(close, volume), "ridge", min_train=24, retrain_every=6)
    changed = close.copy()
    changed.iloc[1500:, :12] *= 1.4  # alter only the far future
    after = walk_forward_scores(build_panel(changed, volume), "ridge", min_train=24, retrain_every=6)
    cutoff = idx[1400]
    pd.testing.assert_series_equal(base[base.index.get_level_values("date") < cutoff].dropna(),
                                   after[after.index.get_level_values("date") < cutoff].dropna())
    assert base.notna().sum() > 0


def test_paper_sizing_handles_partial_positions():
    from aitrader.paper import plan_trade
    # 60% weight of $10k at $500/share = 12 shares
    assert plan_trade(0.6, 500, 0, 10_000) == {"action": "buy", "qty": 12.0, "desired": 12.0, "close_all": False}
    # already holding 12: nothing to do; a tiny drift under the 2% band is ignored
    assert plan_trade(0.6, 500, 12, 10_000)["action"] == "hold"
    assert plan_trade(0.61, 500, 12, 10_000)["action"] == "hold"
    # weight drops to 30%: sell down to 6 shares
    assert plan_trade(0.3, 500, 12, 10_000) == {"action": "sell", "qty": 6.0, "desired": 6.0, "close_all": False}
    # full exit always goes through, however small the position
    assert plan_trade(0.0, 500, 0.2, 10_000)["close_all"] is True
    # weights outside 0-1 are clipped
    assert plan_trade(1.7, 500, 0, 10_000)["desired"] == 20.0


def test_leverage_financing_and_cash_math():
    from aitrader.riskstudy import portfolio_returns
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    r = pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=idx)
    cash = pd.Series(0.0002, index=idx)
    # half invested: half the move plus cash on the other half (position set at close t acts on t+1)
    half = portfolio_returns(r, pd.Series(0.5, index=idx), cash, spread=0.0, cost_bps=0.0)
    assert abs(half.iloc[3] - (0.5 * 0.01 + 0.5 * 0.0002)) < 1e-12
    # 1.5x: extra half borrowed at the cash rate + spread
    lev = portfolio_returns(r, pd.Series(1.5, index=idx), cash, spread=0.005, cost_bps=0.0)
    assert abs(lev.iloc[3] - (1.5 * 0.01 - 0.5 * 0.0002 - 0.5 * 0.005 / 252)) < 1e-12
    # the first bar uses no position (decided at the previous close), so it earns only cash
    assert abs(half.iloc[0] - 0.0002) < 1e-12


def test_vol_forecasters_ignore_the_future():
    from aitrader.volforecast import garch_asset, har_forecast, pooled_forecast
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-04", periods=700, freq="B")
    scale = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, (700, 3)), axis=0) * 0.2)
    R = pd.DataFrame(rng.normal(0, 1, (700, 3)) * scale, index=idx, columns=list("ABC"))
    vix = pd.Series(0.2 + 0.05 * rng.random(700), index=idx)
    vixvar = (vix ** 2) / 252
    changed = R.copy()
    changed.iloc[560:] *= 3.0  # alter only the future
    cut = 520
    for fn in (lambda r: har_forecast(r, vixvar, True, refit_every=30, warm=400),
               lambda r: pooled_forecast(r, vixvar, "gbm", True, refit_every=60, warm=400),
               lambda r: pd.DataFrame({"A": garch_asset(r["A"], refit_every=60, warm=400)})):
        a, b = fn(R).iloc[:cut], fn(changed).iloc[:cut]
        assert a.iloc[400:].notna().all().all()  # forecasts exist after the warm-up
        pd.testing.assert_frame_equal(a, b, atol=1e-12)


def test_regime_probabilities_ignore_the_future():
    from aitrader.regime import hmm_filtered, shift_test
    rng = np.random.default_rng(9)
    state = np.cumsum(rng.random(900) < 0.02) % 2  # alternating calm / stress periods
    r = pd.Series(np.where(state == 0, rng.normal(0.0006, 0.007, 900), rng.normal(-0.001, 0.02, 900)),
                  index=pd.date_range("2010-01-04", periods=900, freq="B"))
    changed = r.copy()
    changed.iloc[700:] *= 2.5  # alter only the future
    a = hmm_filtered(r, refit_every=60, warm=300).iloc[:640]
    b = hmm_filtered(changed, refit_every=60, warm=300).iloc[:640]
    pd.testing.assert_frame_equal(a, b, atol=1e-9)
    p = a["p_stress"].iloc[300:]
    assert p.notna().all() and p.between(-1e-9, 1 + 1e-9).all()
    # random timing cannot beat itself: a constant position has p close to 1
    cash = pd.Series(0.0001, index=r.index)
    _, p = shift_test(pd.Series(1.0, index=r.index), r, cash, lo=300, k=200)
    assert p > 0.4


def test_global_study_weights_use_only_past_and_scale_with_ppy():
    from aitrader.globalstudy import clean, weights
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))), index=pd.date_range("2015-01-01", periods=600))
    R = close.pct_change().fillna(0.0)
    changed = close.copy()
    changed.iloc[400:] *= 1.5  # alter only the future
    Rc = changed.pct_change().fillna(0.0)
    for kind in ("vt10_cap1", "trend200"):
        a = weights(R, close, kind, 252).iloc[:399]
        b = weights(Rc, changed, kind, 252).iloc[:399]
        pd.testing.assert_series_equal(a, b)
    # a 365-day market sees larger annualized vol for the same daily moves, so it is invested less
    assert weights(R, close, "vt10_cap1", 365).iloc[100:].mean() < weights(R, close, "vt10_cap1", 252).iloc[100:].mean()
    spiked = close.copy()
    spiked.iloc[300] *= 3  # a 200% one-day jump is treated as a bad print
    assert len(clean(spiked, 252, "x")) < len(spiked)


def test_forward_tracks_are_isolated_and_frozen(tmp_path, monkeypatch):
    import json
    from aitrader import paper
    assert paper.TRACKS["vol"]["rule"] == {"track": "vol_target_spy_v1", "symbol": "SPY", "strategy": "vol_target",
                                           "target_vol": 0.10, "window": 20, "cap": 1.0}  # the original rule is unchanged
    syms = {t["rule"]["symbol"] for t in paper.TRACKS.values()}
    logs = {t["log"] for t in paper.TRACKS.values()}
    specs = {t["spec"] for t in paper.TRACKS.values()}
    assert len(syms) == len(logs) == len(specs) == len(paper.TRACKS)  # separate symbol, log and spec per track
    # editing a track's rule after it is pinned is refused
    monkeypatch.setitem(paper.TRACKS["trend"], "spec", tmp_path / "spec.json")
    paper.ensure_spec("trend")
    saved = json.loads((tmp_path / "spec.json").read_text())["rule"]
    assert saved == paper.TREND_RULE
    monkeypatch.setitem(paper.TRACKS["trend"], "rule", {**paper.TREND_RULE, "window": 150})
    import pytest
    with pytest.raises(RuntimeError):
        paper.ensure_spec("trend")


def test_costs_reduce_returns():
    df = synthetic_bars(500)
    s = SMACrossover(5, 20)
    assert run_backtest(df, s, 20).metrics["total_return"] < run_backtest(df, s, 0).metrics["total_return"]


def test_walk_forward_covers_out_of_sample_only():
    df = synthetic_bars(900)
    res = walk_forward(df, SMACrossover, {"fast": [5, 10], "slow": [30, 60]}, train_bars=300, test_bars=100)
    assert res.returns.index[0] == df.index[300]
    assert len(res.returns) == 600
    assert len(res.params_log) == 6


def test_gap_detector_counts_real_gaps_but_spread_kills_tiny_ones():
    rng = np.random.default_rng(1)
    n = 3000
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    noise = rng.normal(0, 3e-4, n)  # ~3 bps of idiosyncratic noise between the two instruments
    a, b = pd.Series(base * (1 + noise)), pd.Series(base)
    cheap = analyze_gaps(quotes_from_bars(a, b, half_spread_bps=0.0), threshold_bps=3, latency=1, hold=1, fee_bps=0)
    costly = analyze_gaps(quotes_from_bars(a, b, half_spread_bps=2.0), threshold_bps=3, latency=1, hold=1, fee_bps=0)
    assert cheap["signals"] > 0
    assert costly["mean_net_bps"] < cheap["mean_net_bps"]
