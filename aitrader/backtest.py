from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series
    position: pd.Series
    equity: pd.Series
    metrics: dict


def compute_metrics(strat_ret: pd.Series, pos: pd.Series, bench_ret: pd.Series, ppy: int = 252) -> dict:
    def sharpe(r):
        sd = r.std()
        return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0

    equity = (1 + strat_ret).cumprod()
    years = max(len(strat_ret) / ppy, 1e-9)
    bench_eq = (1 + bench_ret).cumprod()
    entries = int(((pos > 0) & (pos.shift(1).fillna(0) == 0)).sum())
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "sharpe": sharpe(strat_ret),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trades": entries,
        "exposure": float((pos > 0).mean()),
        "buy_hold_return": float(bench_eq.iloc[-1] - 1),
        "buy_hold_sharpe": sharpe(bench_ret),
        "buy_hold_max_drawdown": float((bench_eq / bench_eq.cummax() - 1).min()),
    }


def position_returns(close: pd.Series, sig: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    """Per-bar strategy returns: yesterday's signal held over today's return, minus turnover costs."""
    pos = sig.reindex(close.index).fillna(0).clip(0, 1).shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * close.pct_change().fillna(0) - turnover * cost_bps / 1e4


def run_backtest(df: pd.DataFrame, strategy: Strategy, cost_bps: float = 1.0,
                 initial_cash: float = 10_000.0, ppy: int = 252, warmup: int | None = None) -> BacktestResult:
    """Long/flat backtest. Signal at close of bar t is traded at the close of bar t+1's return
    (position shifted one bar), so there is no lookahead. `cost_bps` is charged per unit of turnover.
    Metrics (strategy and buy & hold) cover only the bars after `strategy.warmup`, so both are
    measured over the same window."""
    sig = strategy.signals(df).reindex(df.index).fillna(0).clip(0, 1)
    pos = sig.shift(1).fillna(0)
    ret = df["close"].pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_ret = pos * ret - turnover * cost_bps / 1e4
    w = min(strategy.warmup if warmup is None else warmup, len(df) - 2)
    return BacktestResult(
        returns=strat_ret,
        position=pos,
        equity=initial_cash * (1 + strat_ret).cumprod(),
        metrics=compute_metrics(strat_ret.iloc[w:], pos.iloc[w:], ret.iloc[w:], ppy),
    )
