"""Walk-forward optimization: tune parameters on a trailing window, trade the next window blind, repeat."""
import itertools
from dataclasses import dataclass

import pandas as pd

from .backtest import compute_metrics, run_backtest


@dataclass
class WalkForwardResult:
    returns: pd.Series
    position: pd.Series
    params_log: pd.DataFrame
    metrics: dict


def walk_forward(df: pd.DataFrame, make_strategy, grid: dict, train_bars: int = 504, test_bars: int = 63,
                 cost_bps: float = 1.0, objective: str = "sharpe") -> WalkForwardResult:
    combos = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        try:
            make_strategy(**params)
        except ValueError:
            continue
        combos.append(params)
    if not combos:
        raise ValueError("No valid parameter combinations in grid")
    if len(df) <= train_bars:
        raise ValueError("Not enough data for one training window")

    rets, poss, log = [], [], []
    i = train_bars
    while i < len(df):
        end = min(i + test_bars, len(df))
        train = df.iloc[i - train_bars:i]
        best = max(combos, key=lambda p: run_backtest(train, make_strategy(**p), cost_bps).metrics[objective])
        # Include the training window as indicator warm-up, keep only the out-of-sample slice.
        res = run_backtest(df.iloc[i - train_bars:end], make_strategy(**best), cost_bps)
        n = end - i
        rets.append(res.returns.iloc[-n:])
        poss.append(res.position.iloc[-n:])
        log.append({"start": df.index[i], **best})
        i = end

    returns, position = pd.concat(rets), pd.concat(poss)
    bench = df["close"].pct_change().reindex(returns.index).fillna(0)
    return WalkForwardResult(returns, position, pd.DataFrame(log), compute_metrics(returns, position, bench))
