"""Run every strategy on several tickers over the SAME window and compare each to buy & hold.

Tickers and strategies are fixed up front and every result is reported (no cherry-picking).
"""
import pandas as pd

from .backtest import run_backtest
from .ml_strategy import MLStrategy
from .strategy import SMACrossover, TrendFilter, VolTarget
from .walkforward import walk_forward

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "AAPL", "JPM"]
WARMUP = 500  # common start bar for every strategy, so all are measured over the same period


def evaluate(df: pd.DataFrame, cost_bps: float = 1.0) -> dict[str, dict]:
    out = {}
    strategies = {
        "sma_20_50": SMACrossover(20, 50),
        "trend_200": TrendFilter(200),
        "vol_target_10": VolTarget(0.10),
        "ml_logreg": MLStrategy(min_train=WARMUP),
    }
    for name, s in strategies.items():
        out[name] = run_backtest(df, s, cost_bps, warmup=WARMUP).metrics
    grid = {"fast": [5, 10, 20, 30], "slow": [40, 50, 100, 150, 200]}
    out["walkforward_sma"] = walk_forward(df, SMACrossover, grid, train_bars=WARMUP, test_bars=63,
                                          cost_bps=cost_bps).metrics
    return out


def compare(frames: dict[str, pd.DataFrame], cost_bps: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sym, df in frames.items():
        for name, m in evaluate(df, cost_bps).items():
            rows.append({"symbol": sym, "strategy": name, "return": m["total_return"], "sharpe": m["sharpe"],
                         "max_dd": m["max_drawdown"], "exposure": m["exposure"],
                         "bh_return": m["buy_hold_return"], "bh_sharpe": m["buy_hold_sharpe"],
                         "bh_max_dd": m["buy_hold_max_drawdown"]})
    table = pd.DataFrame(rows)
    table["beats_bh_sharpe"] = table["sharpe"] > table["bh_sharpe"]
    table["beats_bh_return"] = table["return"] > table["bh_return"]
    table["smaller_dd"] = table["max_dd"] > table["bh_max_dd"]  # drawdowns are negative
    summary = table.groupby("strategy").agg(
        tickers=("symbol", "count"), beats_bh_sharpe=("beats_bh_sharpe", "sum"),
        beats_bh_return=("beats_bh_return", "sum"), smaller_dd=("smaller_dd", "sum"),
        avg_sharpe=("sharpe", "mean"), avg_bh_sharpe=("bh_sharpe", "mean"),
        avg_max_dd=("max_dd", "mean"), avg_bh_max_dd=("bh_max_dd", "mean"))
    return table, summary
