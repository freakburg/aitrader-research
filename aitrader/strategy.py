"""Strategy interface. A strategy maps a bar history to a target position (0 = flat, 1 = long) per bar.

The signal at bar t may only use data up to and including bar t; the backtester trades it on bar t+1.
"""
import pandas as pd


import numpy as np


class Strategy:
    name = "base"
    params: dict = {}
    warmup = 0  # bars before the strategy can produce meaningful signals; metrics start after this

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class SMACrossover(Strategy):
    name = "sma"

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast, self.slow = fast, slow
        self.warmup = slow
        self.params = {"fast": fast, "slow": slow}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].rolling(self.fast).mean()
        slow = df["close"].rolling(self.slow).mean()
        return (fast > slow).astype(float)


class TrendFilter(Strategy):
    """Hold while price is above its long moving average, otherwise cash."""
    name = "trend"

    def __init__(self, window: int = 200):
        self.window, self.warmup = window, window
        self.params = {"window": window}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] > df["close"].rolling(self.window).mean()).astype(float)


def _latch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Long from an enter signal until an exit signal; flat before the first enter."""
    state = pd.Series(np.nan, index=enter.index)
    state[enter] = 1.0
    state[exit_] = 0.0
    return state.ffill().fillna(0.0)


class RSIReversion(Strategy):
    """Buy short-term oversold dips (RSI below `lo`), sell when RSI recovers above `hi`."""
    name = "rsi"

    def __init__(self, period: int = 2, lo: float = 10, hi: float = 60):
        self.period, self.lo, self.hi, self.warmup = period, lo, hi, period + 1
        self.params = {"period": period, "lo": lo, "hi": hi}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        d = df["close"].diff()
        gain = d.clip(lower=0).rolling(self.period).mean()
        loss = (-d.clip(upper=0)).rolling(self.period).mean().replace(0, 1e-12)
        rsi = 100 - 100 / (1 + gain / loss)
        return _latch(rsi < self.lo, rsi > self.hi)


class Breakout(Strategy):
    """Buy a new `n`-bar closing high, exit on a new `m`-bar closing low."""
    name = "breakout"

    def __init__(self, n: int = 50, m: int = 20):
        self.n, self.m, self.warmup = n, m, n
        self.params = {"n": n, "m": m}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        return _latch(c > c.rolling(self.n).max().shift(1), c < c.rolling(self.m).min().shift(1))


class VolTarget(Strategy):
    """Always invested, but scale the position down when recent volatility exceeds the target."""
    name = "vol"

    def __init__(self, target_vol: float = 0.10, window: int = 20, ppy: int = 252):
        self.target_vol, self.window, self.ppy, self.warmup = target_vol, window, ppy, window
        self.params = {"target_vol": target_vol, "window": window}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        vol = df["close"].pct_change().rolling(self.window).std() * np.sqrt(self.ppy)
        return (self.target_vol / vol).clip(upper=1.0).fillna(0.0)
