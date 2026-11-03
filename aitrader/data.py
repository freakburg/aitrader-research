"""Data feed: Alpaca historical bars (free IEX feed) with a CSV cache, plus synthetic data for offline tests."""
import numpy as np
import pandas as pd

from .config import DATA_DIR, alpaca_keys


def synthetic_bars(n=1500, seed=0, start="2018-01-01", freq="B", drift=0.0003, vol=0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998, "close": close, "volume": 1e6},
        index=idx,
    )


def fetch_history(symbol: str, start: str = "1993-01-01", end: str | None = None) -> pd.DataFrame:
    """Decades of daily dividend/split-adjusted bars from Yahoo Finance (unofficial, free). Cached if `end` is fixed."""
    cache = DATA_DIR / f"{symbol}_yahoo_{start}_{end}.csv" if end else None
    if cache is not None and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df = df[df["close"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    if cache is not None:
        df.to_csv(cache)
    return df


def fetch_tbill_daily(start: str = "2003-01-01", end: str | None = None) -> pd.Series:
    """Daily T-bill return from the 13-week yield (^IRX), lagged one day so it is known when used."""
    cache = DATA_DIR / f"IRX_{start}_{end}.csv" if end else None
    if cache is not None and cache.exists():
        y = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        df = yf.download("^IRX", start=start, end=end, progress=False)
        y = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        y.index = pd.DatetimeIndex(y.index).tz_localize(None)
        y = y.dropna()
        if cache is not None:
            y.to_csv(cache)
    return (y / 100 / 252).shift(1).rename("cash")


def fetch_vix(start: str = "2003-01-01", end: str | None = None) -> pd.Series:
    """VIX index close (annualized implied volatility in percent), cached when `end` is fixed."""
    cache = DATA_DIR / f"VIX_{start}_{end}.csv" if end else None
    if cache is not None and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0].rename("vix")
    import yfinance as yf

    df = yf.download("^VIX", start=start, end=end, progress=False)
    y = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    y.index = pd.DatetimeIndex(y.index).tz_localize(None)
    y = y.dropna().rename("vix")
    if cache is not None:
        y.to_csv(cache)
    return y


def get_bars(source: str, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """source: "alpaca" (about 5 years, needs keys) or "yahoo" (decades, no keys)."""
    return fetch_history(symbol, start, end) if source == "yahoo" else fetch_bars(symbol, start, end)


def fetch_bars(symbol: str, start: str, end: str | None = None, timeframe: str = "1Day") -> pd.DataFrame:
    """OHLCV bars indexed by timestamp. Cached on disk only when `end` is fixed."""
    cache = DATA_DIR / f"{symbol}_{timeframe}_{start}_{end}.csv" if end else None
    if cache is not None and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    tf = {"1Day": TimeFrame.Day, "1Hour": TimeFrame.Hour, "1Min": TimeFrame.Minute}[timeframe]
    client = StockHistoricalDataClient(*alpaca_keys())
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") if end else None,
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        raise RuntimeError(f"No bars returned for {symbol}")
    df = df.droplevel("symbol")[["open", "high", "low", "close", "volume"]]
    if timeframe == "1Day":
        df.index = df.index.tz_convert("America/New_York").normalize().tz_localize(None)
    if cache is not None:
        df.to_csv(cache)
    return df
