"""Cross-market gap detector: how often is an apparent price gap between two near-identical instruments
(default SPY vs IVV, both S&P 500 ETFs) actually tradable after bid/ask spreads, fees and latency?

Quote frame columns: bid_a, ask_a, bid_b, ask_b, indexed by time. Gap = deviation of the A/B log price
ratio from its trailing mean. On a signal at sample t we assume orders land `latency` samples later
(crossing the spread both ways), and the position is closed `hold` samples after that.
"""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import DATA_DIR, alpaca_keys


def quotes_from_bars(a: pd.Series, b: pd.Series, half_spread_bps: float = 0.5) -> pd.DataFrame:
    """Approximate a quote frame from two close series by assuming a fixed half-spread."""
    a, b = a.align(b, join="inner")
    h = half_spread_bps / 1e4
    return pd.DataFrame({"bid_a": a * (1 - h), "ask_a": a * (1 + h), "bid_b": b * (1 - h), "ask_b": b * (1 + h)})


def _day_signals(q: pd.DataFrame, threshold_bps, latency, hold, fee_bps, window):
    """Apparent gap size and net P&L (bps) of every signal within one continuous trading session."""
    mid_a, mid_b = (q.bid_a + q.ask_a) / 2, (q.bid_b + q.ask_b) / 2
    ratio = np.log(mid_a / mid_b) * 1e4
    dev = (ratio - ratio.rolling(window).mean().shift(1)).to_numpy()  # trailing mean excludes the current sample

    n = len(q)
    bid_a, ask_a, bid_b, ask_b = (np.log(q[c].to_numpy()) * 1e4 for c in ("bid_a", "ask_a", "bid_b", "ask_b"))
    t = np.arange(max(n - latency - hold, 0))
    t = t[np.abs(dev[t]) > threshold_bps]
    e, x = t + latency, t + latency + hold

    short_a = dev[t] > 0  # A looks rich: sell A / buy B, then unwind
    pnl = np.where(
        short_a,
        (bid_a[e] - ask_a[x]) + (bid_b[x] - ask_b[e]),
        (bid_a[x] - ask_a[e]) + (bid_b[e] - ask_b[x]),
    ) - fee_bps
    return n, np.abs(dev[t]), pnl


def analyze_gaps(q: pd.DataFrame, threshold_bps: float = 2.0, latency: int = 1, hold: int = 5,
                 fee_bps: float = 0.1, window: int = 60) -> dict:
    """Multi-day frames are split by calendar day so no trade or rolling average spans the overnight gap."""
    days = [g for _, g in q.groupby(q.index.date)] if isinstance(q.index, pd.DatetimeIndex) else [q]
    parts = [_day_signals(g, threshold_bps, latency, hold, fee_bps, window) for g in days]
    n = sum(p[0] for p in parts)
    gross_signal = np.concatenate([p[1] for p in parts])
    pnl = np.concatenate([p[2] for p in parts])
    t = gross_signal
    return {
        "samples": n,
        "days": len(days),
        "signals": int(len(t)),
        "signals_per_1000_samples": float(len(t) / max(n, 1) * 1000),
        "mean_apparent_gap_bps": float(gross_signal.mean()) if len(t) else 0.0,
        "mean_net_bps": float(pnl.mean()) if len(t) else 0.0,
        "pct_net_positive": float((pnl > 0).mean()) if len(t) else 0.0,
        "total_net_bps": float(pnl.sum()) if len(t) else 0.0,
        "latency_samples": latency,
        "hold_samples": hold,
    }


def sweep(q: pd.DataFrame, thresholds=(1, 2, 3, 5), latencies=(0, 1, 2, 5), **kw) -> pd.DataFrame:
    rows = [analyze_gaps(q, th, lat, **kw) | {"threshold_bps": th} for th in thresholds for lat in latencies]
    cols = ["threshold_bps", "latency_samples", "signals", "mean_apparent_gap_bps", "mean_net_bps",
            "pct_net_positive", "total_net_bps"]
    return pd.DataFrame(rows)[cols]


def market_is_open() -> tuple[bool, str]:
    from alpaca.trading.client import TradingClient

    clock = TradingClient(*alpaca_keys(), paper=True).get_clock()
    return clock.is_open, f"market {'open' if clock.is_open else 'closed'}; next open {clock.next_open}"


def record_quotes(sym_a: str, sym_b: str, minutes: float = 30, interval: float = 1.0, path=None,
                  require_open: bool = True) -> "Path":
    """Sample live bid/ask for both symbols (market data only, no orders) and append to a CSV."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    is_open, msg = market_is_open()
    if require_open and not is_open:
        raise RuntimeError(f"Not recording: {msg}")

    client = StockHistoricalDataClient(*alpaca_keys())
    path = path or DATA_DIR / f"quotes_{sym_a}_{sym_b}.csv"
    cols = ["time_utc", "bid_a", "ask_a", "bid_b", "ask_b", "rtt_ms"]
    new = not path.exists()
    deadline = time.time() + minutes * 60
    with open(path, "a") as f:
        if new:
            f.write(",".join(cols) + "\n")
        while time.time() < deadline:
            t0 = time.time()
            try:
                q = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[sym_a, sym_b],
                                                                          feed=DataFeed.IEX))
            except Exception as exc:  # network blip / rate limit: skip this sample, keep recording
                print(f"{datetime.now():%H:%M:%S} sample failed: {exc}")
                time.sleep(5)
                continue
            rtt = (time.time() - t0) * 1000
            a, b = q[sym_a], q[sym_b]
            if min(a.bid_price, a.ask_price, b.bid_price, b.ask_price) > 0:
                f.write(f"{datetime.now(timezone.utc).isoformat()},{a.bid_price},{a.ask_price},"
                        f"{b.bid_price},{b.ask_price},{rtt:.0f}\n")
                f.flush()
            time.sleep(max(0, interval - (time.time() - t0)))
    return path


def record_sessions(sym_a: str, sym_b: str, days: int = 1, interval: float = 1.0,
                    open_buffer_min: float = 5, close_buffer_min: float = 5) -> None:
    """Unattended: wait for the market to open, record until shortly before the close, repeat for `days` sessions.
    Skips the noisy first/last minutes of each session. Keeps the PC awake while running (Windows)."""
    from alpaca.trading.client import TradingClient

    from .config import keep_awake
    keep_awake()

    trading = TradingClient(*alpaca_keys(), paper=True)
    recorded, waited = 0, False
    while recorded < days:
        try:
            clock = trading.get_clock()
        except Exception as exc:
            print(f"{datetime.now():%H:%M:%S} clock check failed: {exc}")
            time.sleep(30)
            continue
        if not clock.is_open:
            waited = True
            secs = (clock.next_open - clock.timestamp).total_seconds()
            print(f"{datetime.now():%H:%M:%S} market closed; next open {clock.next_open}; "
                  f"sessions recorded {recorded}/{days}")
            time.sleep(min(max(secs, 5), 600))
            continue
        if waited:  # we just saw the open: skip the first minutes
            time.sleep(open_buffer_min * 60)
            waited = False
            continue
        minutes = (clock.next_close - clock.timestamp).total_seconds() / 60 - close_buffer_min
        if minutes <= 1:
            time.sleep(60)
            continue
        print(f"{datetime.now():%H:%M:%S} recording {minutes:.0f} minutes")
        record_quotes(sym_a, sym_b, minutes, interval, require_open=False)
        recorded += 1
        waited = True
    print("Done:", recorded, "sessions recorded")


def load_quotes(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)
