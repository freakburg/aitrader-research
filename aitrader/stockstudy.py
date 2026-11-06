"""Individual big stocks vs an index fund (SPY, total return): how often do they win, and how risky is picking them?

Three parts, fixed before running:
  1. Single stocks. For 154 large US stocks (cached by the cross-sectional study) and every month-end start date, the
     growth of $1 over the next 1/3/5/10/20 years vs SPY over the same window: how often does the stock win?
  2. Concentration. Random equal-weight, buy-and-hold baskets of 1/3/5/10/20/30 of those stocks over 10 years vs SPY.
  3. Past mega-caps. The 10 largest US companies at the end of 1999, 2004, 2009 and 2014 (approximate lists written from
     memory, not an official ranking), equal-weight, held 5 and 10 years vs SPY. Companies whose price history no longer
     exists (acquired or delisted) are marked missing and BOUNDED: best case they match SPY, worst case they go to zero.

SURVIVORSHIP BIAS: parts 1 and 2 use today's large caps, i.e. companies that survived and grew, so they flatter single
stocks. Treat their win rates as UPPER BOUNDS. Part 3 is the attempt to remove the bias, at the cost of approximate lists.
Prices are dividend- and split-adjusted (total return). Windows overlap heavily, so there are far fewer independent
windows than rows; read the numbers as a pattern, not as precise odds.
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR
from .crosssec import load_universe
from .data import get_bars

HORIZONS = [1, 3, 5, 10, 20]
N_LIST = [1, 3, 5, 10, 20, 30]
SNAPSHOTS = {
    "1999": ["MSFT", "GE", "CSCO", "XOM", "WMT", "INTC", "C", "IBM", "ORCL", "LU"],
    "2004": ["XOM", "GE", "MSFT", "C", "PFE", "WMT", "BAC", "JNJ", "AIG", "INTC"],
    "2009": ["XOM", "MSFT", "WMT", "AAPL", "BRK-B", "GE", "GOOGL", "JNJ", "PG", "IBM"],
    "2014": ["AAPL", "XOM", "MSFT", "BRK-B", "JNJ", "GE", "WFC", "WMT", "PG", "JPM"],
}


def month_end(close: pd.DataFrame | pd.Series):
    idx = close.index
    me = np.flatnonzero(np.r_[idx.month[1:] != idx.month[:-1], False])
    return close.iloc[me]


def window_ratios(M: pd.DataFrame, spy: pd.Series, years: int):
    """Growth multiple of each stock and of SPY over the next `years`, for every month-end start; NaN where unavailable."""
    h = 12 * years
    r = M.shift(-h) / M
    s = spy.shift(-h) / spy
    return r, s


def single_stock_table(M: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    rows = {}
    for y in HORIZONS:
        r, s = window_ratios(M, spy, y)
        rel = r.div(s, axis=0).to_numpy().ravel()
        rel = rel[np.isfinite(rel)]
        mult = r.to_numpy().ravel()
        mult = mult[np.isfinite(mult)]
        rows[f"{y}y"] = {"windows": len(rel), "beat_spy": (rel > 1).mean(), "median_vs_spy": np.median(rel),
                         "mean_vs_spy": rel.mean(), "p10_vs_spy": np.percentile(rel, 10),
                         "p90_vs_spy": np.percentile(rel, 90), "lost_money": (mult < 1).mean(),
                         "lost_half_vs_spy": (rel < 0.5).mean()}
    return pd.DataFrame(rows).T


def basket_table(M: pd.DataFrame, spy: pd.Series, years: int = 10, k: int = 500, seed: int = 5) -> pd.DataFrame:
    r, s = window_ratios(M, spy, years)
    rng = np.random.default_rng(seed)
    out = {n: [] for n in N_LIST}
    for t in r.index:
        vals = r.loc[t].dropna().to_numpy()
        sp = s.loc[t]
        if len(vals) < 40 or not np.isfinite(sp):
            continue
        order = np.argsort(rng.random((k, len(vals))), axis=1)
        cs = np.cumsum(vals[order], axis=1)  # buy-and-hold equal-weight multiples for nested baskets
        for n in N_LIST:
            out[n].append((cs[:, n - 1] / n) / sp)
    rows = {}
    for n, parts in out.items():
        rel = np.concatenate(parts)
        rows[f"{n} stocks"] = {"beat_spy": (rel > 1).mean(), "median_vs_spy": np.median(rel),
                               "p10_vs_spy": np.percentile(rel, 10), "p90_vs_spy": np.percentile(rel, 90),
                               "lost_30pct_vs_spy": (rel < 0.7).mean()}
    return pd.DataFrame(rows).T


def snapshot_table(end: str) -> pd.DataFrame:
    spy = get_bars("yahoo", "SPY", "1993-01-01", end)["close"]
    rows = []
    for yr, tickers in SNAPSHOTS.items():
        start = pd.Timestamp(f"{yr}-12-31")
        i0 = spy.index.searchsorted(start, side="right")  # first trading day after the snapshot date
        for years in (5, 10):
            i1 = i0 + 252 * years
            if i1 >= len(spy):
                continue
            d0, d1 = spy.index[i0], spy.index[i1]
            spy_mult = spy.iloc[i1] / spy.iloc[i0]
            mults, missing = {}, []
            for t in tickers:
                try:
                    px = get_bars("yahoo", t, "1993-01-01", end)["close"]
                    if px.index[0] > d0 or d1 > px.index[-1]:
                        raise ValueError("no history over the window")
                    mults[t] = px.loc[d1:d1].iloc[0] / px.loc[d0:d0].iloc[0] if d1 in px.index and d0 in px.index else \
                        px.iloc[px.index.searchsorted(d1)] / px.iloc[px.index.searchsorted(d0)]
                except Exception:
                    missing.append(t)
            w = 1 / len(tickers)
            known = sum(w * m for m in mults.values())
            rows.append({"snapshot": yr, "years": years, "window": f"{d0.date()}..{d1.date()}", "spy_multiple": spy_mult,
                         "basket_known_part": known / (w * len(mults)) if mults else np.nan,
                         "basket_best_case": known + w * len(missing) * spy_mult,
                         "basket_worst_case": known, "missing": ",".join(missing) or "-",
                         "stocks_beating_spy": sum(m > spy_mult for m in mults.values()), "stocks_known": len(mults),
                         "best_stock": max(mults, key=mults.get) if mults else "-",
                         "worst_stock": min(mults, key=mults.get) if mults else "-"})
    return pd.DataFrame(rows)


def run_stock_study() -> None:
    from datetime import date
    end = date.today().isoformat()
    close, _ = load_universe()
    spy = get_bars("yahoo", "SPY", "1993-01-01", end)["close"]
    close = close.reindex(spy.index.union(close.index)).ffill()
    close, spy = close.reindex(spy.index), spy
    M, spy_m = month_end(close), month_end(spy)
    pd.set_option("display.width", 200)
    print(f"{close.shape[1]} large US stocks, {close.index[0].date()}..{close.index[-1].date()}; SPY total return as the index fund.\n"
          "SURVIVORSHIP-BIASED universe: win rates below are upper bounds for stock picking.\n")
    t1 = single_stock_table(M, spy_m)
    t1.to_csv(DATA_DIR / "stockstudy_single.csv")
    print("PART 1: a single stock vs SPY over the next N years, every month-end start (beat_spy = share of stock-windows that won; "
          "vs_spy = stock multiple / SPY multiple; lost_money = stock ended below its start):\n" + t1.round(3).to_string())
    t2 = basket_table(M, spy_m)
    t2.to_csv(DATA_DIR / "stockstudy_baskets.csv")
    print("\nPART 2: random equal-weight baskets, 10 years, vs SPY (500 random baskets per start month):\n" + t2.round(3).to_string())
    t3 = snapshot_table(end)
    t3.to_csv(DATA_DIR / "stockstudy_snapshots.csv", index=False)
    print("\nPART 3: the 10 largest US companies at each date (approximate lists), equal-weight, buy-and-hold vs SPY. "
          "best/worst case bound the companies with no price history left (missing):\n" + t3.round(3).to_string(index=False))
    print("\nReading it: a typical single stock needs to beat the index in most windows to be 'better'; compare beat_spy with 0.5, "
          "and remember parts 1-2 are flattered by survivorship.")
