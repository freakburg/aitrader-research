"""Minervini-style trend-template (SEPA-like) stock portfolio, tested against buy-and-hold.

Rules, all computed from daily closes available at the signal bar:
  trend template: close > 150d and 200d SMA, 150d > 200d, 200d rising over the last month, 50d > 150d,
                  close > 50d SMA, >= 30% above the 52-week low, within 25% of the 52-week high,
                  12-1 month momentum rank in the top 30% of the universe.
  trigger:        close above the prior 50-day closing high on >= 1.4x average volume.
  exit:           close at least `stop` below entry, then below the 50-day SMA (or a trailing stop).
  market filter:  no new entries (and everything sold) while SPY is below its 200-day SMA.

Orders decided at the close of bar t are filled at the close of bar t+1 (one extra bar of delay compared with
the single-asset backtester), at `cost_bps` per side. Idle cash earns the T-bill rate. Positions are sized as a
fixed fraction of equity at entry (Minervini-style: risk ~1% of equity with a ~8% stop is ~12% of equity).

Not modelled: intraday stops (a stop is triggered on the close and filled the next close, so gaps and slippage
are partly captured but fills are not intraday-realistic), "tight base" pattern recognition, fundamentals.
SURVIVORSHIP BIAS: the universe is today's large caps. The equal-weight basket of the same stocks is the fair
yardstick for stock selection; SPY is the fair yardstick for "should I just hold the index".
"""
import time
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .crosssec import load_universe
from .data import fetch_tbill_daily
from .sweep import BLOCK

MIN_HISTORY = 260  # bars before the first scored day (252-day high/low + 1)


@dataclass(frozen=True)
class Params:
    name: str
    max_pos: int = 8
    size: float = 0.125          # fraction of equity per new position
    stop: float = 0.08           # initial stop below entry (close basis)
    trail: float | None = None   # trailing stop below the post-entry peak close
    sma50_exit: bool = True      # exit when close < 50-day SMA
    market_filter: bool = False  # SPY above its 200-day SMA required
    breakout: bool = True        # require the volume breakout trigger (else any template stock by RS rank)
    rs_min: float = 0.70


VARIANTS = [
    Params("base"),
    Params("market_filter", market_filter=True),
    Params("trail20_mkt", sma50_exit=False, trail=0.20, market_filter=True),
    Params("template_hold_mkt", breakout=False, market_filter=True),
    Params("diversified15_mkt", max_pos=15, size=1 / 15, market_filter=True),
]


def build_signals(close: pd.DataFrame, volume: pd.DataFrame | None, spy: pd.Series, p: Params) -> dict:
    """Boolean/float panels aligned to `close`. Every value at bar t uses data up to and including bar t."""
    c = close
    sma50, sma150, sma200 = (c.rolling(w).mean() for w in (50, 150, 200))
    hi52, lo52 = c.rolling(252).max(), c.rolling(252).min()
    template = ((c > sma150) & (c > sma200) & (sma150 > sma200) & (sma200 > sma200.shift(21)) &
                (sma50 > sma150) & (c > sma50) & (c >= 1.3 * lo52) & (c >= 0.75 * hi52))
    rs = (c.shift(21) / c.shift(252) - 1).rank(axis=1, pct=True)
    eligible = template & (rs >= p.rs_min)
    if p.breakout:
        trig = c > c.rolling(50).max().shift(1)
        if volume is not None:
            trig &= volume > 1.4 * volume.rolling(50).mean().shift(1)
        entry = eligible & trig
    else:
        entry = eligible
    spy_ok = (spy > spy.rolling(200).mean()).reindex(c.index).fillna(False)
    return {"entry": entry.to_numpy(), "rs": rs.fillna(0).to_numpy(), "below50": (c < sma50).to_numpy(),
            "mkt_ok": spy_ok.to_numpy()}


def simulate(close: pd.DataFrame, sig: dict, p: Params, cash_ret: np.ndarray, cost_bps: float = 10.0):
    """Daily portfolio simulation. Returns (daily returns Series, trades DataFrame)."""
    C = close.to_numpy()
    T, N = C.shape
    R = np.nan_to_num(C[1:] / C[:-1] - 1, nan=0.0, posinf=0.0, neginf=0.0)
    cost = cost_bps / 1e4
    v = np.zeros(N)                    # dollar value per position
    entry_px, peak = np.zeros(N), np.zeros(N)
    entry_day = np.zeros(N, dtype=int)
    held = np.zeros(N, dtype=bool)
    cash, eq = 1.0, np.ones(T)
    pend_exit: list[int] = []
    pend_entry: list[int] = []
    trades = []
    for t in range(1, T):
        v *= 1 + R[t - 1]
        cash *= 1 + cash_ret[t]
        for s in pend_exit:            # fills at this bar's close for decisions made at t-1
            if held[s]:
                cash += v[s] * (1 - cost)
                trades.append((s, entry_day[s], t, C[t, s] / entry_px[s] - 1))
                v[s], held[s] = 0.0, False
        pend_exit = []
        equity = cash + v.sum()
        for s in pend_entry:
            size = min(p.size * equity, cash)
            if size > 1e-9 and not held[s] and np.isfinite(C[t, s]):
                cash -= size
                v[s], held[s] = size * (1 - cost), True
                entry_px[s] = peak[s] = C[t, s]
                entry_day[s] = t
        pend_entry = []
        eq[t] = cash + v.sum()

        # decisions at this close, filled at the next
        ok = sig["mkt_ok"][t] if p.market_filter else True
        for s in np.flatnonzero(held):
            peak[s] = max(peak[s], C[t, s])
            out = (not ok) or C[t, s] <= entry_px[s] * (1 - p.stop)
            if p.trail is not None and C[t, s] <= peak[s] * (1 - p.trail):
                out = True
            if p.sma50_exit and sig["below50"][t, s]:
                out = True
            if out:
                pend_exit.append(s)
        if ok:
            slots = p.max_pos - (held.sum() - len(pend_exit))
            if slots > 0:
                cand = np.flatnonzero(sig["entry"][t] & ~held)
                if len(cand):
                    cand = cand[np.argsort(-sig["rs"][t, cand])][:slots]
                    pend_entry = [int(s) for s in cand]
    for s in np.flatnonzero(held):     # mark open trades to the last close
        trades.append((s, entry_day[s], T - 1, C[T - 1, s] / entry_px[s] - 1))
    ret = pd.Series(eq, index=close.index).pct_change().fillna(0.0)
    tr = pd.DataFrame(trades, columns=["stock", "entry", "exit", "ret"])
    tr["stock"] = [close.columns[i] for i in tr["stock"]]
    return ret, tr


def perf(r: pd.Series, cash: pd.Series | None = None) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    ex = r - cash.reindex(r.index).fillna(0) if cash is not None else r
    return {"cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1)}


def alpha_vs(r: pd.Series, bench: pd.Series, cash: pd.Series) -> dict:
    """Daily regression of excess strategy return on excess benchmark return: annual alpha, beta, alpha t-stat."""
    y, x = (r - cash).to_numpy(), (bench - cash).to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    se = np.sqrt(res.var(ddof=2) * np.linalg.inv(X.T @ X)[0, 0])
    return {"alpha_yr": float(coef[0] * 252), "beta": float(coef[1]), "alpha_t": float(coef[0] / se)}


def load_inputs():
    close, volume = load_universe()
    spy = pd.read_csv(DATA_DIR / "SPY_yahoo_1993-01-01_2026-11-02.csv", index_col=0, parse_dates=True)["close"]
    spy = spy.reindex(close.index).ffill()
    cash = fetch_tbill_daily("1993-01-01", date.today().isoformat()).reindex(close.index).ffill().fillna(0.0)
    return close, volume, spy, cash


def run_variants(close, volume, spy, cash, variants=VARIANTS, cost_bps=10.0):
    out = {}
    for p in variants:
        sig = build_signals(close, volume, spy, p)
        out[p.name] = simulate(close, sig, p, cash.to_numpy(), cost_bps)
    return out


def _bootstrap_idx(T: int, rng: np.random.Generator) -> np.ndarray:
    idx: list[int] = []
    while len(idx) < T:
        s = int(rng.integers(0, T - BLOCK + 1))
        idx.extend(range(s, s + BLOCK))
    return np.array(idx[:T])


def null_panel(close: pd.DataFrame, volume: pd.DataFrame, spy: pd.Series, rng: np.random.Generator):
    """Resample whole days in blocks, jointly across stocks, SPY and volume: keeps return distribution and
    correlations, destroys any real trend/momentum structure."""
    idx = _bootstrap_idx(len(close), rng)
    R = close.pct_change().fillna(0).to_numpy()[idx]
    fake = pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=close.index, columns=close.columns)
    V = pd.DataFrame(volume.to_numpy()[idx], index=close.index, columns=close.columns)
    s = pd.Series(100 * np.cumprod(1 + spy.pct_change().fillna(0).to_numpy()[idx]), index=close.index)
    return fake, V, s


def run_minervini(reps: int = 200, cost_bps: float = 10.0, seed: int = 7):
    t0 = time.time()
    close, volume, spy, cash = load_inputs()
    start = close.index[MIN_HISTORY]
    print(f"universe {close.shape[1]} stocks, {close.index[0].date()} -> {close.index[-1].date()}, "
          f"scored from {start.date()}; cost {cost_bps:g} bps/side; cash earns T-bill")
    res = run_variants(close, volume, spy, cash, cost_bps=cost_bps)
    print(f"simulated {len(res)} variants in {time.time() - t0:.0f}s")

    ret = close.pct_change()
    bench = {"SPY buy&hold": spy.pct_change().fillna(0),
             "equal-weight basket": ret.mean(axis=1).fillna(0)}
    idx = close.index[MIN_HISTORY:]
    n = len(idx)
    segs = {"all": (0, n), "search 50%": (0, n // 2), "validation 25%": (n // 2, 3 * n // 4),
            "holdout 25%": (3 * n // 4, n)}

    rows = []
    allr = {**{k: v for k, v in bench.items()}, **{k: v[0] for k, v in res.items()}}
    for seg, (a, b) in segs.items():
        ix = idx[a:b]
        for name, r in allr.items():
            m = perf(r.loc[ix], cash)
            row = {"segment": seg, "strategy": name, **m}
            if name in res:
                row.update(alpha_vs(r.loc[ix], bench["SPY buy&hold"].loc[ix], cash.loc[ix]))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "minervini_results.csv", index=False)

    pd.set_option("display.width", 200)
    for seg in segs:
        print(f"\n=== {seg} ===")
        t = table[table.segment == seg].set_index("strategy")
        t = t[["cagr", "sharpe", "max_dd", "alpha_yr", "beta", "alpha_t"]]
        t.columns = ["CAGR", "Sharpe", "MaxDD", "alpha/yr vs SPY", "beta", "alpha t"]
        print(t.round(3).to_string())

    print("\n=== trade statistics (all trades) ===")
    for name, (_, tr) in res.items():
        w = tr[tr.ret > 0].ret
        l = tr[tr.ret <= 0].ret
        print(f"{name:20s} trades {len(tr):5d}  win rate {len(w) / max(len(tr), 1):.0%}  "
              f"avg win {w.mean():+.1%}  avg loss {l.mean():+.1%}  "
              f"payoff {abs(w.mean() / l.mean()) if len(l) and len(w) else float('nan'):.2f}")

    # luck test on a balanced panel (stocks with complete history from 2000), real vs bootstrapped
    bal_start = "2000-01-03"
    bal = close.loc[bal_start:].dropna(axis=1)
    bv = volume.loc[bal_start:, bal.columns]
    bspy, bcash = spy.loc[bal_start:], cash.loc[bal_start:]
    real = {p.name: perf(simulate(bal, build_signals(bal, bv, bspy, p), p, bcash.to_numpy(), cost_bps)[0]
                          .iloc[MIN_HISTORY:], bcash)["sharpe"] for p in VARIANTS}
    spy_sh = perf(bspy.pct_change().fillna(0).iloc[MIN_HISTORY:], bcash)["sharpe"]
    best_real = max(real.values())
    print(f"\n=== luck test: balanced panel ({bal.shape[1]} stocks, {bal_start} on), {reps} bootstrapped histories ===")
    print("real Sharpe (excess of cash):", {k: round(v, 3) for k, v in real.items()}, f"| SPY {spy_sh:.3f}")
    rng = np.random.default_rng(seed)
    null_best, null_each = [], []
    t1 = time.time()
    for i in range(reps):
        fc, fv, fs = null_panel(bal, bv, bspy, rng)
        sh = {p.name: perf(simulate(fc, build_signals(fc, fv, fs, p), p, bcash.to_numpy(), cost_bps)[0]
                           .iloc[MIN_HISTORY:], bcash)["sharpe"] for p in VARIANTS}
        null_each.append(sh)
        null_best.append(max(sh.values()))
        if i == 0:
            print(f"  one repetition takes ~{time.time() - t1:.0f}s")
    null = pd.DataFrame(null_each)
    null.to_csv(DATA_DIR / "minervini_null.csv", index=False)
    p_best = float(np.mean(np.array(null_best) >= best_real))
    print(f"best real Sharpe {best_real:.3f}; null best-of-{len(VARIANTS)} median {np.median(null_best):.3f}, "
          f"95th pct {np.percentile(null_best, 95):.3f}; p = {p_best:.3f} (bar: {0.05 / len(VARIANTS):.3f})")
    for k in real:
        print(f"  {k:20s} real {real[k]:+.3f}  null median {null[k].median():+.3f}  "
              f"p(single) {float(np.mean(null[k] >= real[k])):.3f}")
    print(f"\ntotal time {(time.time() - t0) / 60:.1f} min")


def run_filter_baseline(cost_bps: float = 10.0):
    """Does the Minervini stock picking add anything beyond the SPY 200-day market filter alone?
    Compares the five variants with the equal-weight basket and SPY held only while SPY > its 200-day SMA
    (same one-extra-bar delay, same costs, T-bill cash), and regresses each variant on the filtered basket."""
    close, volume, spy, cash = load_inputs()
    res = run_variants(close, volume, spy, cash, cost_bps=cost_bps)
    ok = (spy > spy.rolling(200).mean()).astype(float).shift(1).fillna(0)  # decide at close t, fill t+1
    pos = ok.shift(1).fillna(0)                                            # earns from bar t+2: same delay as above
    def filtered(asset_ret):
        turn = pos.diff().abs().fillna(0)
        return pos * asset_ret + (1 - pos) * cash - turn * cost_bps / 1e4
    basket = close.pct_change().mean(axis=1).fillna(0)
    spyr = spy.pct_change().fillna(0)
    allr = {"SPY buy&hold": spyr, "equal-weight basket": basket, "SPY + filter": filtered(spyr),
            "basket + filter": filtered(basket), **{k: v[0] for k, v in res.items()}}
    idx = close.index[MIN_HISTORY:]
    n = len(idx)
    segs = {"all": (0, n), "search 50%": (0, n // 2), "validation 25%": (n // 2, 3 * n // 4),
            "holdout 25%": (3 * n // 4, n)}
    pd.set_option("display.width", 200)
    print(f"exposure of the filter: {pos.loc[idx].mean():.0%} of days invested\n")
    for seg, (a, b) in segs.items():
        ix = idx[a:b]
        rows = {}
        for name, r in allr.items():
            m = perf(r.loc[ix], cash)
            row = {"CAGR": m["cagr"], "Sharpe": m["sharpe"], "MaxDD": m["max_dd"]}
            if name in res:
                al = alpha_vs(r.loc[ix], allr["basket + filter"].loc[ix], cash.loc[ix])
                row.update({"alpha/yr vs basket+filter": al["alpha_yr"], "beta": al["beta"], "alpha t": al["alpha_t"]})
            rows[name] = row
        print(f"=== {seg} ===")
        print(pd.DataFrame(rows).T.round(3).to_string(), "\n")


def run_rsp_check(cost_bps: float = 10.0):
    """Is the filtered basket's strength real or survivorship? Runs the SPY/own-200-day filter on RSP, a real
    equal-weight S&P 500 fund that includes the companies that later dropped out, next to the survivor basket."""
    import yfinance as yf

    close, _, spy, cash = load_inputs()
    d = yf.download("RSP", start="2003-01-01", auto_adjust=True, progress=False)["Close"].iloc[:, 0]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    rsp = d.reindex(close.index).ffill()
    ix = close.index[close.index >= pd.Timestamp("2004-06-01")]  # leave room for the 200-day average

    def filt(price, asset_ret):
        pos = (price > price.rolling(200).mean()).astype(float).shift(2).fillna(0)  # decide t, fill t+1, earn t+2
        return pos * asset_ret + (1 - pos) * cash - pos.diff().abs().fillna(0) * cost_bps / 1e4

    r_spy, r_rsp = spy.pct_change().fillna(0), rsp.pct_change().fillna(0)
    r_bas = close.pct_change().mean(axis=1).fillna(0)
    rows = {"SPY": r_spy, "RSP (real equal-weight fund)": r_rsp, "survivor basket": r_bas,
            "SPY + filter": filt(spy, r_spy), "RSP + filter": filt(rsp, r_rsp),
            "survivor basket + filter": filt(spy, r_bas)}
    pd.set_option("display.width", 200)
    for label, ixs in (("all", ix), ("last 25%", ix[int(len(ix) * 0.75):])):
        print(f"=== {label}: {ixs[0].date()} -> {ixs[-1].date()} ===")
        print(pd.DataFrame({n: {k: v for k, v in perf(r.loc[ixs], cash).items() if k != "total"}
                            for n, r in rows.items()}).T.round(3).to_string(), "\n")
