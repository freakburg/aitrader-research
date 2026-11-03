"""ETF momentum rotation: each month-end rank a fixed universe of ETFs by trailing return, hold the top K
equal-weighted from the next day, and park the unfilled slots in cash (SHY). With `abs_filter`, an ETF
must also have a positive trailing return ("dual momentum").

The universe is index/sector/country/bond/gold ETFs, so there is almost no survivorship bias (unlike
picking single stocks that happen to still be listed). Search -> validation -> holdout and the luck test
are shared with sweep.py.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import get_bars
from .sweep import _dir, _key, bootstrap_closes, report, run_pipeline, summarize

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
            "EFA", "EEM", "EWJ", "TLT", "IEF", "LQD", "AGG", "GLD", "VNQ", "SHY"]
CASH = "SHY"
WARMUP = 300  # bars before the first scored day: a 12-month lookback needs ~273


def variations() -> list[dict]:
    return [{"lookback_m": L, "top_k": k, "skip_m": s, "abs_filter": a}
            for L, k, s, a in itertools.product([1, 3, 6, 9, 12], [1, 2, 3, 5, 8], [0, 1], [True, False])
            if s < L]


def rotation_returns(closes: pd.DataFrame, lookback_m: int = 12, top_k: int = 3, skip_m: int = 1,
                     abs_filter: bool = True, cost_bps: float = 5.0, cash: str = CASH) -> pd.Series:
    """Daily returns of the rotation. Weights chosen at a month-end close are held from the next bar."""
    cols = list(closes.columns)
    risk = [c for c in cols if c != cash]
    cash_i = cols.index(cash)
    n, m = len(closes), len(cols)

    idx = closes.index
    reb = np.flatnonzero(np.r_[idx.month[1:] != idx.month[:-1], True])  # last trading day of each month
    px = closes[risk]
    score = (px.shift(skip_m * 21) / px.shift(lookback_m * 21) - 1).to_numpy()[reb]

    W = np.full((n, m), np.nan)
    for j, i in enumerate(reb):
        s = score[j]
        valid = ~np.isnan(s)
        if not valid.any():
            continue
        order = np.argsort(np.where(valid, -s, np.inf))[:top_k]
        chosen = [o for o in order if valid[o] and (not abs_filter or s[o] > 0)]
        w = np.zeros(m)
        for o in chosen:
            w[cols.index(risk[o])] = 1.0 / top_k
        w[cash_i] = 1.0 - len(chosen) / top_k
        W[i] = w
    pos = pd.DataFrame(W, index=idx).ffill().fillna(0.0).shift(1).fillna(0.0).to_numpy()

    R = closes.pct_change().fillna(0.0).to_numpy()
    turnover = np.abs(np.diff(pos, axis=0, prepend=0.0)).sum(axis=1)
    return pd.Series((pos * R).sum(axis=1) - turnover * cost_bps / 1e4, index=idx)


def perf(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1, "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            "max_dd": (eq / eq.cummax() - 1).min(), "total_return": eq.iloc[-1] - 1}


def benchmarks(closes: pd.DataFrame) -> dict[str, pd.Series]:
    R = closes.pct_change().fillna(0.0)
    risk = [c for c in closes.columns if c != CASH]
    return {"SPY buy&hold": R["SPY"], "equal-weight all ETFs": R[risk].mean(axis=1),
            "60/40 SPY/IEF": 0.6 * R["SPY"] + 0.4 * R["IEF"]}


def load_universe(source: str, start: str, end: str) -> pd.DataFrame:
    got, missing = {}, []
    for s in UNIVERSE:
        try:
            got[s] = get_bars(source, s, start, end)["close"]
        except Exception:
            missing.append(s)
    if missing:
        print("Skipped (no data):", missing)
    return pd.DataFrame(got).dropna()  # common window: starts when the youngest ETF began trading


def run_rotation(source: str = "yahoo", start: str = "2003-01-01", reps: int = 200, cost_bps: float = 5.0,
                 fresh: bool = False) -> None:
    from datetime import date
    closes = load_universe(source, start, date.today().isoformat())
    print(f"{closes.shape[1]} ETFs, {closes.index[0].date()} .. {closes.index[-1].date()} ({len(closes)} bars)")
    d = _dir("rotation")
    d.mkdir(exist_ok=True)
    real_csv, real_json, finals_csv, null_csv = (d / n for n in
                                                 ("real_variations.csv", "real_summary.json", "real_finalists.csv", "null_reps.csv"))
    if fresh:
        for p in (real_csv, real_json, finals_csv, null_csv):
            p.unlink(missing_ok=True)
    grid = variations()

    def all_returns(c: pd.DataFrame) -> dict:
        return {_key("rot", p): rotation_returns(c, cost_bps=cost_bps, **p).to_frame("SPY") for p in grid}  # column named after the benchmark it is compared with

    t0 = time.time()
    if not real_json.exists():
        table, finalists, ref = run_pipeline(closes[["SPY"]], all_returns(closes), WARMUP)
        table.to_csv(real_csv, index=False)
        finalists.to_csv(finals_csv, index=False)
        real_json.write_text(json.dumps({"ref": ref, "summary": summarize(table, finalists, ref),
                                         "pass_seconds": time.time() - t0}, default=float))
        print(f"Real pass: {len(grid)} variations in {time.time() - t0:.0f}s")

    done = len(pd.read_csv(null_csv)) if null_csv.exists() else 0
    for rep in range(done, reps):
        fake = bootstrap_closes(closes, np.random.default_rng(1000 + rep))
        table, finalists, ref = run_pipeline(fake[["SPY"]], all_returns(fake), WARMUP)
        pd.DataFrame([{"rep": rep, **summarize(table, finalists, ref)}]).to_csv(
            null_csv, mode="a", header=not null_csv.exists(), index=False)
        if (rep + 1) % 25 == 0:
            print(f"  luck-test rep {rep + 1}/{reps}", flush=True)

    print("\n" + report("rotation"))
    print("\nFull-period performance (net of 5 bps/trade costs), whole window:")
    rows = {name: perf(r.iloc[WARMUP:]) for name, r in benchmarks(closes).items()}
    finals = pd.read_csv(finals_csv)
    for k in finals.variation:
        p = json.loads(k.split("|", 1)[1])
        rows[f"rot {p['lookback_m']}m skip{p['skip_m']} top{p['top_k']}{' dual' if p['abs_filter'] else ''}"] = perf(
            rotation_returns(closes, cost_bps=cost_bps, **p).iloc[WARMUP:])
    print(pd.DataFrame(rows).T.round(3).to_string())
