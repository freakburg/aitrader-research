"""Mass strategy search with built-in protection against fooling ourselves.

1. Test hundreds of variations (rules + ML configs) across several tickers.
2. Pick finalists using only earlier data: top by SEARCH window, then best on VALIDATION window.
   The HOLDOUT window is looked at once, for the finalists only.
3. Luck test: repeat the identical search on block-bootstrapped (pattern-free) versions of the same prices.
   If the real "best" is no better than the best you get by chance, it is not evidence of anything.

Results are saved as they arrive (data/sweep/), so the run can be stopped and resumed, and
`python main.py sweep-report` summarizes whatever has finished.
"""
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .backtest import position_returns
from .config import DATA_DIR, keep_awake
from .ml_strategy import MLStrategy
from .strategy import Breakout, RSIReversion, SMACrossover, TrendFilter, VolTarget

SWEEP_DIR = DATA_DIR / "sweep"
CHEAP = {"sma": SMACrossover, "trend": TrendFilter, "vol": VolTarget, "rsi": RSIReversion, "breakout": Breakout}
THRESHOLDS = [0.50, 0.52, 0.55, 0.58]
WARMUP = 500  # every variation is scored from the same bar, so all are comparable (ML needs 500 bars to train)
SEARCH_FRAC, VAL_FRAC = 0.50, 0.25  # remaining 25% is the untouched holdout
TOP_SEARCH, N_FINAL = 15, 5
BLOCK = 10  # bootstrap block length in bars


def _dir(tag: str):
    return DATA_DIR / (f"sweep_{tag}" if tag else "sweep")


def _key(name: str, params: dict) -> str:
    return f"{name}|{json.dumps(params, sort_keys=True)}"


def cheap_variations() -> list[tuple[str, dict]]:
    v = [("sma", {"fast": f, "slow": s}) for f, s in
         itertools.product([5, 10, 15, 20, 30, 40], [30, 50, 75, 100, 150, 200, 250]) if f < s]
    v += [("trend", {"window": w}) for w in [20, 50, 75, 100, 150, 200, 250]]
    v += [("vol", {"target_vol": t, "window": w}) for t, w in itertools.product([.06, .08, .10, .12, .15], [10, 20, 40, 60])]
    v += [("rsi", {"period": p, "lo": lo, "hi": hi}) for p, lo, hi in
          itertools.product([2, 3, 5, 14], [10, 20, 30], [50, 60, 70])]
    v += [("breakout", {"n": n, "m": m}) for n, m in itertools.product([20, 50, 100], [10, 20, 50])]
    return v


def ml_groups(quick: bool = False) -> list[dict]:
    """ML configs that differ in anything except the threshold (thresholds are applied to cached probabilities)."""
    models = [{"model": "logreg", "C": c} for c in (0.01, 0.1, 1.0)]
    models += [{"model": m, "depth": d} for m in ("rf", "hgb") for d in (2, 3)]
    if quick:
        models = models[1:2] + models[3:4]
    groups = [{**m, "horizon": h, "retrain_every": r, "window": w, "min_train": WARMUP}
              for m, h, r, w in itertools.product(models, (1, 5), (21, 63), ("expanding", "rolling"))]
    return groups[:3] if quick else groups


def _run_task(args):
    kind, spec, ticker, close, cost_bps = args
    df = pd.DataFrame({"close": close})
    out = {}
    if kind == "cheap":
        for name, params in spec:
            out[_key(name, params)] = position_returns(df["close"], CHEAP[name](**params).signals(df),
                                                       cost_bps).to_numpy()
    else:
        proba = MLStrategy(**spec).proba(df)
        for th in THRESHOLDS:
            sig = (proba > th).astype(float)
            out[_key("ml", {**spec, "threshold": th})] = position_returns(df["close"], sig, cost_bps).to_numpy()
    return ticker, out


def compute_returns(closes: pd.DataFrame, pool, cost_bps: float, quick: bool = False, log=None) -> dict:
    cheap = cheap_variations()
    futures = []
    for tkr in closes.columns:
        futures.append(pool.submit(_run_task, ("cheap", cheap, tkr, closes[tkr], cost_bps)))
        for g in ml_groups(quick):
            futures.append(pool.submit(_run_task, ("ml", g, tkr, closes[tkr], cost_bps)))
    cols: dict[str, dict] = {}
    for i, fut in enumerate(futures, 1):
        tkr, out = fut.result()
        for key, arr in out.items():
            cols.setdefault(key, {})[tkr] = arr
        if log and (i % max(1, len(futures) // 10) == 0):
            log(f"    {i}/{len(futures)} tasks done")
    return {k: pd.DataFrame(v, index=closes.index) for k, v in cols.items()}


def _sharpe(df: pd.DataFrame) -> pd.Series:
    sd = df.std().replace(0, np.nan)
    return (df.mean() / sd * np.sqrt(252)).fillna(0.0)


def _total_return(df: pd.DataFrame) -> pd.Series:
    return (1 + df).prod() - 1


def run_pipeline(closes: pd.DataFrame, returns: dict[str, pd.DataFrame], warmup: int = WARMUP):
    """Search -> validation -> holdout. Scores are mean Sharpe across tickers.
    `closes` supplies the buy & hold benchmark columns; `warmup` is the first scored bar."""
    rets = {k: v.iloc[warmup:] for k, v in returns.items()}
    bh = closes.pct_change().fillna(0).iloc[warmup:]
    n = len(bh)
    a, b = int(n * SEARCH_FRAC), int(n * (SEARCH_FRAC + VAL_FRAC))
    bh_search, bh_val = _sharpe(bh.iloc[:a]), _sharpe(bh.iloc[a:b])
    rows = []
    for k, df in rets.items():
        s_search = _sharpe(df.iloc[:a])
        rows.append({"variation": k, "search": s_search.mean(), "val": _sharpe(df.iloc[a:b]).mean(),
                     "search_beats_bh": (s_search > bh_search).mean()})
    table = pd.DataFrame(rows)
    finalists = table.nlargest(TOP_SEARCH, "search").nlargest(N_FINAL, "val").copy()
    finalists["holdout"] = [_sharpe(rets[k].iloc[b:]).mean() for k in finalists.variation]
    finalists["holdout_return"] = [_total_return(rets[k].iloc[b:]).mean() for k in finalists.variation]
    ref = {"bh_search": bh_search.mean(), "bh_val": bh_val.mean(), "bh_holdout": _sharpe(bh.iloc[b:]).mean(),
           "bh_holdout_return": _total_return(bh.iloc[b:]).mean(), "variations": len(table),
           "tickers": closes.shape[1], "bars_search": a, "bars_val": b - a, "bars_holdout": n - b}
    return table, finalists, ref


def summarize(table: pd.DataFrame, finalists: pd.DataFrame, ref: dict) -> dict:
    pick = finalists.iloc[0]
    return {"best_search": table["search"].max(), "n_beating_bh_search": int((table["search"] > ref["bh_search"]).sum()),
            "pick_search": pick["search"], "pick_val": pick["val"], "pick_holdout": pick["holdout"],
            "bh_holdout": ref["bh_holdout"], "pick_excess_holdout": pick["holdout"] - ref["bh_holdout"]}


def bootstrap_closes(closes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole days in blocks (same days for every ticker, so correlations survive) -> no real patterns."""
    R = closes.pct_change().fillna(0).to_numpy()
    T = len(R)
    idx: list[int] = []
    while len(idx) < T:
        s = int(rng.integers(0, T - BLOCK + 1))
        idx.extend(range(s, s + BLOCK))
    return pd.DataFrame(100 * np.cumprod(1 + R[np.array(idx[:T])], axis=0), index=closes.index,
                        columns=closes.columns)


def run_sweep(closes: pd.DataFrame, max_hours: float = 8.0, null_reps: int = 100, workers: int | None = None,
              cost_bps: float = 1.0, quick: bool = False, fresh: bool = False, tag: str = ""):
    os.environ["OMP_NUM_THREADS"] = "1"  # one thread per worker process; parallelism comes from the pool
    keep_awake()
    d = _dir(tag)
    d.mkdir(exist_ok=True)
    real_csv, real_json, finals_csv, null_csv = (d / n for n in
                                                 ("real_variations.csv", "real_summary.json", "real_finalists.csv", "null_reps.csv"))
    if fresh:
        for p in (real_csv, real_json, finals_csv, null_csv):
            p.unlink(missing_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()

    def log(msg):
        print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)

    log(f"workers={workers} tickers={list(closes.columns)} bars={len(closes)}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if not real_json.exists():
            log("real data: computing all variations")
            t1 = time.time()
            table, finalists, ref = run_pipeline(closes, compute_returns(closes, pool, cost_bps, quick, log))
            table.to_csv(real_csv, index=False)
            finalists.to_csv(finals_csv, index=False)
            real_json.write_text(json.dumps({"ref": ref, "summary": summarize(table, finalists, ref),
                                             "pass_seconds": time.time() - t1}, default=float))
            log(f"real pass done in {(time.time() - t1) / 60:.1f} min")
        pass_secs = json.loads(real_json.read_text())["pass_seconds"]

        done = len(pd.read_csv(null_csv)) if null_csv.exists() else 0
        for rep in range(done, null_reps):
            if time.time() - t0 + pass_secs * 1.15 > max_hours * 3600:
                log(f"stopping: not enough time left for another luck-test repetition ({rep} done)")
                break
            fake = bootstrap_closes(closes, np.random.default_rng(1000 + rep))
            table, finalists, ref = run_pipeline(fake, compute_returns(fake, pool, cost_bps, quick))
            row = pd.DataFrame([{"rep": rep, **summarize(table, finalists, ref)}])
            row.to_csv(null_csv, mode="a", header=not null_csv.exists(), index=False)
            log(f"luck-test rep {rep + 1}/{null_reps} done (best search Sharpe on fake data: "
                f"{row['best_search'].iloc[0]:.2f})")
    print(report(tag))


def report(tag: str = "") -> str:
    real_json, finals_csv, null_csv = (_dir(tag) / n for n in ("real_summary.json", "real_finalists.csv", "null_reps.csv"))
    if not real_json.exists():
        return "No sweep results yet. Run: python main.py sweep"
    data = json.loads(real_json.read_text())
    ref, s = data["ref"], data["summary"]
    finals = pd.read_csv(finals_csv)
    out = [f"{ref['variations']} variations x {ref['tickers']} tickers. Windows (bars): search {ref['bars_search']}, "
           f"validation {ref['bars_val']}, holdout {ref['bars_holdout']}. Scores = mean Sharpe across tickers.", "",
           f"Buy & hold Sharpe: search {ref['bh_search']:.2f} | validation {ref['bh_val']:.2f} | "
           f"holdout {ref['bh_holdout']:.2f} (return {ref['bh_holdout_return']:.1%})", "",
           "Finalists (picked on search+validation; holdout seen once):",
           finals.assign(variation=finals.variation.str.slice(0, 90)).round(3).to_string(index=False), ""]
    out.append(f"Best search Sharpe of ALL variations: {s['best_search']:.2f}   "
               f"({s['n_beating_bh_search']} of {ref['variations']} beat buy & hold in the search window)")
    if null_csv.exists():
        null = pd.read_csv(null_csv)
        k = len(null)
        p_best = ((null.best_search >= s["best_search"]).sum() + 1) / (k + 1)
        p_hold = ((null.pick_excess_holdout >= s["pick_excess_holdout"]).sum() + 1) / (k + 1)
        out += ["", f"LUCK TEST ({k} repetitions on pattern-free fake prices, same search):",
                f"  best search Sharpe on fake data: median {null.best_search.median():.2f}, "
                f"95th pct {null.best_search.quantile(.95):.2f}   (real: {s['best_search']:.2f})",
                f"  chance of a best-search score this high from luck alone: p = {p_best:.3f}",
                f"  holdout Sharpe vs buy & hold for the pick: real {s['pick_excess_holdout']:+.2f}, "
                f"fake median {null.pick_excess_holdout.median():+.2f}, 95th pct {null.pick_excess_holdout.quantile(.95):+.2f}"
                f"; chance from luck: p = {p_hold:.3f}", "",
                "Reading it: p above ~0.05 means the result is not distinguishable from luck."]
    else:
        out += ["", "No luck-test repetitions yet; without them the result above cannot be trusted."]
    return "\n".join(out)
