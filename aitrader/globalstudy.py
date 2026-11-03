"""Does volatility targeting (the forward-test rule) hold up in other eras, countries and asset types?

Three groups, fixed in advance (19 markets, plus the S&P 500 split by era):
  Long US history   S&P 500 price index 1960-2026 (eras: 1962-79, 1980-99, 2000-09, 2010-26)
  Other countries   Nikkei 225, FTSE 100, DAX, CAC 40, Hang Seng, ASX 200, TSX, Bovespa (from 1995), KOSPI
  Other assets      gold (GLD), silver (SLV), broad commodities (DBC), oil (USO), US dollar (UUP), euro (FXE), yen (FXY),
                    Bitcoin, Ether (crypto trades 365 days a year, so it is annualized with 365)
Strategies (2): the frozen rule vt10_cap1 (10% target volatility, 20-day estimate, never above 100% invested) and the
200-day trend rule as a comparison. Cash earns the US T-bill rate (a proxy for every market), trades cost 1 bp.
Stock indexes are price indexes (no dividends), which understates returns but compares the strategy and buy & hold fairly.

For every market: Sharpe on returns above cash, strategy vs buy & hold, with (a) a paired block-bootstrap 95% interval for
the Sharpe change (sampling uncertainty) and (b) a pattern-free bootstrap p-value (how often the same strategy on
resampled 10-day blocks, with no real volatility clustering or trend, gains as much).
Support for "volatility targeting helps" requires: at least two thirds of ALL markets improve, and the cross-market
bootstrap interval for the average Sharpe change excludes zero. Counting markets with p < 0.05 is descriptive only: about
1 in 20 pass by chance.
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import fetch_tbill_daily, get_bars
from .riskstudy import portfolio_returns

GROUPS = {
    "Long US history": ["^GSPC"],
    "Other countries": ["^N225", "^FTSE", "^GDAXI", "^FCHI", "^HSI", "^AXJO", "^GSPTSE", "^BVSP", "^KS11"],
    "Other assets": ["GLD", "SLV", "DBC", "USO", "UUP", "FXE", "FXY", "BTC-USD", "ETH-USD"],
}
NAMES = {"^GSPC": "S&P 500 (1960+)", "^N225": "Japan (Nikkei)", "^FTSE": "UK (FTSE 100)", "^GDAXI": "Germany (DAX)",
         "^FCHI": "France (CAC 40)", "^HSI": "Hong Kong", "^AXJO": "Australia", "^GSPTSE": "Canada", "^BVSP": "Brazil",
         "^KS11": "South Korea", "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities", "USO": "Oil", "UUP": "US dollar",
         "FXE": "Euro", "FXY": "Yen", "BTC-USD": "Bitcoin", "ETH-USD": "Ether"}
START = {"^BVSP": "1995-01-01"}
CRYPTO = {"BTC-USD", "ETH-USD"}
ERAS = {"1962-79": ("1962-01-01", "1979-12-31"), "1980-99": ("1980-01-01", "1999-12-31"),
        "2000-09": ("2000-01-01", "2009-12-31"), "2010-26": ("2010-01-01", "2026-12-31")}
KINDS = ["vt10_cap1", "trend200"]
WARM, BLOCK_NULL, BLOCK_CI = 252, 10, 20


def weights(R: pd.Series, C: pd.Series, kind: str, ppy: int) -> pd.Series:
    if kind == "vt10_cap1":
        return (0.10 / (R.rolling(20).std() * np.sqrt(ppy))).clip(upper=1.0).fillna(0.0)
    return (C > C.rolling(200).mean()).astype(float)


def stats(ret: pd.Series, cash: pd.Series, ppy: int) -> dict:
    ex = ret - cash
    eq = (1 + ret).cumprod()
    return {"sharpe": ex.mean() / ex.std() * np.sqrt(ppy), "cagr": eq.iloc[-1] ** (ppy / len(ret)) - 1,
            "maxdd": (eq / eq.cummax() - 1).min()}


def clean(close: pd.Series, ppy: int, name: str) -> pd.Series:
    """Drop obviously bad prints (a daily move beyond 40%, 60% for crypto) that would be data errors."""
    r = close.pct_change()
    bad = r.abs() > (0.6 if ppy == 365 else 0.4)
    if bad.any():
        print(f"  note: {name}: dropped {int(bad.sum())} implausible daily moves {r[bad].round(2).to_dict()}")
        close = close[~bad]
    return close


def blocks(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    idx: list[int] = []
    while len(idx) < T:
        s = int(rng.integers(0, T - block + 1))
        idx.extend(range(s, s + block))
    return np.array(idx[:T])


def run_market(close: pd.Series, cash_daily: pd.Series, ppy: int, reps_null: int, reps_ci: int, rng) -> dict:
    cash = (cash_daily * (252 / ppy)).reindex(close.index).ffill().fillna(0.0)
    R = close.pct_change().fillna(0.0)
    res = {"R": R, "cash": cash, "close": close}
    for kind in KINDS:
        res[kind] = portfolio_returns(R, weights(R, close, kind, ppy), cash)
    T = len(R) - WARM

    def sharpe(x):
        return x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(ppy)

    out = {}
    for kind in KINDS:
        ex_s = (res[kind] - cash).to_numpy()[WARM:]
        ex_b = (R - cash).to_numpy()[WARM:]
        real = sharpe(ex_s) - sharpe(ex_b)
        boot = np.array([sharpe(ex_s[i]) - sharpe(ex_b[i]) for i in (blocks(T, BLOCK_CI, rng) for _ in range(reps_ci))])
        null = []
        r_np, c_np = R.to_numpy()[WARM:], cash.to_numpy()[WARM:]
        for _ in range(reps_null):
            i = blocks(len(r_np), BLOCK_NULL, rng)
            fake = pd.Series(100 * np.cumprod(1 + r_np[i]))
            fr, fc = fake.pct_change().fillna(0.0), pd.Series(c_np[i])
            fret = portfolio_returns(fr, weights(fr, fake, kind, ppy), fc)
            null.append(sharpe((fret - fc).to_numpy()[WARM:]) - sharpe((fr - fc).to_numpy()[WARM:]))
        out[kind] = {"d_sharpe": real, "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "p_null": (np.sum(np.array(null) >= real) + 1) / (reps_null + 1)}
    res["inference"] = out
    return res


def window_stats(res: dict, lo: int, hi: int | None, ppy: int) -> dict:
    sl = slice(lo, hi)
    cash, base = res["cash"].iloc[sl], stats(res["R"].iloc[sl], res["cash"].iloc[sl], ppy)
    out = {"bh_sharpe": base["sharpe"], "bh_cagr": base["cagr"], "bh_maxdd": base["maxdd"]}
    for k in KINDS:
        s = stats(res[k].iloc[sl], cash, ppy)
        out[k] = {"sharpe": s["sharpe"], "d_sharpe": s["sharpe"] - base["sharpe"], "cagr": s["cagr"],
                  "d_cagr": s["cagr"] - base["cagr"], "maxdd": s["maxdd"], "d_maxdd": s["maxdd"] - base["maxdd"]}
    return out


def run_global_study(reps_null: int = 300, reps_ci: int = 1000, seed: int = 21) -> None:
    from datetime import date
    end = date.today().isoformat()
    cash_daily = fetch_tbill_daily("1960-01-01", end)
    rng = np.random.default_rng(seed)
    rows, results = [], {}
    for group, syms in GROUPS.items():
        for sym in syms:
            ppy = 365 if sym in CRYPTO else 252
            try:
                close = get_bars("yahoo", sym, START.get(sym, "1960-01-01"), end)["close"]
            except Exception as exc:
                print(f"  could not load {sym}: {exc}")
                continue
            close = clean(close, ppy, sym)
            res = run_market(close, cash_daily, ppy, reps_null, reps_ci, rng)
            results[sym] = res
            n = len(close)
            full, hold = window_stats(res, WARM, None, ppy), window_stats(res, WARM + int((n - WARM) * 0.75), None, ppy)
            for kind in KINDS:
                inf = res["inference"][kind]
                rows.append({"group": group, "market": NAMES[sym], "kind": kind, "start": close.index[0].date(),
                             "years": round((n - WARM) / ppy, 1), "bh_sharpe": full["bh_sharpe"],
                             "sharpe": full[kind]["sharpe"], "d_sharpe": inf["d_sharpe"], "ci_lo": inf["ci_lo"],
                             "ci_hi": inf["ci_hi"], "p_null": inf["p_null"], "bh_maxdd": full["bh_maxdd"],
                             "maxdd": full[kind]["maxdd"], "d_cagr": full[kind]["d_cagr"],
                             "hold_d_sharpe": hold[kind]["d_sharpe"], "hold_d_maxdd": hold[kind]["d_maxdd"]})
            print(f"  {NAMES[sym]} done ({close.index[0].date()}..{close.index[-1].date()}, {n} days)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "globalstudy_markets.csv", index=False)
    pd.set_option("display.width", 220)

    for kind in KINDS:
        d = df[df.kind == kind]
        print(f"\n=== {kind}: per market (Sharpe above cash; d_sharpe = strategy minus buy & hold; CI = paired bootstrap 95%; "
              f"p_null = pattern-free bootstrap) ===")
        print(d.drop(columns=["kind"]).round(3).to_string(index=False))
        g = d.groupby("group").agg(markets=("market", "count"), improved=("d_sharpe", lambda s: int((s > 0).sum())),
                                   mean_d_sharpe=("d_sharpe", "mean"), mean_d_cagr=("d_cagr", "mean"),
                                   mean_maxdd_bh=("bh_maxdd", "mean"), mean_maxdd_strategy=("maxdd", "mean"),
                                   p_below_05=("p_null", lambda s: int((s < 0.05).sum())),
                                   hold_improved=("hold_d_sharpe", lambda s: int((s > 0).sum())),
                                   hold_mean_d_sharpe=("hold_d_sharpe", "mean"))
        print("\nBy group:\n" + g.round(3).to_string())
        boots = np.array([d.d_sharpe.sample(len(d), replace=True, random_state=int(i)).mean() for i in range(2000)])
        print(f"\nAll {len(d)} markets: {int((d.d_sharpe > 0).sum())} improved ({(d.d_sharpe > 0).mean():.0%}); mean Sharpe change "
              f"{d.d_sharpe.mean():+.3f}, cross-market 95% interval [{np.percentile(boots, 2.5):+.3f}, "
              f"{np.percentile(boots, 97.5):+.3f}]; markets with p_null < 0.05: {int((d.p_null < 0.05).sum())} "
              f"(about {0.05 * len(d):.1f} expected by chance).")

    spy = results["^GSPC"]
    print("\n=== S&P 500 BY ERA (frozen rule vt10_cap1 vs buy & hold) ===")
    era_rows = []
    close = spy["close"]
    for era, (a, b) in ERAS.items():
        lo = close.index.searchsorted(pd.Timestamp(a))
        hi = close.index.searchsorted(pd.Timestamp(b), side="right")
        if hi - lo < 300:
            continue
        w = window_stats(spy, max(lo, WARM), hi, 252)
        era_rows.append({"era": era, "days": hi - lo, "bh_sharpe": w["bh_sharpe"], "vt_sharpe": w["vt10_cap1"]["sharpe"],
                         "d_sharpe": w["vt10_cap1"]["d_sharpe"], "bh_cagr": w["bh_cagr"], "d_cagr": w["vt10_cap1"]["d_cagr"],
                         "bh_maxdd": w["bh_maxdd"], "vt_maxdd": w["vt10_cap1"]["maxdd"],
                         "trend_d_sharpe": w["trend200"]["d_sharpe"], "trend_maxdd": w["trend200"]["maxdd"]})
    eras = pd.DataFrame(era_rows)
    eras.to_csv(DATA_DIR / "globalstudy_eras.csv", index=False)
    print(eras.round(3).to_string(index=False))
    print("\nReading it: the frozen rule is supported only if most markets improve AND the cross-market interval excludes zero.")
