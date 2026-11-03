"""Risk-control study: does volatility targeting (and friends) work across asset classes, with realistic cash and leverage?

Pre-registered variants (fixed BEFORE running, 13 in total, so the luck-test bar is stricter):
  single asset: vol target {10%, 15%} x leverage cap {1.0, 1.3, 1.5} (6), 200-day trend (1),
                vol target 10%/cap 1.0 x trend (1), vol target 15%/cap 1.3 x trend (1)           = 9
  multi-asset (SPY, EFA, TLT, GLD, VNQ), inverse-volatility weights:
                unscaled, scaled to 10% vol cap 1.0, cap 1.5, cap 1.5 with per-asset trend filter = 4

Realism fixes vs the earlier backtests: uninvested money earns the T-bill rate, borrowed money (weight above 1)
pays the T-bill rate plus 0.5%/yr, trades cost 1 bp of turnover. Sharpe ratios are on returns in excess of cash.
Buy & hold holds the asset itself (no cash). Everything uses total-return (adjusted) prices.

Luck test: the same 13 variants on block-bootstrapped history (whole days resampled in 10-day blocks, jointly for
every asset and the T-bill rate), which destroys month-scale volatility clustering and any trend. A variant only
counts if its average Sharpe improvement beats what the bootstrap produces.
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import fetch_tbill_daily, get_bars

CLASSES = {"US stocks": ["SPY", "QQQ", "IWM"], "International": ["EFA", "EEM", "EWJ"],
           "Sectors": ["XLE", "XLF", "XLK", "XLV", "XLU"], "Bonds": ["TLT", "IEF", "LQD", "AGG"],
           "Real assets": ["GLD", "VNQ"]}
ASSETS = [a for v in CLASSES.values() for a in v]
RP_ASSETS = ["SPY", "EFA", "TLT", "GLD", "VNQ"]
SPREAD, COST_BPS, WARM, BLOCK = 0.005, 1.0, 252, 10


def vol_est(r, window: int = 20):
    return r.rolling(window).std() * np.sqrt(252)


def single_variants() -> dict:
    v = {}
    for tgt in (0.10, 0.15):
        for cap in (1.0, 1.3, 1.5):
            v[f"vt{int(tgt * 100)}_cap{cap:g}"] = lambda R, C, t=tgt, k=cap: (t / vol_est(R)).clip(upper=k)
    v["trend200"] = lambda R, C: (C > C.rolling(200).mean()).astype(float)
    v["vt10_cap1_trend"] = lambda R, C: (0.10 / vol_est(R)).clip(upper=1.0) * (C > C.rolling(200).mean())
    v["vt15_cap1.3_trend"] = lambda R, C: (0.15 / vol_est(R)).clip(upper=1.3) * (C > C.rolling(200).mean())
    return v


def portfolio_returns(R, W, cash, spread: float = SPREAD, cost_bps: float = COST_BPS):
    """Returns from weights `W` decided at each close and held from the next bar. Idle money earns `cash`,
    money borrowed (total weight above 1) pays `cash` + spread, turnover costs `cost_bps`."""
    pos = W.shift(1).fillna(0.0)
    total = pos.sum(axis=1) if isinstance(pos, pd.DataFrame) else pos
    gross = (pos * R).sum(axis=1) if isinstance(pos, pd.DataFrame) else pos * R
    turnover = pos.diff().abs().fillna(0.0)
    turnover = turnover.sum(axis=1) if isinstance(turnover, pd.DataFrame) else turnover
    return gross + (1 - total) * cash - (total - 1).clip(lower=0) * spread / 252 - turnover * cost_bps / 1e4


def _per_asset(R: pd.DataFrame, W: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    pos = W.shift(1).fillna(0.0)
    return (pos * R).add((1 - pos).mul(cash, axis=0)) - (pos - 1).clip(lower=0) * SPREAD / 252 \
        - pos.diff().abs().fillna(0.0) * COST_BPS / 1e4


def rp_weights(R, C, scale_to: float | None, cap: float, trend: bool, window: int = 60):
    inv = 1 / R[RP_ASSETS].rolling(window).std()
    base = inv.div(inv.sum(axis=1), axis=0)  # weights sum to 1 across the five assets
    if trend:
        base = base * (C[RP_ASSETS] > C[RP_ASSETS].rolling(200).mean())
    if scale_to is None:
        return base
    pr = (base.shift(1) * R[RP_ASSETS]).sum(axis=1)
    s = (scale_to / (pr.rolling(20).std() * np.sqrt(252))).clip(upper=cap).fillna(0.0)
    return base.mul(s, axis=0)


def compute_all(C: pd.DataFrame, cash: pd.Series) -> dict:
    R = C.pct_change().fillna(0.0)
    out = {"single": {name: _per_asset(R, fn(R, C), cash) for name, fn in single_variants().items()}, "bh": R}
    rp = {"rp_unscaled": rp_weights(R, C, None, 1.0, False), "rp_vt10_cap1": rp_weights(R, C, 0.10, 1.0, False),
          "rp_vt10_cap1.5": rp_weights(R, C, 0.10, 1.5, False), "rp_vt10_cap1.5_trend": rp_weights(R, C, 0.10, 1.5, True)}
    out["rp"] = {k: portfolio_returns(R[RP_ASSETS], w, cash) for k, w in rp.items()}
    out["rp_bench"] = {"SPY buy&hold": R["SPY"], "60/40 SPY/IEF": 0.6 * R["SPY"] + 0.4 * R["IEF"],
                       "equal-weight 5": R[RP_ASSETS].mean(axis=1)}
    return out


def _stats(ret, cash) -> pd.DataFrame:
    ret = ret.to_frame("x") if isinstance(ret, pd.Series) else ret
    ex = ret.sub(cash, axis=0)
    eq = (1 + ret).cumprod()
    return pd.DataFrame({"sharpe": ex.mean() / ex.std() * np.sqrt(252),
                         "cagr": eq.iloc[-1] ** (252 / len(ret)) - 1, "vol": ret.std() * np.sqrt(252),
                         "maxdd": (eq / eq.cummax() - 1).min()})


def _window(df, cash, which: str):
    n = len(df) - WARM
    start = WARM if which == "full" else WARM + int(n * 0.75)
    return df.iloc[start:], cash.iloc[start:]


def summarize_single(res: dict, cash: pd.Series, which: str) -> pd.DataFrame:
    bh, c = _window(res["bh"], cash, which)
    sb = _stats(bh, c)
    rows = {}
    for name, df in res["single"].items():
        d, c = _window(df, cash, which)
        s = _stats(d, c)
        rows[name] = {"d_sharpe": (s.sharpe - sb.sharpe).mean(), "sharpe_improved": (s.sharpe > sb.sharpe).mean(),
                      "smaller_maxdd": (s.maxdd > sb.maxdd).mean(), "d_cagr": (s.cagr - sb.cagr).mean(),
                      "d_maxdd": (s.maxdd - sb.maxdd).mean(),
                      "match_return_lower_risk": ((s.cagr >= sb.cagr) & (s.maxdd > sb.maxdd)).mean()}
    return pd.DataFrame(rows).T


def summarize_by_class(res: dict, cash: pd.Series, variants: list[str]) -> pd.DataFrame:
    bh, c = _window(res["bh"], cash, "full")
    sb = _stats(bh, c)
    rows = {}
    for name in variants:
        d, c = _window(res["single"][name], cash, "full")
        s = _stats(d, c)
        for cls, members in CLASSES.items():
            rows[(name, cls)] = {"assets": len(members), "d_sharpe": (s.sharpe - sb.sharpe)[members].mean(),
                                 "sharpe_improved": (s.sharpe > sb.sharpe)[members].mean(),
                                 "d_maxdd": (s.maxdd - sb.maxdd)[members].mean(),
                                 "d_cagr": (s.cagr - sb.cagr)[members].mean()}
    return pd.DataFrame(rows).T


def summarize_rp(res: dict, cash: pd.Series, which: str) -> pd.DataFrame:
    rows = {}
    for group in ("rp_bench", "rp"):
        for name, r in res[group].items():
            d, c = _window(r, cash, which)
            rows[name] = _stats(d, c).iloc[0]
    return pd.DataFrame(rows).T


def stat_vector(res: dict, cash: pd.Series) -> dict:
    """The numbers the luck test compares: average Sharpe improvement per single-asset variant, and Sharpe edge over
    the equal-weight 5-asset basket for each multi-asset variant (full window)."""
    s = summarize_single(res, cash, "full")["d_sharpe"].to_dict()
    rp = summarize_rp(res, cash, "full")["sharpe"]
    for k in res["rp"]:
        s[k] = rp[k] - rp["equal-weight 5"]
    return s


def bootstrap(C: pd.DataFrame, cash: pd.Series, rng: np.random.Generator):
    R = C.pct_change().fillna(0.0).to_numpy()
    T = len(R)
    idx: list[int] = []
    while len(idx) < T:
        s = int(rng.integers(0, T - BLOCK + 1))
        idx.extend(range(s, s + BLOCK))
    idx = np.array(idx[:T])
    fake = pd.DataFrame(100 * np.cumprod(1 + R[idx], axis=0), index=C.index, columns=C.columns)
    return fake, pd.Series(cash.to_numpy()[idx], index=cash.index)


def run_risk_study(reps: int = 500, start: str = "2003-01-01") -> None:
    from datetime import date
    end = date.today().isoformat()
    C = pd.DataFrame({a: get_bars("yahoo", a, start, end)["close"] for a in sorted(set(ASSETS + ["IEF"]))}).dropna()
    cash = fetch_tbill_daily(start, end).reindex(C.index).ffill().fillna(0.0)
    print(f"{C.shape[1]} ETFs, {C.index[0].date()} .. {C.index[-1].date()} ({len(C)} bars). "
          f"Average T-bill yield {cash.mean() * 252:.2%}.\n13 variants pre-registered; luck-test bar: p < 0.004 "
          "(0.05 / 13).\n")
    res = compute_all(C, cash)
    full, hold = summarize_single(res, cash, "full"), summarize_single(res, cash, "holdout")
    pd.set_option("display.width", 200)
    print("SINGLE-ASSET VARIANTS vs buy & hold, averaged over 17 ETFs (excess-return Sharpe, cash earns T-bill rate)")
    print("All months:\n" + full.round(3).to_string())
    print("\nLast 25% of months:\n" + hold.round(3).to_string())
    key = ["vt10_cap1", "vt10_cap1.5", "vt15_cap1.5", "trend200", "vt10_cap1_trend"]
    print("\nBY ASSET CLASS (all months):\n" + summarize_by_class(res, cash, key).round(3).to_string())
    print("\nMULTI-ASSET (SPY, EFA, TLT, GLD, VNQ), all months:\n" + summarize_rp(res, cash, "full").round(3).to_string())
    print("\nMULTI-ASSET, last 25% of months:\n" + summarize_rp(res, cash, "holdout").round(3).to_string())

    real = stat_vector(res, cash)
    null_rows = []
    rng = np.random.default_rng(7)
    for i in range(reps):
        fakeC, fakecash = bootstrap(C, cash, rng)
        null_rows.append(stat_vector(compute_all(fakeC, fakecash), fakecash))
        if (i + 1) % 100 == 0:
            print(f"  luck-test rep {i + 1}/{reps}", flush=True)
    null = pd.DataFrame(null_rows)
    null.to_csv(DATA_DIR / "riskstudy_null.csv", index=False)
    full.to_csv(DATA_DIR / "riskstudy_single_full.csv")
    hold.to_csv(DATA_DIR / "riskstudy_single_holdout.csv")
    summarize_rp(res, cash, "full").to_csv(DATA_DIR / "riskstudy_rp_full.csv")
    out = pd.DataFrame({"real": pd.Series(real), "null_median": null.median(), "null_95th": null.quantile(0.95),
                        "p_value": pd.Series({k: ((null[k] >= v).sum() + 1) / (reps + 1) for k, v in real.items()})})
    print(f"\nLUCK TEST ({reps} bootstrap repetitions). Statistic: average Sharpe improvement vs buy & hold "
          "(single-asset) or vs the equal-weight 5 basket (rp_*):")
    print(out.round(3).to_string())
    print("\nA variant counts only if p < 0.004. Fully bootstrapped data has no volatility clustering or trend, so "
          "a variant that only works through those effects should fail here.")
