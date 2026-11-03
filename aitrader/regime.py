"""Regime-switching overlay on buy and hold: can we tell calm "bull" markets from stressed ones, and does stepping
aside during stress beat simply holding?

A 2-state Markov-switching model (each state has its own mean and variance) is refit on past data every 63 days;
between refits the state probability is filtered forward with fixed parameters, so each day's probability of being in
the stress (high-variance) state only uses data up to that day. Index funds with 27-33 years of history are used
(SPY, QQQ, IWM, DIA, EFA, EEM, EWJ), which covers 1998-2003, 2008 and 2020.

Pre-registered variants (3, so the luck-test bar is p < 0.017):
  hmm_hard        hold the fund while P(stress) < 0.5, else T-bill cash
  hmm_soft        hold a fraction 1 - P(stress)
  hmm_hard_trend  hold only if P(stress) < 0.5 AND price is above its 200-day average
Baselines: buy & hold and the 200-day trend rule. Luck test: circularly shift the position series by a random amount
(keeps the exposure and the number/length of switches, destroys the alignment with returns) and ask how often random
timing matches the real Sharpe. Cash earns the T-bill rate; trades cost 1 bp.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import DATA_DIR, keep_awake
from .data import fetch_tbill_daily, get_bars
from .riskstudy import _stats, portfolio_returns

ASSETS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ"]
WARM, REFIT = 504, 63
VARIANTS = ["hmm_hard", "hmm_soft", "hmm_hard_trend"]


def hmm_filtered(r: pd.Series, refit_every: int = REFIT, warm: int = WARM) -> pd.DataFrame:
    """Walk-forward filtered probability of the stress (high-variance) state, plus the model's expected duration (in
    days) of the calm and stress states at that time."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    x = (r * 100).to_numpy()
    n = len(x)
    out = np.full((n, 3), np.nan)  # p_stress, calm_days, stress_days
    params, stress, durs = None, None, None
    t = warm
    while t < n:
        end = min(t + refit_every, n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = MarkovRegression(x[:t], k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
                names = res.model.param_names
                var = [res.params[i] for i, nm in enumerate(names) if nm.startswith("sigma2")]
                if np.all(np.isfinite(res.params)) and np.all(np.isfinite(res.expected_durations)):
                    params, stress, durs = res.params, int(np.argmax(var)), res.expected_durations
        except Exception:
            pass  # keep the previous parameters
        if params is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probs = MarkovRegression(x[:end], k_regimes=2, trend="c", switching_variance=True).filter(params) \
                    .filtered_marginal_probabilities
            out[t:end, 0] = np.asarray(probs)[t:end, stress]
            out[t:end, 1], out[t:end, 2] = durs[1 - stress], durs[stress]
        t = end
    return pd.DataFrame(out, index=r.index, columns=["p_stress", "calm_days", "stress_days"])


def _task(args):
    asset, r = args
    return asset, hmm_filtered(r)


def positions(p_stress: pd.Series, close: pd.Series) -> dict:
    trend = (close > close.rolling(200).mean()).astype(float)
    hard = (p_stress < 0.5).astype(float)
    return {"hmm_hard": hard, "hmm_soft": (1 - p_stress).fillna(0.0), "hmm_hard_trend": hard * trend,
            "trend200": trend}


def shift_test(w: pd.Series, R: pd.Series, cash: pd.Series, lo: int, k: int = 1000, seed: int = 3) -> tuple[float, float]:
    """Sharpe of the real position series vs random circular shifts of it (same exposure and switching)."""
    a = w.shift(1).fillna(0.0).to_numpy()[lo:]
    r, c = R.to_numpy()[lo:], cash.to_numpy()[lo:]
    T = len(a)

    def sharpe(A):
        ret = A * r + (1 - A) * c - np.abs(np.diff(A, axis=-1, prepend=A[..., :1])) * 1e-4
        ex = ret - c
        return ex.mean(axis=-1) / ex.std(axis=-1) * np.sqrt(252)

    real = float(sharpe(a))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(252, T - 252, size=k)
    null = sharpe(a[(np.arange(T)[None, :] - shifts[:, None]) % T])
    return real, float(((null >= real).sum() + 1) / (k + 1))


def drawdown_episodes(close: pd.Series, p_stress: pd.Series, thresh: float = 0.10) -> pd.DataFrame:
    peak = close.cummax()
    dd = close / peak - 1
    grp = peak.ne(peak.shift()).cumsum()
    rows = []
    for _, idx in close.groupby(grp).groups.items():
        seg = dd.loc[idx]
        if seg.min() > -thresh:
            continue
        trough = seg.idxmin()
        pk = idx[0]
        nxt = close.index.get_loc(idx[-1]) + 1
        recovered = close.index[nxt] if nxt < len(close) else pd.NaT
        flagged = p_stress.loc[pk:trough]
        flag = flagged[flagged > 0.5].index.min() if flagged.notna().any() else pd.NaT
        rows.append({"peak": pk.date(), "trough": trough.date(), "depth": seg.min(),
                     "days_to_trough": close.index.get_loc(trough) - close.index.get_loc(pk),
                     "days_to_recover": (close.index.get_loc(recovered) - close.index.get_loc(pk)) if pd.notna(recovered) else np.nan,
                     "flag_date": flag.date() if pd.notna(flag) else None,
                     "fall_done_at_flag": (dd.loc[flag] / seg.min()) if pd.notna(flag) else np.nan})
    return pd.DataFrame(rows)


def stress_runs(p: pd.Series, durations: pd.Series, min_len: int = 5) -> pd.DataFrame:
    on = (p > 0.5).to_numpy()
    rows, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= min_len:
                rows.append({"start": p.index[i].date(), "length": j - i, "predicted": durations.iloc[i]})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def run_regime_study(workers: int | None = None, start: str = "1993-01-01") -> None:
    import os
    from datetime import date
    keep_awake()
    end = date.today().isoformat()
    C = {a: get_bars("yahoo", a, start, end)["close"] for a in ASSETS}
    cash_all = fetch_tbill_daily(start, end)
    with ProcessPoolExecutor(max_workers=workers or min(len(ASSETS), max(1, (os.cpu_count() or 2) - 2))) as pool:
        probs = dict(pool.map(_task, [(a, c.pct_change().fillna(0.0)) for a, c in C.items()]))
    pd.set_option("display.width", 220)
    rows_full, rows_hold, runs_all, eps_all = [], [], [], {}
    for a in ASSETS:
        close, P = C[a], probs[a]
        R = close.pct_change().fillna(0.0)
        cash = cash_all.reindex(close.index).ffill().fillna(0.0)
        pos = positions(P["p_stress"], close)
        n = len(R)
        rets = {"buy_hold": R, **{k: portfolio_returns(R, w, cash) for k, w in pos.items()}}
        for which, lo, store in (("full", WARM, rows_full), ("holdout", WARM + int((n - WARM) * 0.75), rows_hold)):
            stats = {k: _stats(v.iloc[lo:], cash.iloc[lo:]).iloc[0] for k, v in rets.items()}
            for k in rets:
                row = {"asset": a, "strategy": k, "sharpe": stats[k].sharpe, "cagr": stats[k].cagr,
                       "maxdd": stats[k].maxdd, "d_sharpe": stats[k].sharpe - stats["buy_hold"].sharpe,
                       "invested": float(pos[k].iloc[lo:].mean()) if k in pos else 1.0,
                       "switches": float(pos[k].iloc[lo:].diff().abs().sum()) if k in pos else 0.0}
                if k in pos:
                    row["p_random_timing"] = shift_test(pos[k], R, cash, lo)[1]
                store.append(row)
        runs = stress_runs(P["p_stress"], P["stress_days"])
        runs["asset"] = a
        runs_all.append(runs)
        eps_all[a] = drawdown_episodes(close, P["p_stress"])
    full, hold = pd.DataFrame(rows_full), pd.DataFrame(rows_hold)
    full.to_csv(DATA_DIR / "regime_full.csv", index=False)
    hold.to_csv(DATA_DIR / "regime_holdout.csv", index=False)
    pd.concat(probs, names=["asset"]).to_csv(DATA_DIR / "regime_probs.csv")

    def agg(df):
        g = df.groupby("strategy")
        t = g.agg(avg_sharpe=("sharpe", "mean"), avg_d_sharpe=("d_sharpe", "mean"), avg_cagr=("cagr", "mean"),
                  avg_maxdd=("maxdd", "mean"), invested=("invested", "mean"), switches=("switches", "mean"),
                  assets_sharpe_up=("d_sharpe", lambda s: int((s > 0).sum())),
                  median_p_random=("p_random_timing", "median"))
        return t.loc[["buy_hold", "trend200", *VARIANTS]]
    print(f"{len(ASSETS)} index funds {ASSETS}, history from {start}; regime model refit every {REFIT} days, "
          f"first {WARM} days training only.\n")
    print("ALL OUT-OF-SAMPLE DAYS, averaged over the 7 funds (Sharpe on returns above T-bill cash):\n" + agg(full).round(3).to_string())
    print("\nLAST 25% OF DAYS:\n" + agg(hold).round(3).to_string())
    print("\nPER FUND, all days (d_sharpe = Sharpe minus buy & hold; p_random_timing = chance random timing with the same\n"
          "exposure and switching matches the real Sharpe; bar for 3 variants: p < 0.017):")
    print(full[full.strategy.isin(VARIANTS)].round(3).to_string(index=False))

    spy_eps = eps_all["SPY"]
    print("\nSPY DRAWDOWNS OF 10% OR MORE, and when the regime model flagged stress (days = trading days):")
    print(spy_eps.round(3).to_string(index=False))
    runs = pd.concat(runs_all)
    if len(runs) > 3:
        rho = runs[["length", "predicted"]].corr(method="spearman").iloc[0, 1]
        print(f"\nSTRESS PERIODS (flagged for 5+ days) across the 7 funds: {len(runs)} periods, median length "
              f"{runs.length.median():.0f} days, longest {runs.length.max()} days; the model's expected duration at the "
              f"onset has Spearman correlation {rho:.2f} with the realized length (1 = perfect, 0 = no information).")
    print("\nA variant counts only if it beats buy & hold on Sharpe AND random timing (p < 0.017) AND holds in the last 25%.")
