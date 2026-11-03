"""Volatility forecasting study. Volatility is the one market quantity that is genuinely predictable, and our best
(only consistent) strategy, volatility targeting, depends on forecasting it. Does a better forecast help?

Ten forecasters, fixed in advance (variance over the next H = 5 trading days, for 17 ETFs, 2004-2026):
  naive20 (the frozen forward-test rule's estimate), naive60, ewma94 (RiskMetrics), garch11 (Student-t),
  har (Corsi: daily/weekly/monthly realized variance), har_vix (HAR + VIX), gbm, gbm_vix (pooled gradient boosting),
  mlp, mlp_vix (pooled small neural net, 2-seed average).
All are walk-forward: a forecast at date t only uses data through t, and models only train on targets already realized.
Step 1 scores the forecasts directly (QLIKE loss, the standard measure). Step 2 plugs each into the volatility-targeting
rule (10% target, cap 1.0; secondary: 15% target, cap 1.5) with T-bill cash and 1 bp costs, and tests the Sharpe change
against naive20 with a block bootstrap. Primary test: vt10_cap1, 9 comparisons, so the bar is p < 0.0055.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from .config import DATA_DIR, keep_awake
from .data import fetch_tbill_daily, fetch_vix, get_bars
from .riskstudy import ASSETS, CLASSES, _per_asset, _stats

H, EPS, WARM, BLOCK = 5, 1e-8, 504, 20
MODELS = ["naive20", "naive60", "ewma94", "garch11", "har", "har_vix", "gbm", "gbm_vix", "mlp", "mlp_vix"]


def future_var(R: pd.DataFrame) -> pd.DataFrame:
    """Mean squared return over t+1..t+H, aligned to t (the quantity being forecast)."""
    return (R ** 2).rolling(H).mean().shift(-H)


def build_features(R: pd.DataFrame, vixvar: pd.Series | None = None) -> dict:
    r2 = R ** 2
    f = {"rv1": np.log(r2 + EPS), "rv5": np.log(r2.rolling(5).mean() + EPS),
         "rv22": np.log(r2.rolling(22).mean() + EPS), "rv66": np.log(r2.rolling(66).mean() + EPS),
         "rv252": np.log(r2.rolling(252).mean() + EPS),
         "dsv22": np.log((R.clip(upper=0) ** 2).rolling(22).mean() + EPS),
         "ret5": R.rolling(5).sum(), "ret22": R.rolling(22).sum()}
    if vixvar is not None:
        lv = np.log(vixvar.reindex(R.index).ffill() + EPS)
        f["vix"] = pd.DataFrame({c: lv for c in R.columns})
        f["vix_rel"] = f["vix"] - f["rv22"]
    return f


def har_forecast(R, vixvar, use_vix: bool, refit_every: int = 63, warm: int = WARM) -> pd.DataFrame:
    """Per-asset linear regression of next-H log variance on daily/weekly/monthly (and VIX) terms."""
    feats = build_features(R, vixvar if use_vix else None)
    names = ["rv1", "rv5", "rv22"] + (["vix"] if use_vix else [])
    y = np.log(future_var(R) + EPS)
    n = len(R)
    out = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for a in R.columns:
        X = np.column_stack([np.ones(n)] + [feats[k][a].to_numpy() for k in names])
        yy = y[a].to_numpy()
        t = warm
        while t < n:
            end = min(t + refit_every, n)
            Xtr, ytr = X[:t - H + 1], yy[:t - H + 1]  # rows whose H-day outcome is complete by t
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
            beta = np.linalg.lstsq(Xtr[ok], ytr[ok], rcond=None)[0]
            s2 = np.var(ytr[ok] - Xtr[ok] @ beta)
            out.iloc[t:end, out.columns.get_loc(a)] = np.exp(X[t:end] @ beta + 0.5 * s2)
            t = end
    return out


class _ScaledMLP:
    """Small neural net on standardized inputs and a standardized target, averaged over two seeds."""

    def __init__(self, seeds=(0, 1)):
        self.seeds, self.nets = seeds, []

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.sx, self.my, self.sy = StandardScaler().fit(X), y.mean(), y.std()
        Xs, ys = self.sx.transform(X), (y - self.my) / self.sy
        self.nets = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=100,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                                  random_state=s).fit(Xs, ys) for s in self.seeds]
        return self

    def predict(self, X):
        Xs = self.sx.transform(X)
        return np.mean([n.predict(Xs) for n in self.nets], axis=0) * self.sy + self.my


def _make(model: str):
    if model == "mlp":
        return _ScaledMLP()
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100,
                                         random_state=0)


def pooled_forecast(R, vixvar, model: str, use_vix: bool, refit_every: int = 126, warm: int = WARM) -> pd.DataFrame:
    """One model trained on every asset together (dimensionless log-variance features), refit every `refit_every` days."""
    feats = build_features(R, vixvar if use_vix else None)
    X = np.stack([feats[k].to_numpy() for k in feats], axis=-1)  # time x asset x feature
    y = np.log(future_var(R).to_numpy() + EPS)
    n, A, F = X.shape
    out = np.full((n, A), np.nan)
    t = warm
    while t < n:
        end = min(t + refit_every, n)
        Xtr, ytr = X[:t - H + 1].reshape(-1, F), y[:t - H + 1].reshape(-1)
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        m = _make(model).fit(Xtr[ok], ytr[ok])
        s2 = np.var(ytr[ok] - m.predict(Xtr[ok]))
        Xte = X[t:end].reshape(-1, F)
        good = np.isfinite(Xte).all(axis=1)
        pred = np.full(len(Xte), np.nan)
        pred[good] = np.exp(m.predict(Xte[good]) + 0.5 * s2)
        out[t:end] = pred.reshape(end - t, A)
        t = end
    return pd.DataFrame(out, index=R.index, columns=R.columns)


def garch_asset(r: pd.Series, refit_every: int = 63, warm: int = WARM) -> pd.Series:
    """Zero-mean GARCH(1,1) with Student-t errors, refit on past data every `refit_every` days; between refits the
    variance is filtered forward with fixed parameters. Returns the mean forecast over the next H days (daily variance)."""
    from arch import arch_model
    x = r * 100
    out = pd.Series(np.nan, index=r.index)
    n, t = len(x), warm
    while t < n:
        end = min(t + refit_every, n)
        res = arch_model(x.iloc[:t], mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
        fixed = arch_model(x.iloc[:end], mean="Zero", vol="GARCH", p=1, q=1, dist="t").fix(res.params)
        f = fixed.forecast(horizon=H, start=t, reindex=False).variance
        out.iloc[t:end] = f.mean(axis=1).to_numpy()[:end - t] / 1e4
        t = end
    return out


def _task(args):
    kind, payload = args
    if kind == "garch":
        asset, r = payload
        return kind, asset, garch_asset(r)
    name, R, vixvar = payload
    return kind, name, pooled_forecast(R, vixvar, name.split("_")[0], name.endswith("_vix"))


def compute_forecasts(R: pd.DataFrame, vixvar: pd.Series, workers: int | None = None) -> dict:
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    f = {"naive20": R.rolling(20).var(), "naive60": R.rolling(60).var(),
         "ewma94": (R ** 2).ewm(alpha=0.06, adjust=False).mean(),
         "har": har_forecast(R, vixvar, False), "har_vix": har_forecast(R, vixvar, True)}
    tasks = [("garch", (a, R[a])) for a in R.columns] + \
            [("pooled", (m, R, vixvar)) for m in ("gbm", "gbm_vix", "mlp", "mlp_vix")]
    garch = {}
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for kind, key, val in pool.map(_task, tasks):
            if kind == "garch":
                garch[key] = val
            else:
                f[key] = val
                print(f"  {key} forecasts done", flush=True)
    f["garch11"] = pd.DataFrame(garch)[R.columns]
    return {m: f[m] for m in MODELS}


def qlike(real: np.ndarray, fc: np.ndarray) -> np.ndarray:
    ratio = np.maximum(real, EPS) / np.maximum(fc, EPS)
    return ratio - np.log(ratio) - 1


def hac_t(d: np.ndarray, lag: int = 10) -> float:
    d = d[np.isfinite(d)]
    n, dc = len(d), d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return float(d.mean() / np.sqrt(v / n))


def accuracy_table(R: pd.DataFrame, fc: dict, which: str) -> pd.DataFrame:
    real = future_var(R).to_numpy()
    n = len(R) - H
    lo = WARM if which == "full" else WARM + int((n - WARM) * 0.75)
    sl = slice(lo, n)
    loss = {m: qlike(real[sl], f.to_numpy()[sl]) for m, f in fc.items()}  # time x asset
    base = loss["naive20"]
    rows = {}
    for m, L in loss.items():
        per_asset = np.nanmean(L, axis=0)
        t_stats = np.array([hac_t(base[:, a] - L[:, a]) for a in range(L.shape[1])])
        rows[m] = {"qlike": per_asset.mean(), "vs_naive20": 1 - per_asset.mean() / np.nanmean(base, axis=0).mean(),
                   "assets_better": int((per_asset < np.nanmean(base, axis=0)).sum()),
                   "assets_signif_better": int((t_stats > 2).sum()), "mean_dm_t": t_stats.mean()}
    return pd.DataFrame(rows).T


def strategy_returns(R, fc, cash, target: float, cap: float) -> dict:
    out = {}
    for m, f in fc.items():
        vol = np.sqrt(f * 252)
        out[m] = _per_asset(R, (target / vol).clip(upper=cap).fillna(0.0), cash)
    return out


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) / x.std(axis=0) * np.sqrt(252)


def strategy_table(R, cash, rets: dict, which: str, reps: int = 500, seed: int = 11) -> pd.DataFrame:
    n = len(R)
    lo = WARM if which == "full" else WARM + int((n - WARM) * 0.75)
    ex = {m: r.sub(cash, axis=0).iloc[lo:].to_numpy() for m, r in rets.items()}
    T = ex["naive20"].shape[0]
    rng = np.random.default_rng(seed)
    idxs = []
    for _ in range(reps):
        idx: list[int] = []
        while len(idx) < T:
            s = int(rng.integers(0, T - BLOCK + 1))
            idx.extend(range(s, s + BLOCK))
        idxs.append(np.array(idx[:T]))
    base_obs = _sharpe_cols(ex["naive20"]).mean()
    rows = {}
    for m, x in ex.items():
        obs = _sharpe_cols(x).mean() - base_obs
        boot = np.array([_sharpe_cols(x[i]).mean() - _sharpe_cols(ex["naive20"][i]).mean() for i in idxs])
        st = _stats(rets[m].iloc[lo:], cash.iloc[lo:])
        rows[m] = {"avg_sharpe": _sharpe_cols(x).mean(), "d_sharpe_vs_naive20": obs,
                   "p_better": float(((boot - obs >= obs).sum() + 1) / (reps + 1)) if m != "naive20" else np.nan,
                   "avg_cagr": st.cagr.mean(), "avg_maxdd": st.maxdd.mean()}
    return pd.DataFrame(rows).T


def run_vol_study(reps: int = 500, workers: int | None = None) -> None:
    from datetime import date
    keep_awake()
    end = date.today().isoformat()
    C = pd.DataFrame({a: get_bars("yahoo", a, "2003-01-01", end)["close"] for a in sorted(set(ASSETS))}).dropna()
    R = C.pct_change().fillna(0.0)
    cash = fetch_tbill_daily("2003-01-01", end).reindex(C.index).ffill().fillna(0.0)
    vix = fetch_vix("2003-01-01", end).reindex(C.index).ffill()
    vixvar = (vix / 100) ** 2 / 252
    print(f"{R.shape[1]} ETFs, {C.index[0].date()} .. {C.index[-1].date()}; forecasting mean squared return over the "
          f"next {H} days; first {WARM} days are training only.\n")
    fc = compute_forecasts(R, vixvar, workers)
    pd.set_option("display.width", 200)
    acc_full, acc_hold = accuracy_table(R, fc, "full"), accuracy_table(R, fc, "holdout")
    acc_full.to_csv(DATA_DIR / "volforecast_accuracy_full.csv")
    acc_hold.to_csv(DATA_DIR / "volforecast_accuracy_holdout.csv")
    print("\nFORECAST ACCURACY (QLIKE loss, lower is better; vs_naive20 = improvement; DM t > 2 = significantly "
          "better than naive20)\nAll months:\n" + acc_full.round(3).to_string())
    print("\nLast 25% of months:\n" + acc_hold.round(3).to_string())

    summary = {}
    for label, tgt, cap in (("vt10_cap1 (primary, the forward-test rule)", 0.10, 1.0), ("vt15_cap1.5 (secondary)", 0.15, 1.5)):
        rets = strategy_returns(R, fc, cash, tgt, cap)
        full, hold = strategy_table(R, cash, rets, "full", reps), strategy_table(R, cash, rets, "holdout", reps)
        key = label.split()[0]
        full.to_csv(DATA_DIR / f"volforecast_strategy_{key}_full.csv")
        hold.to_csv(DATA_DIR / f"volforecast_strategy_{key}_holdout.csv")
        summary[key] = rets
        print(f"\nSTRATEGY: {label}, averaged over 17 ETFs; p_better = block-bootstrap p-value that the Sharpe "
              f"gain over naive20 is real (primary bar p < 0.0055)\nAll months:\n" + full.round(4).to_string())
        print("\nLast 25% of months:\n" + hold.round(4).to_string())
    bh = _stats(R.iloc[WARM:], cash.iloc[WARM:])
    print(f"\nBuy & hold reference (same period): average Sharpe {bh.sharpe.mean():.3f}, "
          f"average max drop {bh.maxdd.mean():.1%}")
    print("\nA forecaster only matters for the strategy if it beats naive20 on BOTH accuracy and Sharpe.")
