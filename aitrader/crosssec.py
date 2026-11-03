"""Cross-sectional stock selection: pool ~155 large US stocks, predict each stock's rank of next-month return
from rank-normalized features, hold the top fifth, and compare against the equal-weight basket.

Walk-forward: models train only on months whose outcomes were already known, retrained once a year.
Hyperparameters are fixed in advance (not tuned), so the number of "trials" is just the three models below.

SURVIVORSHIP BIAS: the universe is today's large caps, i.e. companies that survived and grew. Benchmarking
against the equal-weight basket of the same stocks removes most of the level bias, but not all of it. Treat
every result as an optimistic upper bound. Fixing this needs point-in-time constituent data (paid).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .config import DATA_DIR

UNIVERSE = """AAPL MSFT INTC CSCO ORCL IBM QCOM TXN ADBE NVDA AMD MU AMAT ADI KLAC LRCX HPQ INTU ADSK CTSH
T VZ CMCSA DIS NFLX GOOGL META AMZN HD LOW MCD SBUX NKE TGT F GM TJX ROST YUM MAR EBAY BBY ORLY GPC
PG KO PEP WMT COST MDLZ CL KMB GIS MO PM KR SYY HSY CLX STZ ADM
JNJ PFE MRK ABT LLY BMY AMGN GILD UNH CVS CI HUM MDT BAX BDX SYK ZBH TMO DHR BIIB REGN ISRG
JPM BAC WFC C GS MS AXP USB PNC BK SCHW BLK MET PRU AIG CB TRV ALL MMC SPGI AFL CME
GE HON CAT DE MMM UPS FDX BA LMT NOC GD EMR ITW ETN CSX UNP NSC WM LUV ROK PCAR
XOM CVX COP SLB OXY EOG HAL VLO APA DVN LIN APD ECL SHW DD NEM FCX NUE
NEE DUK SO D AEP EXC SRE XEL ED SPG PLD AMT""".split()

FEATURES = ["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "vol_3m", "hi_52w", "vol_trend", "beta"]
MODELS = ["mom12_1", "ridge", "gbm"]


def load_universe(start: str = "1993-01-01", refresh: bool = False, symbols: list[str] | None = None):
    """Adjusted daily close and volume for the universe, cached in data/."""
    cc, vc = DATA_DIR / "universe_close.csv", DATA_DIR / "universe_volume.csv"
    if not refresh and cc.exists() and vc.exists():
        close = pd.read_csv(cc, index_col=0, parse_dates=True)
        volume = pd.read_csv(vc, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(sorted(set(symbols or UNIVERSE)), start=start, auto_adjust=True, progress=False,
                          threads=True)
        close, volume = raw["Close"], raw["Volume"]
        for df in (close, volume):
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        close = close.dropna(axis=1, how="all")
        volume = volume[close.columns]
        close.to_csv(cc)
        volume.to_csv(vc)
    if symbols:
        close, volume = close[[s for s in symbols if s in close]], volume[[s for s in symbols if s in volume]]
    return close, volume


def build_panel(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """One row per (month-end, stock): rank-normalized features, forward 1-month return and its rank (y)."""
    R = close.pct_change()
    mkt = R.mean(axis=1)
    feats = {
        "mom_1m": close / close.shift(21) - 1,
        "mom_3m": close / close.shift(63) - 1,
        "mom_6m": close / close.shift(126) - 1,
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "vol_3m": R.rolling(63).std(),
        "hi_52w": close / close.rolling(252).max() - 1,
        "vol_trend": np.log(volume.rolling(21).mean() / volume.rolling(252).mean()),
        "beta": ((R.mul(mkt, axis=0).rolling(252).mean() - R.rolling(252).mean().mul(mkt.rolling(252).mean(), axis=0))
                 .div((mkt ** 2).rolling(252).mean() - mkt.rolling(252).mean() ** 2, axis=0)),
    }
    idx = close.index
    me = np.flatnonzero(np.r_[idx.month[1:] != idx.month[:-1], False])  # complete months only
    mc = close.iloc[me]
    fwd = mc.shift(-1) / mc - 1
    mi = pd.MultiIndex.from_product([idx[me], close.columns], names=["date", "symbol"])
    panel = pd.DataFrame({k: v.iloc[me].to_numpy().ravel() for k, v in feats.items()}, index=mi)
    panel["fwd"] = fwd.to_numpy().ravel()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel[panel[FEATURES].notna().all(axis=1)].copy()
    # Beta-adjusted forward return: what the stock did beyond its trailing beta times the equal-weight basket.
    panel["beta_raw"] = panel["beta"]
    panel["resid"] = panel["fwd"] - panel["beta_raw"] * panel.groupby(level="date")["fwd"].transform("mean")
    g = panel.groupby(level="date")
    for f in FEATURES:
        panel[f] = g[f].rank(pct=True) - 0.5
    panel["y"] = g["fwd"].rank(pct=True) - 0.5
    panel["y_resid"] = g["resid"].rank(pct=True) - 0.5
    return panel


class _SeedAverage:
    """Average of several small neural nets trained from different random seeds (reduces seed noise)."""

    def __init__(self, seeds=(0, 1, 2)):
        self.seeds = seeds
        self.nets = []

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        self.nets = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=200,
                                  early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
                                  random_state=s).fit(X, y) for s in self.seeds]
        return self

    def predict(self, X):
        return np.mean([n.predict(X) for n in self.nets], axis=0)


def _make_model(name: str):
    if name == "mlp":
        return _SeedAverage()
    if name == "ridge":
        return Ridge(alpha=10.0)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)


def walk_forward_scores(panel: pd.DataFrame, model: str, min_train: int = 60, retrain_every: int = 12,
                        target: str = "y", features: list[str] | None = None) -> pd.Series:
    """Out-of-sample score for every (month, stock) from `min_train` months in. Training rows are months before the
    scored month whose forward return is already realized. `target` is "y" (return rank) or "y_resid"
    (beta-adjusted return rank)."""
    feats = features or FEATURES
    dates = panel.index.get_level_values("date")
    udates = dates.unique().sort_values()
    by_date = panel.groupby(level="date").indices
    scores = pd.Series(np.nan, index=panel.index)
    fitted = None
    for i in range(min_train, len(udates)):
        rows = by_date[udates[i]]
        X = panel.iloc[rows][feats]
        if model == "mom12_1":
            scores.iloc[rows] = panel.iloc[rows]["mom_12_1"].to_numpy()
            continue
        if fitted is None or (i - min_train) % retrain_every == 0:
            train = panel[(dates <= udates[i - 1]) & panel[target].notna()]
            fitted = _make_model(model).fit(train[feats], train[target])
        scores.iloc[rows] = fitted.predict(X)
    return scores


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {"ann_return": eq.iloc[-1] ** (12 / len(r)) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0,
            "max_dd": (eq / eq.cummax() - 1).min()}


def monthly_results(panel: pd.DataFrame, scores: pd.Series, top_frac: float = 0.2, cost_bps: float = 10.0) -> pd.DataFrame:
    """Per month: top-fifth return (gross/net of costs), bottom-fifth, equal-weight basket, rank IC."""
    df = panel[["fwd", "resid", "beta_raw"]].assign(score=scores).dropna()
    rows, prev = [], set()
    for d, g in df.groupby(level="date"):
        g = g.droplevel("date")
        n = max(3, int(round(top_frac * len(g))))
        top, bottom = g.nlargest(n, "score"), g.nsmallest(n, "score")
        replaced = 1 - len(prev & set(top.index)) / n if prev else 1.0
        cost = replaced * 2 * cost_bps / 1e4 if prev else cost_bps / 1e4
        prev = set(top.index)
        rows.append({"date": d, "top_gross": top["fwd"].mean(), "top_net": top["fwd"].mean() - cost,
                     "bottom": bottom["fwd"].mean(), "ew": g["fwd"].mean(),
                     "ic": g["score"].corr(g["fwd"], method="spearman"), "n_stocks": len(g), "n_picks": n,
                     # beta-hedged view: the picks' return minus (their average beta x the equal-weight basket)
                     "top_resid": top["resid"].mean(), "resid_ew": g["resid"].mean(), "top_beta": top["beta_raw"].mean(),
                     "ic_resid": g["score"].corr(g["resid"], method="spearman"), "cost": cost})
    return pd.DataFrame(rows).set_index("date")


def summarize(res: pd.DataFrame) -> dict:
    ex = res["top_net"] - res["ew"]
    ic = res["ic"].dropna()
    s = _stats(res["top_net"])
    e = _stats(res["ew"])
    return {"months": len(res), "top_ann_return": s["ann_return"], "ew_ann_return": e["ann_return"],
            "excess_ann": s["ann_return"] - e["ann_return"], "top_sharpe": s["sharpe"], "ew_sharpe": e["sharpe"],
            "top_max_dd": s["max_dd"], "ew_max_dd": e["max_dd"],
            "ic_mean": ic.mean(), "ic_tstat": ic.mean() / (ic.std() / np.sqrt(len(ic))),
            "months_beating_ew": float((ex > 0).mean()),
            "info_ratio": ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else 0.0}


def summarize_hedged(res: pd.DataFrame) -> dict:
    """Metrics for the beta-hedged portfolio (long the picks, short beta x equal-weight basket), net of trading costs."""
    h = res["top_resid"] - res["cost"]
    ic = res["ic_resid"].dropna()
    return {"months": len(res), "alpha_ann": h.mean() * 12, "hedged_sharpe": h.mean() / h.std() * np.sqrt(12),
            "alpha_tstat": h.mean() / (h.std() / np.sqrt(len(h))), "ic_resid": ic.mean(),
            "ic_resid_tstat": ic.mean() / (ic.std() / np.sqrt(len(ic))), "avg_pick_beta": res["top_beta"].mean(),
            "months_positive": float((h > 0).mean())}


def random_pick_pvalue(panel: pd.DataFrame, res: pd.DataFrame, reps: int = 1000, seed: int = 0,
                       resid: bool = False) -> float:
    """Luck test: how often do RANDOM top-fifth baskets (same size, same months) match the model's gross excess
    return over the equal-weight basket? Small p means the model picked better than chance. With `resid`, returns
    are beta-adjusted first, so simply picking riskier stocks no longer counts as skill."""
    rng = np.random.default_rng(seed)
    fwd = panel["resid" if resid else "fwd"].dropna()
    total = np.zeros(reps)
    months = 0
    for d in res.index:
        v = fwd.xs(d, level="date").to_numpy()
        m, n = len(v), int(res.loc[d, "n_picks"])
        pick = np.argpartition(rng.random((reps, m)), n - 1, axis=1)[:, :n]
        total += v[pick].mean(axis=1) - v.mean()
        months += 1
    real = (res["top_resid"] - res["resid_ew"]).sum() if resid else (res["top_gross"] - res["ew"]).sum()
    return float(((total >= real).sum() + 1) / (reps + 1))


def run_neutral(start: str = "1993-01-01", top_frac: float = 0.2, cost_bps: float = 10.0, reps: int = 1000,
                only: list[str] | None = None) -> None:
    """Does any signal survive once market exposure is removed? Variants are fixed in advance. `only` runs a subset
    (e.g. just the neural-net variants plus a reference)."""
    close, volume = load_universe(start)
    print(f"{close.shape[1]} stocks, {close.index[0].date()} .. {close.index[-1].date()}  (survivorship-biased universe)\n")
    panel = build_panel(close, volume)
    norisk = [f for f in FEATURES if f not in ("beta", "vol_3m")]
    variants = {
        "mom12_1": ("mom12_1", "y", None),
        "ridge_raw": ("ridge", "y", None), "gbm_raw": ("gbm", "y", None),
        "ridge_resid": ("ridge", "y_resid", None), "gbm_resid": ("gbm", "y_resid", None),
        "ridge_resid_norisk": ("ridge", "y_resid", norisk), "gbm_resid_norisk": ("gbm", "y_resid", norisk),
        "mlp_raw": ("mlp", "y", None), "mlp_resid": ("mlp", "y_resid", None),
        "mlp_resid_norisk": ("mlp", "y_resid", norisk),
    }
    total_trials = len(variants)
    full, hold = {}, {}
    for name, (model, target, feats) in variants.items():
        if (only is not None and name not in only) or (only is None and model == "mlp"):
            continue
        res = monthly_results(panel, walk_forward_scores(panel, model, target=target, features=feats),
                              top_frac, cost_bps)
        res.to_csv(DATA_DIR / f"crosssec_neutral_{name}.csv")
        full[name] = summarize_hedged(res) | {"p_random": random_pick_pvalue(panel, res, reps, resid=True)}
        h = res.iloc[int(len(res) * 0.75):]
        hold[name] = summarize_hedged(h) | {"p_random": random_pick_pvalue(panel, h, reps, resid=True)}
        print(f"  {name} done", flush=True)
    cols = ["months", "alpha_ann", "hedged_sharpe", "alpha_tstat", "ic_resid", "ic_resid_tstat", "avg_pick_beta",
            "months_positive", "p_random"]
    print("\nBeta-HEDGED results, all out-of-sample months (long the picks, short beta x equal-weight basket, net of costs):")
    print(pd.DataFrame(full).T[cols].round(3).to_string())
    print("\nLast 25% of months only:")
    print(pd.DataFrame(hold).T[cols].round(3).to_string())
    print("\n_raw = trained on raw return rank, then hedged afterwards. _resid = trained on beta-adjusted return rank. "
          "_norisk = also without the beta and volatility features.\n"
          f"{total_trials} variants are tried in total, so demand more than the usual bar: alpha_tstat above ~3 and "
          "p_random below ~0.01.")


def run_crosssec(start: str = "1993-01-01", refresh: bool = False, top_frac: float = 0.2, cost_bps: float = 10.0,
                 reps: int = 1000, symbols: list[str] | None = None) -> None:
    close, volume = load_universe(start, refresh, symbols)
    print(f"{close.shape[1]} stocks, {close.index[0].date()} .. {close.index[-1].date()}")
    print("NOTE: survivorship-biased universe (today's large caps). Results are optimistic upper bounds.\n")
    panel = build_panel(close, volume)
    all_rows, hold_rows = {}, {}
    for model in MODELS:
        res = monthly_results(panel, walk_forward_scores(panel, model), top_frac, cost_bps)
        res.to_csv(DATA_DIR / f"crosssec_monthly_{model}.csv")
        s = summarize(res)
        s["p_random_picks"] = random_pick_pvalue(panel, res, reps)
        all_rows[model] = s
        h = res.iloc[int(len(res) * 0.75):]
        hold_rows[model] = summarize(h) | {"p_random_picks": random_pick_pvalue(panel, h, reps)}
        print(f"  {model} done", flush=True)
    cols = ["months", "top_ann_return", "ew_ann_return", "excess_ann", "top_sharpe", "ew_sharpe", "top_max_dd",
            "ew_max_dd", "ic_mean", "ic_tstat", "months_beating_ew", "info_ratio", "p_random_picks"]
    print(f"\nAll out-of-sample months (top {top_frac:.0%} of stocks, net of {cost_bps:g} bps per trade):")
    print(pd.DataFrame(all_rows).T[cols].round(3).to_string())
    print("\nLast 25% of months only (a holdout view; no model was tuned on any period):")
    print(pd.DataFrame(hold_rows).T[cols].round(3).to_string())
    print("\nReading it: ic_tstat above ~2 and p_random_picks below ~0.05 would suggest real skill. "
          "Excess return that is below those bars is indistinguishable from luck.")
