"""Generate the README charts into docs/. Needs the cached data and saved study results (run the studies first).

    python make_charts.py
"""
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aitrader.backtest import position_returns
from aitrader.config import DATA_DIR
from aitrader.data import get_bars
from aitrader.rotation import WARMUP as ROT_WARMUP
from aitrader.rotation import benchmarks, load_universe, rotation_returns
from aitrader.strategy import TrendFilter, VolTarget

OUT = Path(__file__).parent / "docs"
OUT.mkdir(exist_ok=True)

# Reference palette (light surface); categorical slots 1-3 validate for all pairs.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, BLUE_STEP = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
BENCH = MUTED  # the thing every strategy is compared against stays neutral

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9,
})


def finish(ax, title, subtitle, path, fig):
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / path, dpi=180)
    plt.close(fig)
    print("wrote", OUT / path)


def spread(values: dict, min_gap: float) -> dict:
    """Nudge label y-positions apart so end labels don't collide."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    ys = [v for _, v in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    return {k: y for (k, _), y in zip(items, ys)}


def equity_chart(curves: dict, colors: dict, title, subtitle, path, shade_from=None, shade_mask=None):
    fig, ax = plt.subplots(figsize=(9, 4.9))
    if shade_mask is not None:  # light bands where the mask is True (e.g. days a model flagged stress)
        idx = list(curves.values())[0].index
        m = shade_mask.reindex(idx).fillna(False).to_numpy(bool)
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                ax.axvspan(idx[i], idx[min(j, len(m) - 1)], color=INK, alpha=0.09, lw=0)
                i = j
            else:
                i += 1
    for name, r in curves.items():
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq, color=colors[name], lw=2, label=name, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.5, 1, 2, 3, 5, 8, 10, 20, 30) if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}x" for t in ticks])
    ax.minorticks_off()
    if shade_from is not None:
        ax.axvspan(shade_from, list(curves.values())[0].index[-1], color=INK, alpha=0.05, lw=0)
        ax.text(shade_from, 0.02, "  holdout: never used to pick strategies", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=INK2, va="bottom")
    last = {n: np.log10((1 + r).cumprod().iloc[-1]) for n, r in curves.items()}
    lo, hi = ax.get_ylim()
    ypos = spread(last, 0.036 * np.log10(hi / lo))
    x_end = list(curves.values())[0].index[-1]
    pad = (x_end - list(curves.values())[0].index[0]) * 0.012
    for name, r in curves.items():
        mult = (1 + r).cumprod().iloc[-1]
        ax.plot([x_end + pad * 0.4], [10 ** ypos[name]], marker="o", ms=5, color=colors[name], clip_on=False)
        ax.text(x_end + pad * 1.6, 10 ** ypos[name], f"{mult:.1f}x", color=INK, fontsize=9, va="center")
    ax.set_xlim(right=x_end + (x_end - list(curves.values())[0].index[0]) * 0.07)
    ax.legend(loc="upper left", ncol=min(len(curves), 3))
    finish(ax, title, subtitle, path, fig)


def drawdown_chart(curves: dict, colors: dict, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, r in curves.items():
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd * 100, color=colors[name], lw=1.6, label=f"{name} (worst {dd.min():.0%})")
    ax.set_ylabel("Drop from previous peak (%)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlim(list(curves.values())[0].index[0], list(curves.values())[0].index[-1])
    ax.legend(loc="lower left")
    finish(ax, title, subtitle, path, fig)


VARIANT_LABELS = {
    "mom12_1": "Momentum alone", "ridge_raw": "Ridge (hedged afterwards)", "gbm_raw": "Gradient boosting (hedged afterwards)",
    "ridge_resid": "Ridge (beta-adjusted target)", "gbm_resid": "Gradient boosting (beta-adjusted target)",
    "ridge_resid_norisk": "Ridge, no beta/vol features", "gbm_resid_norisk": "Gradient boosting, no beta/vol features",
    "mlp_raw": "Neural net (hedged afterwards)", "mlp_resid": "Neural net (beta-adjusted target)",
    "mlp_resid_norisk": "Neural net, no beta/vol features",
}


def crosssec_charts():
    """Beta-hedged alpha per variant (all months vs the last 25%) and the rolling rank-IC of the best models."""
    rows = {}
    for key, label in VARIANT_LABELS.items():
        f = DATA_DIR / f"crosssec_neutral_{key}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f, index_col=0, parse_dates=True)
        h = r["top_resid"] - r["cost"]
        rows[label] = (h.mean() * 12 * 100, h.iloc[int(len(h) * 0.75):].mean() * 12 * 100)
    labels = list(rows)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.9))
    y = np.arange(len(labels))
    allv = np.array([rows[l][0] for l in labels])
    hold = np.array([rows[l][1] for l in labels])
    for i in range(len(labels)):
        ax.plot([allv[i], hold[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.scatter(allv, y, s=60, color=BLUE, label="All months (1998-2026)", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(hold, y, s=60, color=ORANGE, label="Last 25% of months", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Beta-hedged alpha, % per year, after 10 bps trading costs")
    ax.legend(loc="lower right")
    finish(ax, "Hedged against market risk, alpha is small and not significant",
           "154 large US stocks, long the top fifth, short beta x the equal-weight basket. All t-stats below 2.2.",
           "crosssec_alpha.png", fig)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for key, color, label in (("gbm_resid", BLUE, "Gradient boosting"), ("ridge_resid", ORANGE, "Ridge")):
        f = DATA_DIR / f"crosssec_neutral_{key}.csv"
        if f.exists():
            ic = pd.read_csv(f, index_col=0, parse_dates=True)["ic_resid"]
            ax.plot(ic.index, ic.rolling(60).mean(), color=color, lw=2, label=label)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("Rank correlation with next-month\nbeta-adjusted return (5-yr average)")
    ax.legend(loc="upper right")
    finish(ax, "The predictive signal faded toward zero in recent years",
           "Average monthly rank correlation between the model's score and the stock's beta-adjusted return.",
           "crosssec_ic.png", fig)


def riskstudy_charts():
    """Volatility-target Sharpe change by asset class, and multi-asset growth curves with the holdout shaded."""
    from aitrader.data import fetch_tbill_daily
    from aitrader.riskstudy import ASSETS, WARM, compute_all, summarize_by_class

    end = date.today().isoformat()
    C = pd.DataFrame({a: get_bars("yahoo", a, "2003-01-01", end)["close"] for a in sorted(set(ASSETS + ["IEF"]))}).dropna()
    cash = fetch_tbill_daily("2003-01-01", end).reindex(C.index).ffill().fillna(0.0)
    res = compute_all(C, cash)

    tab = summarize_by_class(res, cash, ["vt10_cap1"]).loc["vt10_cap1"]
    classes = list(tab.index)[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    vals = tab.loc[classes, "d_sharpe"].to_numpy(dtype=float)
    ax.barh(classes, vals, color=BLUE, height=0.55)
    ax.axvline(0, color=AXIS, lw=1)
    for i, c in enumerate(classes):
        n = int(tab.loc[c, "assets"])
        k = int(round(tab.loc[c, "sharpe_improved"] * n))
        ax.text(vals[i] + (0.004 if vals[i] >= 0 else -0.004), i, f"{vals[i]:+.2f}  ({k} of {n} improved)",
                va="center", ha="left" if vals[i] >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlim(-0.17, 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average change in Sharpe ratio vs buy-and-hold (10% target, no leverage)")
    finish(ax, "Volatility targeting helped US stocks and sectors, not international or bonds",
           "17 ETFs, 2004-2026. Cash earns the T-bill rate; Sharpe measured on returns above cash.",
           "riskstudy_by_class.png", fig)

    names = {"SPY buy&hold": "SPY buy & hold", "60/40 SPY/IEF": "60/40 SPY/IEF"}
    curves = {"SPY buy & hold": res["rp_bench"]["SPY buy&hold"], "60/40 SPY/IEF": res["rp_bench"]["60/40 SPY/IEF"],
              "Risk parity, 5 assets": res["rp"]["rp_unscaled"],
              "Risk parity, 10% vol, up to 1.5x": res["rp"]["rp_vt10_cap1.5"]}
    curves = {k: v.iloc[WARM:] for k, v in curves.items()}
    n = len(next(iter(curves.values())))
    holdout = next(iter(curves.values())).index[int(n * 0.75)]
    colors = {"SPY buy & hold": BENCH, "60/40 SPY/IEF": AQUA, "Risk parity, 5 assets": BLUE,
              "Risk parity, 10% vol, up to 1.5x": ORANGE}
    equity_chart(curves, colors, "Diversified risk parity had far smaller crashes, but lagged in recent years",
                 "SPY, EFA, TLT, GLD, VNQ weighted by inverse volatility. T-bill cash, 0.5% borrowing spread, 1 bp costs.",
                 "riskstudy_multiasset.png", shade_from=holdout)


def globalstudy_chart():
    """Sharpe change vs buy-and-hold in 19 markets for the frozen volatility-targeting rule and the 200-day trend rule."""
    df = pd.read_csv(DATA_DIR / "globalstudy_markets.csv")
    markets = df[df.kind == "vt10_cap1"][["group", "market"]].reset_index(drop=True)
    rows, ypos, labels, y = [], {}, [], 0
    for grp in dict.fromkeys(markets.group):
        labels.append((y, grp, True))
        y += 1
        for m in markets[markets.group == grp].market:
            ypos[m] = y
            labels.append((y, m, False))
            y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.2), sharey=True)
    for ax, kind, title in ((axes[0], "vt10_cap1", "Volatility targeting (the forward-test rule)"),
                            (axes[1], "trend200", "200-day trend rule")):
        d = df[df.kind == kind]
        yy = d.market.map(ypos).to_numpy()
        ax.errorbar(d.d_sharpe, yy, xerr=[d.d_sharpe - d.ci_lo, d.ci_hi - d.d_sharpe], fmt="none", ecolor=AXIS, lw=1.6, zorder=1)
        ax.scatter(d.d_sharpe, yy, s=48, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{title}\n{int((d.d_sharpe > 0).sum())} of {len(d)} markets improved; average change {d.d_sharpe.mean():+.3f}",
                     loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlim(-0.45, 0.85)
        ax.set_xlabel("Sharpe change vs buy-and-hold (bar = 95% interval)")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks([l[0] for l in labels], [l[1] for l in labels])
    for tick, (_, _, header) in zip(axes[0].get_yticklabels(), labels):
        if header:
            tick.set_fontweight("semibold")
            tick.set_color(INK)
    axes[0].invert_yaxis()
    fig.suptitle("Across 19 markets, volatility targeting did not raise Sharpe; the 200-day trend rule did on average",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "globalstudy.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "globalstudy.png")


def regime_chart():
    """SPY buy-and-hold vs the soft regime overlay and the 200-day rule, with model-flagged stress days shaded."""
    from aitrader.regime import WARM as RW
    from aitrader.regime import positions
    from aitrader.data import fetch_tbill_daily
    from aitrader.riskstudy import portfolio_returns

    end = date.today().isoformat()
    close = get_bars("yahoo", "SPY", "1993-01-01", end)["close"]
    raw = pd.read_csv(DATA_DIR / "regime_probs.csv")
    raw = raw[raw["asset"] == "SPY"].rename(columns={raw.columns[1]: "date"})
    P = raw.set_index(pd.to_datetime(raw["date"]))["p_stress"].reindex(close.index)
    cash = fetch_tbill_daily("1993-01-01", end).reindex(close.index).ffill().fillna(0.0)
    R = close.pct_change().fillna(0.0)
    pos = positions(P, close)
    curves = {"SPY buy & hold": R.iloc[RW:], "Regime overlay (soft)": portfolio_returns(R, pos["hmm_soft"], cash).iloc[RW:],
              "200-day trend": portfolio_returns(R, pos["trend200"], cash).iloc[RW:]}
    colors = {"SPY buy & hold": BENCH, "Regime overlay (soft)": BLUE, "200-day trend": ORANGE}
    equity_chart(curves, colors, "A regime model cut the crashes but gave up return, and flagged stress late",
                 "SPY since 1995. Shaded: days the Markov-switching model said stress. T-bill cash, 1 bp costs.",
                 "regime_spy.png", shade_mask=(P > 0.5))


def volforecast_chart():
    """Forecast accuracy gain vs the 20-day rule next to the Sharpe change it produced."""
    acc = pd.read_csv(DATA_DIR / "volforecast_accuracy_full.csv", index_col=0)
    strat = pd.read_csv(DATA_DIR / "volforecast_strategy_vt10_cap1_full.csv", index_col=0)
    labels = {"naive60": "60-day average", "ewma94": "EWMA (RiskMetrics)", "garch11": "GARCH(1,1)", "har": "HAR regression",
              "har_vix": "HAR + VIX", "gbm": "Gradient boosting", "gbm_vix": "Gradient boosting + VIX",
              "mlp": "Neural net", "mlp_vix": "Neural net + VIX"}
    order = list(labels)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    panels = ((axes[0], (acc.loc[order, "vs_naive20"] * 100).to_numpy(), "Forecast error vs the 20-day rule\n(% better; higher is better)", "{:+.0f}%"),
              (axes[1], strat.loc[order, "d_sharpe_vs_naive20"].to_numpy(), "Strategy Sharpe vs the 20-day rule\n(change; higher is better)", "{:+.3f}"))
    for ax, vals, title, fmt in panels:
        ax.barh([labels[o] for o in order], vals, color=BLUE, height=0.6)
        ax.axvline(0, color=AXIS, lw=1)
        span = max(abs(vals).max(), 1e-9)
        for i, v in enumerate(vals):
            ax.text(v + (0.03 if v >= 0 else -0.03) * span, i, fmt.format(v), va="center", fontsize=8.5, color=INK,
                    ha="left" if v >= 0 else "right")
        ax.set_xlim(-span * 1.4, span * 1.4)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Better volatility forecasts did not make the strategy better", x=0.01, ha="left", fontsize=13,
                 fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "volforecast.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "volforecast.png")


def main():
    end = date.today().isoformat()

    # 1 & 2: SPY since 1993, buy-and-hold vs volatility targeting vs 200-day trend filter.
    spy = get_bars("yahoo", "SPY", "1993-01-01", end)
    w = 250
    curves = {
        "SPY buy & hold": spy["close"].pct_change().fillna(0).iloc[w:],
        "Vol target (10%)": position_returns(spy["close"], VolTarget(0.10).signals(spy), 1.0).iloc[w:],
        "Trend 200-day": position_returns(spy["close"], TrendFilter(200).signals(spy), 1.0).iloc[w:],
    }
    colors = {"SPY buy & hold": BENCH, "Vol target (10%)": BLUE, "Trend 200-day": ORANGE}
    equity_chart(curves, colors, "Volatility targeting gives up return to avoid the worst crashes",
                 "SPY, 1994-2026, after 1 bp trading costs. Cash earns 0%.", "equity_spy_long.png")
    drawdown_chart(curves, colors, "The crashes are where the risk controls earn their keep",
                   "Same strategies as above. The worst drop falls from -55% to -34% and -28%.",
                   "drawdown_spy_long.png")

    # 3: ETF rotation vs benchmarks, holdout shaded.
    closes = load_universe("yahoo", "2003-01-01", end)
    bm = benchmarks(closes)
    rot = rotation_returns(closes, lookback_m=3, top_k=8, skip_m=0, abs_filter=True, cost_bps=5.0)
    rc = {"SPY buy & hold": bm["SPY buy&hold"].iloc[ROT_WARMUP:], "60/40 SPY/IEF": bm["60/40 SPY/IEF"].iloc[ROT_WARMUP:],
          "ETF rotation (3m, top 8, dual)": rot.iloc[ROT_WARMUP:]}
    n = len(rc["SPY buy & hold"])
    holdout_start = rc["SPY buy & hold"].index[int(n * 0.75)]
    rcol = {"SPY buy & hold": BENCH, "60/40 SPY/IEF": AQUA, "ETF rotation (3m, top 8, dual)": BLUE}
    equity_chart(rc, rcol, "ETF rotation tracked SPY with smaller crashes, then lagged in the holdout",
                 "23 ETFs, monthly rebalance, 5 bps per trade. Picked using the earlier 75% of data only.",
                 "equity_rotation.png", shade_from=holdout_start)

    # 4: luck test.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, tag, label in ((axes[0], "sweep", "336-variation sweep\n(10 stocks/ETFs, 2020-26)"),
                           (axes[1], "sweep_rotation", "ETF rotation search\n(90 variations, 2004-26)")):
        null = pd.read_csv(DATA_DIR / tag / "null_reps.csv")["best_search"]
        real = json.loads((DATA_DIR / tag / "real_summary.json").read_text())["summary"]["best_search"]
        ax.grid(False)
        ax.hist(null, bins=18, color=BLUE_STEP, edgecolor=SURFACE, linewidth=2)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.14)
        ax.axvline(real, color=ORANGE, lw=2)
        ax.text(real, top * 1.15, f"real best {real:.2f}", color=INK, fontsize=9, va="bottom", ha="center")
        p = ((null >= real).sum() + 1) / (len(null) + 1)
        ax.set_title(f"{label}\nluck alone gets this high {p:.0%} of the time", loc="left", fontsize=10,
                     color=INK, pad=30)
        ax.set_xlabel("Best Sharpe ratio found on fake, pattern-free prices")
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_yticks([])
    fig.suptitle("Searching fake data finds winners as good as the real one", x=0.01, ha="left", fontsize=13,
                 fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "luck_test.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "luck_test.png")

    # 5: Sharpe by ticker, decades of data.
    cmp = pd.read_csv(DATA_DIR / "compare_results_yahoo.csv")
    piv = cmp.pivot(index="symbol", columns="strategy", values="sharpe")
    bh = cmp.groupby("symbol")["bh_sharpe"].first()
    order = bh.sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.9))
    y = np.arange(len(order))
    for i, s in enumerate(order):
        vals = [bh[s], piv.loc[s, "vol_target_10"], piv.loc[s, "trend_200"], piv.loc[s, "ml_logreg"]]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(bh[order], y, marker="D", s=46, color=BENCH, label="Buy & hold", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(piv.loc[order, "trend_200"], y, s=60, color=ORANGE, label="Trend 200-day", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(piv.loc[order, "vol_target_10"], y, s=60, color=BLUE, label="Vol target", zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(piv.loc[order, "ml_logreg"], y, s=60, color=AQUA, label="ML (logistic regression)", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe ratio (higher is better)")
    ax.legend(loc="lower right")
    finish(ax, "Volatility targeting beat buy-and-hold on 9 of 10 tickers; ML on none",
           "Daily data from each fund's start (1993-2000) through 2026, after 1 bp trading costs.", "sharpe_by_ticker.png", fig)
    crosssec_charts()
    riskstudy_charts()
    volforecast_chart()
    regime_chart()
    globalstudy_chart()


if __name__ == "__main__":
    main()
