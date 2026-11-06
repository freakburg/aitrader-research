"""CLI. Examples:
  python main.py backtest --symbol SPY --start 2018-01-01 --strategy sma
  python main.py backtest --symbol SPY --start 2018-01-01 --strategy ml
  python main.py walkforward --symbol SPY --start 2015-01-01
  python main.py paper --symbol SPY --strategy sma --dry-run
  python main.py gaps-record --symbols SPY IVV --minutes 30
  python main.py gaps-analyze --symbols SPY IVV --start 2024-06-03 --end 2024-06-07   # minute bars
  python main.py gaps-analyze --csv data/quotes_SPY_IVV.csv                           # recorded quotes
Add --synthetic to any backtest/walkforward command to run offline on fake data.
"""
import argparse

import pandas as pd

from aitrader import gap
from aitrader.backtest import run_backtest
from aitrader.data import fetch_bars, get_bars, synthetic_bars
from aitrader.ml_strategy import MLStrategy
from aitrader.strategy import SMACrossover, TrendFilter, VolTarget
from aitrader.walkforward import walk_forward

pd.set_option("display.width", 160)


def load(args):
    return synthetic_bars() if args.synthetic else get_bars(args.source, args.symbol, args.start, args.end)


def build_strategy(args):
    if args.strategy == "ml":
        return MLStrategy(model=args.model)
    if args.strategy == "trend":
        return TrendFilter()
    if args.strategy == "vol":
        return VolTarget()
    return SMACrossover(args.fast, args.slow)


def show(metrics: dict):
    pct = {"total_return", "cagr", "max_drawdown", "exposure", "buy_hold_return", "buy_hold_max_drawdown"}
    for k, v in metrics.items():
        print(f"  {k:<24}{v:>10.1%}" if k in pct else f"  {k:<24}{v:>10.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--symbol", default="SPY")
        sp.add_argument("--start", default="2018-01-01")
        sp.add_argument("--end", default=None)
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--source", choices=["alpaca", "yahoo"], default="alpaca")
        sp.add_argument("--cost-bps", type=float, default=1.0)

    def strat(sp):
        sp.add_argument("--strategy", choices=["sma", "ml", "trend", "vol"], default="sma")
        sp.add_argument("--fast", type=int, default=20)
        sp.add_argument("--slow", type=int, default=50)
        sp.add_argument("--model", choices=["logreg", "rf"], default="logreg")

    bt = sub.add_parser("backtest"); common(bt); strat(bt)
    wf = sub.add_parser("walkforward"); common(wf)
    wf.add_argument("--train-bars", type=int, default=504)
    wf.add_argument("--test-bars", type=int, default=63)
    cp = sub.add_parser("compare")
    cp.add_argument("--symbols", nargs="+", default=None)
    cp.add_argument("--start", default="2018-01-01")
    cp.add_argument("--source", choices=["alpaca", "yahoo"], default="alpaca")
    cp.add_argument("--cost-bps", type=float, default=1.0)
    pp = sub.add_parser("paper"); strat(pp)
    pp.add_argument("--symbol", default="SPY")
    pp.add_argument("--notional", type=float, default=1000.0)
    pp.add_argument("--dry-run", action="store_true")
    fw = sub.add_parser("forward", help="frozen forward tests on an Alpaca PAPER account: vol target (SPY) and 200-day trend (IVV)")
    fw.add_argument("--notional", type=float, default=10_000.0, help="dollars allocated to EACH track")
    fw.add_argument("--dry-run", action="store_true", help="compute and log the decision without placing an order")
    fw.add_argument("--track", choices=["vol", "trend", "all"], default="all")
    fr = sub.add_parser("forward-report", help="forward-test results vs buy & hold")
    fr.add_argument("--track", choices=["vol", "trend", "all"], default="all")
    gr = sub.add_parser("gaps-record")
    gr.add_argument("--symbols", nargs=2, default=["SPY", "IVV"])
    gr.add_argument("--minutes", type=float, default=30)
    gr.add_argument("--interval", type=float, default=1.0)
    gr.add_argument("--ignore-clock", action="store_true", help="record even if the market is closed")
    sw = sub.add_parser("sweep", help="mass strategy search + luck test (long-running, resumable)")
    sw.add_argument("--symbols", nargs="+", default=None)
    sw.add_argument("--start", default="2018-01-01")
    sw.add_argument("--source", choices=["alpaca", "yahoo"], default="alpaca")
    sw.add_argument("--max-hours", type=float, default=8.0)
    sw.add_argument("--null-reps", type=int, default=100)
    sw.add_argument("--workers", type=int, default=None)
    sw.add_argument("--cost-bps", type=float, default=1.0)
    sw.add_argument("--quick", action="store_true", help="tiny version to test the pipeline")
    sw.add_argument("--fresh", action="store_true", help="discard saved results and start over")
    ro = sub.add_parser("rotation", help="ETF momentum rotation study with holdout + luck test")
    ro.add_argument("--source", choices=["alpaca", "yahoo"], default="yahoo")
    ro.add_argument("--start", default="2003-01-01")
    ro.add_argument("--reps", type=int, default=200)
    ro.add_argument("--cost-bps", type=float, default=5.0)
    ro.add_argument("--fresh", action="store_true")
    cs = sub.add_parser("crosssec", help="pooled-stock cross-sectional models (momentum, ridge, gradient boosting)")
    cs.add_argument("--start", default="1993-01-01")
    cs.add_argument("--refresh", action="store_true", help="re-download the stock universe")
    cs.add_argument("--top-frac", type=float, default=0.2)
    cs.add_argument("--cost-bps", type=float, default=10.0)
    cs.add_argument("--reps", type=int, default=1000)
    cs.add_argument("--neutral", action="store_true", help="beta-neutral check: does signal survive hedging out the market?")
    cs.add_argument("--mlp", action="store_true", help="with --neutral: run the small neural-net variants (plus a reference)")
    rs = sub.add_parser("riskstudy", help="vol targeting / trend / risk parity across asset classes, with cash interest and leverage")
    rs.add_argument("--reps", type=int, default=500)
    vf = sub.add_parser("volforecast", help="does better volatility forecasting improve volatility targeting?")
    vf.add_argument("--reps", type=int, default=500)
    vf.add_argument("--workers", type=int, default=None)
    mv = sub.add_parser("minervini", help="Minervini-style trend-template stock portfolio vs buy-and-hold + luck test")
    mv.add_argument("--reps", type=int, default=200)
    mv.add_argument("--cost-bps", type=float, default=10.0)
    mv.add_argument("--baseline", action="store_true", help="only compare against the SPY-200d-filtered basket")
    mv.add_argument("--rsp", action="store_true", help="only run the RSP survivorship check on the filtered basket")
    rg = sub.add_parser("regime", help="regime-switching overlay on buy and hold (Markov-switching model)")
    rg.add_argument("--workers", type=int, default=None)
    gl = sub.add_parser("globalstudy", help="frozen vol-targeting rule across eras, countries and asset types")
    gl.add_argument("--reps", type=int, default=300)
    sub.add_parser("stockstudy", help="individual big stocks vs an index fund (rolling windows, random baskets, past mega-caps)")
    sr = sub.add_parser("sweep-report", help="summarize saved sweep results")
    sr.add_argument("--source", choices=["alpaca", "yahoo"], default="alpaca")
    gs = sub.add_parser("gaps-session", help="unattended: wait for open, record full sessions for N days")
    gs.add_argument("--symbols", nargs=2, default=["SPY", "IVV"])
    gs.add_argument("--days", type=int, default=1)
    gs.add_argument("--interval", type=float, default=1.0)
    ga = sub.add_parser("gaps-analyze")
    ga.add_argument("--symbols", nargs=2, default=["SPY", "IVV"])
    ga.add_argument("--csv")
    ga.add_argument("--start", default="2024-06-03")
    ga.add_argument("--end", default="2024-06-07")
    ga.add_argument("--half-spread-bps", type=float, default=0.5)
    ga.add_argument("--hold", type=int, default=5)
    ga.add_argument("--fee-bps", type=float, default=0.1)

    args = p.parse_args()

    if args.cmd == "backtest":
        df = load(args)
        res = run_backtest(df, build_strategy(args), args.cost_bps)
        print(f"{args.symbol} {args.strategy} backtest, {len(df)} bars:")
        show(res.metrics)
    elif args.cmd == "walkforward":
        df = load(args)
        grid = {"fast": [5, 10, 20, 30], "slow": [40, 50, 100, 150, 200]}
        res = walk_forward(df, SMACrossover, grid, args.train_bars, args.test_bars, args.cost_bps)
        print(f"{args.symbol} walk-forward SMA (out-of-sample only):")
        show(res.metrics)
        print("\nChosen parameters per window:")
        print(res.params_log.to_string(index=False))
    elif args.cmd == "compare":
        from datetime import date
        from aitrader.compare import DEFAULT_SYMBOLS, compare
        end = date.today().isoformat()
        frames = {s: get_bars(args.source, s, args.start, end) for s in (args.symbols or DEFAULT_SYMBOLS)}
        table, summary = compare(frames, args.cost_bps)
        table.to_csv(f"data/compare_results_{args.source}.csv", index=False)
        print({s: f"{f.index[0].date()}..{f.index[-1].date()} ({len(f)} bars)" for s, f in frames.items()})
        print(table.drop(columns=["beats_bh_sharpe", "beats_bh_return", "smaller_dd"]).round(3).to_string(index=False))
        print("\nSummary across tickers (counts are out of `tickers`):")
        print(summary.round(3).to_string())
    elif args.cmd == "paper":
        from aitrader.paper import run_once
        print(run_once(args.symbol, build_strategy(args), args.notional, args.dry_run))
    elif args.cmd == "forward":
        from aitrader.paper import TRACKS, run_forward
        for tr in (list(TRACKS) if args.track == "all" else [args.track]):
            print(f"[{TRACKS[tr]['rule']['track']}]", run_forward(args.notional, args.dry_run, tr))
    elif args.cmd == "forward-report":
        from aitrader.paper import TRACKS, forward_report
        for tr in (list(TRACKS) if args.track == "all" else [args.track]):
            print(forward_report(tr) + "\n")
    elif args.cmd == "gaps-record":
        print("Recording to", gap.record_quotes(*args.symbols, args.minutes, args.interval,
                                                require_open=not args.ignore_clock))
    elif args.cmd == "sweep":
        from datetime import date
        from aitrader.compare import DEFAULT_SYMBOLS
        from aitrader.sweep import run_sweep
        syms = args.symbols or (DEFAULT_SYMBOLS[:4] if args.quick else DEFAULT_SYMBOLS)
        end = date.today().isoformat()
        closes = pd.DataFrame({s: get_bars(args.source, s, args.start, end)["close"] for s in syms}).dropna()
        run_sweep(closes, args.max_hours, 2 if args.quick else args.null_reps, args.workers,
                  args.cost_bps, args.quick, args.fresh, tag="" if args.source == "alpaca" else args.source)
    elif args.cmd == "rotation":
        from aitrader.rotation import run_rotation
        run_rotation(args.source, args.start, args.reps, args.cost_bps, args.fresh)
    elif args.cmd == "crosssec":
        from aitrader.crosssec import run_crosssec, run_neutral
        if args.neutral:
            only = ["gbm_resid", "mlp_raw", "mlp_resid", "mlp_resid_norisk"] if args.mlp else None
            run_neutral(args.start, args.top_frac, args.cost_bps, args.reps, only)
        else:
            run_crosssec(args.start, args.refresh, args.top_frac, args.cost_bps, args.reps)
    elif args.cmd == "riskstudy":
        from aitrader.riskstudy import run_risk_study
        run_risk_study(args.reps)
    elif args.cmd == "volforecast":
        from aitrader.volforecast import run_vol_study
        run_vol_study(args.reps, args.workers)
    elif args.cmd == "minervini":
        from aitrader.minervini import run_filter_baseline, run_minervini, run_rsp_check
        if args.rsp:
            run_rsp_check(args.cost_bps)
        elif args.baseline:
            run_filter_baseline(args.cost_bps)
        else:
            run_minervini(args.reps, args.cost_bps)
    elif args.cmd == "regime":
        from aitrader.regime import run_regime_study
        run_regime_study(args.workers)
    elif args.cmd == "globalstudy":
        from aitrader.globalstudy import run_global_study
        run_global_study(args.reps)
    elif args.cmd == "stockstudy":
        from aitrader.stockstudy import run_stock_study
        run_stock_study()
    elif args.cmd == "sweep-report":
        from aitrader.sweep import report
        print(report("" if args.source == "alpaca" else args.source))
    elif args.cmd == "gaps-session":
        gap.record_sessions(*args.symbols, days=args.days, interval=args.interval)
    elif args.cmd == "gaps-analyze":
        if args.csv:
            q = gap.load_quotes(args.csv)
        else:
            a, b = (fetch_bars(s, args.start, args.end, "1Min")["close"] for s in args.symbols)
            q = gap.quotes_from_bars(a, b, args.half_spread_bps)
        print(gap.sweep(q, hold=args.hold, fee_bps=args.fee_bps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
