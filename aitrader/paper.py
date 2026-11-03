"""Paper trader: run a strategy against Alpaca PAPER trading (fake money, real prices).

The client is hard-wired to paper=True; there is deliberately no live-trading option.
Run once per trading day AFTER the close (about 4:30pm ET) so the last daily bar is complete. Orders placed after
the close wait for the next open, so the fill is one session after the signal, like the backtests assume.

`run_forward` runs a FROZEN forward-test track. Track "vol" is volatility targeting on SPY (the original); track "trend"
is the 200-day trend rule on SPY (added later, with its own spec and log). Each rule is written to its own spec file on
first use and the command refuses to run if that rule in the code ever differs, so a track cannot be quietly tuned.
A better idea gets a new track name and a separate log instead.
"""
import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import DATA_DIR, alpaca_keys
from .data import fetch_bars
from .strategy import Strategy, TrendFilter, VolTarget

LOG_PATH = DATA_DIR / "paper_log.csv"
FORWARD_LOG = DATA_DIR / "forward_log.csv"
SPEC_PATH = DATA_DIR / "forward_test_spec.json"
FROZEN_RULE = {"track": "vol_target_spy_v1", "symbol": "SPY", "strategy": "vol_target",
               "target_vol": 0.10, "window": 20, "cap": 1.0}
FORWARD_LOG_TREND = DATA_DIR / "forward_log_trend200.csv"
SPEC_PATH_TREND = DATA_DIR / "forward_test_spec_trend200.json"
# Traded in IVV (the same S&P 500 index as SPY) so the two tracks hold separate positions: Alpaca keeps one net
# position per symbol, and each track sizes itself from the shares it sees in the account.
TREND_RULE = {"track": "trend200_ivv_v1", "symbol": "IVV", "strategy": "trend200", "window": 200}
TRACKS = {
    "vol": {"rule": FROZEN_RULE, "log": FORWARD_LOG, "spec": SPEC_PATH, "label": "Vol target (forward)",
            "make": lambda r: VolTarget(r["target_vol"], r["window"])},
    "trend": {"rule": TREND_RULE, "log": FORWARD_LOG_TREND, "spec": SPEC_PATH_TREND, "label": "200-day trend (forward)",
              "make": lambda r: TrendFilter(r["window"])},
}
BAND = 0.02  # ignore rebalances smaller than 2% of the allocation (avoids daily noise trades)
LOG_FIELDS = ["time_utc", "symbol", "strategy", "params", "last_bar", "price", "weight", "desired_qty", "held_qty",
              "action", "qty", "order_id", "dry_run", "equity"]


def plan_trade(weight: float, price: float, held: float, notional: float, band: float = BAND) -> dict:
    """Turn a target weight (0-1 of `notional`) into an action. Fractional shares; full exits always go through."""
    desired = round(notional * min(max(weight, 0.0), 1.0) / price, 2)
    if desired == 0 and held > 0:
        return {"action": "sell", "qty": held, "desired": 0.0, "close_all": True}
    delta = round(desired - held, 2)
    if abs(delta) * price >= band * notional:
        return {"action": "buy" if delta > 0 else "sell", "qty": abs(delta), "desired": desired, "close_all": False}
    return {"action": "hold", "qty": 0.0, "desired": desired, "close_all": False}


def log_row(row: dict, path=LOG_PATH):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def already_ran(path, symbol: str, strategy: str, last_bar) -> bool:
    """True if a real (non-dry-run) decision was already logged for this bar, so a re-run cannot double-trade."""
    if not path.exists():
        return False
    log = pd.read_csv(path)
    real = log[(log["dry_run"].astype(str) == "False") & (log["symbol"] == symbol) & (log["strategy"] == strategy)]
    return str(pd.Timestamp(last_bar).date()) in set(pd.to_datetime(real["last_bar"]).dt.date.astype(str))


def run_once(symbol: str, strategy: Strategy, notional: float = 1000.0, dry_run: bool = False,
             lookback_days: int = 800, log_path=LOG_PATH, band: float = BAND) -> dict:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    client = TradingClient(*alpaca_keys(), paper=True)
    if client.get_clock().is_open:
        raise RuntimeError("The market is open, so today's daily bar is incomplete. Run after the 4:00pm ET close.")
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    df = fetch_bars(symbol, start)
    weight = float(np.clip(strategy.signals(df).iloc[-1], 0.0, 1.0))
    price, last_bar = float(df["close"].iloc[-1]), df.index[-1]

    if not dry_run and already_ran(log_path, symbol, strategy.name, last_bar):
        return {"action": "skip", "reason": f"already ran for the {last_bar.date()} bar"}

    held = next((float(p.qty) for p in client.get_all_positions() if p.symbol == symbol), 0.0)
    plan = plan_trade(weight, price, held, notional, band)
    order_id = ""
    if plan["action"] != "hold" and not dry_run:
        if plan["close_all"]:
            order_id = str(client.close_position(symbol).id)
        else:
            side = OrderSide.BUY if plan["action"] == "buy" else OrderSide.SELL
            order_id = str(client.submit_order(MarketOrderRequest(symbol=symbol, qty=plan["qty"], side=side,
                                                                  time_in_force=TimeInForce.DAY)).id)
    row = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": symbol,
        "strategy": strategy.name, "params": json.dumps(strategy.params), "last_bar": last_bar.date(),
        "price": price, "weight": round(weight, 4), "desired_qty": plan["desired"], "held_qty": held,
        "action": plan["action"], "qty": plan["qty"], "order_id": order_id, "dry_run": dry_run,
        "equity": client.get_account().equity,
    }
    log_row(row, log_path)
    return row


def ensure_spec(track: str = "vol") -> None:
    """Pin the track's rule on first use; refuse to run if the code's rule has since changed."""
    t = TRACKS[track]
    if not t["spec"].exists():
        t["spec"].write_text(json.dumps({"rule": t["rule"], "frozen_on": datetime.now().date().isoformat()}, indent=2))
        return
    saved = json.loads(t["spec"].read_text())["rule"]
    if saved != t["rule"]:
        raise RuntimeError(f"The forward-test rule was changed. Pinned: {saved}. In code: {t['rule']}. "
                           "Revert the change, or give the new rule a new track name and a separate log.")


def run_forward(notional: float = 10_000.0, dry_run: bool = False, track: str = "vol") -> dict:
    ensure_spec(track)
    t = TRACKS[track]
    return run_once(t["rule"]["symbol"], t["make"](t["rule"]), notional, dry_run, log_path=t["log"])


def forward_report(track: str = "vol") -> str:
    """Forward-test results from the logged weights vs SPY buy & hold, same 1 bp cost as the backtests."""
    t = TRACKS[track]
    rule, log_path = t["rule"], t["log"]
    if not log_path.exists():
        return f"No log yet for track '{rule['track']}'. Run: python main.py forward --track {track}"
    log = pd.read_csv(log_path)
    log = log[log["dry_run"].astype(str) == "False"].drop_duplicates("last_bar", keep="last")
    if log.empty:
        return f"Only dry runs so far for '{rule['track']}'. Run: python main.py forward --track {track} (without --dry-run)"
    w = log.set_index(pd.to_datetime(log["last_bar"]))["weight"].sort_index()
    sym = rule["symbol"]
    px = fetch_bars(sym, (w.index[0] - timedelta(days=7)).date().isoformat())["close"]
    pos = w.reindex(px.index).ffill().shift(1)  # weight decided at a close applies from the next bar
    ret = px.pct_change()
    live = pos.notna() & (px.index > w.index[0])
    pos, ret = pos[live], ret[live]
    strat = pos * ret - pos.diff().abs().fillna(pos.abs()) * 1e-4
    if len(strat) < 2:
        return f"Forward test started {w.index[0].date()}; need at least 2 trading days after that for results."

    def line(name, r):
        eq = (1 + r).cumprod()
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float("nan")
        return f"  {name:<24}{eq.iloc[-1] - 1:>9.2%}{sh:>9.2f}{(eq / eq.cummax() - 1).min():>10.2%}"
    return "\n".join([
        f"Forward test '{rule['track']}': {len(strat)} trading days after {w.index[0].date()} "
        f"(through {strat.index[-1].date()})", f"  {'':<24}{'return':>9}{'Sharpe':>9}{'max drop':>10}",
        line(t["label"], strat), line(f"{sym} buy & hold", ret),
        "", "Fewer than ~250 trading days is far too short to judge. Judge a risk-control rule by its max drop "
            "during a down period, not by its return in a rising market."])
