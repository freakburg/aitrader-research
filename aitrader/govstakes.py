"""US government equity stakes in companies: a sourced list, and how the listed stocks have done since each announcement.

The list is compiled by hand from reporting and filings (CNBC 7 Feb 2026, a US News/WTOP piece 21 Jun 2026 citing the
Council on Foreign Relations deal tracker, a 31 May 2026 newsletter that cites SEC filings and a NIST release, and a
Trilogy Metals SEC release dated 11 Sep 2026). The sources disagree on counts and some dates, so every row carries a
confidence level and the sources behind it. It is NOT a complete or official register, and deals after the newest source
are not covered.

Performance uses the close on the announcement date as the starting point (the first trading day on or after it), with
dividend-adjusted Yahoo Finance prices, and compares each stock with SPY over exactly the same dates. Stocks whose
announcement date the sources do not give are left out of that comparison rather than guessed.

    python main.py govstakes        # writes data/govstakes/{stakes,prices,summary}.json
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import get_bars

OUT_DIR = DATA_DIR / "govstakes"
INTEL_SHARES, INTEL_PRICE_PAID = 433.3e6, 20.47  # both reported identically by CNBC and the US News/WTOP piece

CNBC = "https://www.cnbc.com/2026/02/07/trump-administration-equity-portfolio-grows-these-are-investments-so-far.html"
WTOP = "https://wtop.com/news/2026/06/uncle-sam-is-buying-stocks-should-investors-follow/"
LETTER = "https://moeonmargin.substack.com/p/the-us-government-is-now-a-shareholder"
TRILOGY = "https://www.sec.gov/Archives/edgar/data/0001543418/000110465926107442/tm2624736d1_ex99-1.htm"

# ref_date: announcement date used for the performance comparison (None when the sources do not give one).
STAKES = [
    dict(company="Intel", ticker="INTC", listed=True, agency="Commerce Dept (reports differ)", announced="Aug 2025",
         ref_date="2025-08-22", status="Agreed", instrument="Common stock, 433.3M shares at $20.47",
         stake="About 9.9-10%", amount="About $8.9B (433.3M x $20.47)", purpose="Chipmaking; converted CHIPS Act grants",
         confidence="High", sources="CNBC Feb 2026; WTOP Jun 2026; newsletter May 2026", url=CNBC),
    dict(company="MP Materials", ticker="MP", listed=True, agency="Defense Dept", announced="Jul 2025",
         ref_date="2025-07-10", status="Agreed", instrument="Convertible preferred plus warrants; 10-year price floor and purchase deal",
         stake="About 15% if converted (company statement)", amount="$400M", purpose="Rare earth magnets and supply",
         confidence="High", sources="CNBC Feb 2026; WTOP Jun 2026; newsletter May 2026", url=CNBC),
    dict(company="Lithium Americas", ticker="LAC", listed=True, agency="Energy Dept", announced="2025 (date not given)",
         ref_date=None, status="Agreed", instrument="Equity/warrants tied to a $2.3B federal loan (instrument not specified)",
         stake="5% of company plus 5% of the Thacker Pass venture with GM", amount="Not stated",
         purpose="Lithium mine in Nevada", confidence="High", sources="CNBC Feb 2026; WTOP Jun 2026; newsletter May 2026", url=CNBC),
    dict(company="Trilogy Metals", ticker="TMQ", listed=True, agency="Dept of War per SEC release (news said White House)",
         announced="Oct 2025; closed 11 Sep 2026", ref_date=None, status="Closed",
         instrument="Shares plus a 10-year option over shares held by South32 (early reports said warrants)",
         stake="About 10%", amount="US$35.6M", purpose="Alaska copper-zinc project and Ambler Road",
         confidence="High", sources="Company SEC release Sep 2026; CNBC Feb 2026; WTOP Jun 2026", url=TRILOGY),
    dict(company="USA Rare Earth", ticker="USAR", listed=True, agency="Commerce Dept", announced="26 Jan 2026",
         ref_date="2026-01-26", status="Letter of intent", instrument="16.1M common shares plus about 17.6M warrants; $1.3B loan",
         stake="8-16% depending on warrants", amount="$277M funding plus $1.3B loan", purpose="Rare earth magnets, Oklahoma and Texas",
         confidence="Medium", sources="CNBC Feb 2026; WTOP Jun 2026; newsletter May 2026", url=CNBC),
    dict(company="L3Harris (missile unit)", ticker="LHX", listed=True, agency="Pentagon", announced="Dec 2025 to Jan 2026 (sources differ)",
         ref_date="2026-01-13", status="Announced", instrument="$1B convertible preferred in the rocket-motor unit; converts at a planned IPO",
         stake="Not stated", amount="$1B", purpose="Missile and rocket motors", confidence="Medium",
         sources="CNBC Feb 2026; WTOP Jun 2026; newsletter May 2026", url=WTOP,
         note="Later status of the planned unit IPO not checked."),
    dict(company="Westinghouse", ticker="", listed=False, agency="Commerce Dept", announced="Oct 2025", ref_date=None,
         status="Agreed (conditional)", instrument="Participation interest; IPO option if value passes $30B",
         stake="Conditional: about 8% in one scenario, up to 20% option per another source", amount="Government cash not stated",
         purpose="Financing for nuclear plants (owned by Cameco and Brookfield)", confidence="Medium",
         sources="CNBC Feb 2026; newsletter May 2026", url=CNBC),
    dict(company="Vulcan Elements", ticker="", listed=False, agency="Commerce (equity) and Pentagon (warrants, loan)",
         announced="Date not given", ref_date=None, status="Agreed", instrument="$50M equity plus warrants; $620M Pentagon loan",
         stake="Not stated", amount="$50M equity", purpose="Rare earth magnet supply chain", confidence="Medium",
         sources="CNBC Feb 2026; WTOP Jun 2026", url=CNBC),
    dict(company="xLight", ticker="", listed=False, agency="Commerce Dept", announced="Dec 2025", ref_date=None,
         status="Letter of intent", instrument="Equity (type not stated)", stake="Not stated", amount="Up to $150M",
         purpose="Lasers for chip manufacturing", confidence="Medium", sources="CNBC Feb 2026; newsletter May 2026", url=CNBC),
    dict(company="U.S. Steel", ticker="", listed=False, agency="Federal government", announced="Jun 2025", ref_date=None,
         status="Agreed", instrument="Golden share (veto rights)", stake="No economic stake", amount="None",
         purpose="Conditions on the Nippon Steel acquisition; stock no longer trades", confidence="High",
         sources="CNBC Feb 2026; newsletter May 2026", url=CNBC),
    dict(company="IBM (quantum foundry unit)", ticker="IBM", listed=True, agency="Commerce Dept", announced="21 May 2026",
         ref_date="2026-05-21", status="Letter of intent", instrument="Minority equity stake in a quantum foundry",
         stake="Minority", amount="About $1B CHIPS incentives", purpose="Quantum chip manufacturing", confidence="Medium",
         sources="WTOP Jun 2026 (IBM named); newsletter May 2026 (details)", url=LETTER),
    dict(company="GlobalFoundries", ticker="GFS", listed=True, agency="Commerce Dept", announced="21 May 2026",
         ref_date="2026-05-21", status="Letter of intent", instrument="Minority equity stake", stake="Minority",
         amount="$375M", purpose="Chip manufacturing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="D-Wave Quantum", ticker="QBTS", listed=True, agency="Commerce Dept", announced="21 May 2026",
         ref_date="2026-05-21", status="Letter of intent", instrument="Minority equity stake", stake="Minority",
         amount="About $100M", purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="Rigetti Computing", ticker="RGTI", listed=True, agency="Commerce Dept", announced="21 May 2026",
         ref_date="2026-05-21", status="Letter of intent", instrument="Minority equity stake", stake="Minority",
         amount="About $100M", purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="Infleqtion", ticker="INFQ", listed=True, agency="Commerce Dept", announced="21 May 2026",
         ref_date="2026-05-21", status="Letter of intent", instrument="Minority equity stake", stake="Minority",
         amount="About $100M", purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="Atom Computing", ticker="", listed=False, agency="Commerce Dept", announced="21 May 2026", ref_date=None,
         status="Letter of intent", instrument="Minority equity stake", stake="Minority", amount="About $100M",
         purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="PsiQuantum", ticker="", listed=False, agency="Commerce Dept", announced="21 May 2026", ref_date=None,
         status="Letter of intent", instrument="Minority equity stake", stake="Minority", amount="About $100M",
         purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="Quantinuum", ticker="", listed=False, agency="Commerce Dept", announced="21 May 2026", ref_date=None,
         status="Letter of intent", instrument="Minority equity stake", stake="Minority", amount="About $100M",
         purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="Diraq", ticker="", listed=False, agency="Commerce Dept", announced="21 May 2026", ref_date=None,
         status="Letter of intent", instrument="Minority equity stake", stake="Minority", amount="About $100M",
         purpose="Quantum computing", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
    dict(company="ReElement Technologies", ticker="", listed=False, agency="Not stated", announced="Date not given",
         ref_date=None, status="Reported", instrument="Warrants (terms not given)", stake="Not stated", amount="Not stated",
         purpose="Rare earth processing", confidence="Low", sources="WTOP Jun 2026 (named); newsletter May 2026", url=WTOP),
    dict(company="Korea Zinc", ticker="", listed=False, agency="Not stated", announced="Date not given", ref_date=None,
         status="Reported", instrument="Warrants (terms not given)", stake="Not stated", amount="Not stated",
         purpose="Metals", confidence="Low", sources="Newsletter May 2026 only", url=LETTER),
]
COUNTS_DISAGREE = [
    {"source": "CNBC, 7 Feb 2026", "reports": "At least 10 companies", "url": CNBC},
    {"source": "US News via WTOP, 21 Jun 2026 (citing the CFR deal tracker)", "reports": "17 deals worth about $20.9B; names only five", "url": WTOP},
    {"source": "Newsletter, 31 May 2026 (secondary source)", "reports": "26 deals worth $23.9B, but names about 21", "url": LETTER},
    {"source": "Trilogy Metals SEC release, 11 Sep 2026 (primary)", "reports": "One deal: about a 10% stake for US$35.6M, closed", "url": TRILOGY},
]


def ref_index(px: pd.Series, ref: str) -> int:
    """Position of the first trading day on or after `ref`."""
    return int(px.index.searchsorted(pd.Timestamp(ref)))


def performance_row(s: dict, px: pd.Series, spy: pd.Series) -> dict | None:
    """Growth of the stock and of SPY from the announcement date to the latest close."""
    i = ref_index(px, s["ref_date"])
    if i >= len(px) - 1:
        return None
    d0 = px.index[i]
    j = spy.index.searchsorted(d0)
    stock_mult = float(px.iloc[-1] / px.iloc[i])
    spy_mult = float(spy.iloc[-1] / spy.iloc[j])
    return {"company": s["company"], "ticker": s["ticker"], "ref_date": str(d0.date()), "ref_price": round(float(px.iloc[i]), 2),
            "last_date": str(px.index[-1].date()), "last_price": round(float(px.iloc[-1]), 2),
            "multiple": round(stock_mult, 4), "spy_multiple": round(spy_mult, 4),
            "return_pct": round((stock_mult - 1) * 100, 1), "spy_return_pct": round((spy_mult - 1) * 100, 1),
            "excess_pct": round((stock_mult - spy_mult) * 100, 1), "beat_spy": bool(stock_mult > spy_mult),
            "ref_basis": "Close on the announcement date, or the next trading day"}


def build(end: str | None = None) -> dict:
    end = end or date.today().isoformat()
    spy = get_bars("yahoo", "SPY", "2025-01-01", end)["close"]
    rows, skipped = [], []
    for s in STAKES:
        if not (s["listed"] and s["ref_date"]):
            continue
        try:
            px = get_bars("yahoo", s["ticker"], "2025-01-01", end)["close"]
            r = performance_row(s, px, spy)
            (rows if r else skipped).append(r or s["ticker"])
        except Exception as exc:
            skipped.append(f"{s['ticker']} ({type(exc).__name__})")
    prices = pd.DataFrame(rows).sort_values("multiple", ascending=False).to_dict("records") if rows else []
    intel = next((r for r in prices if r["ticker"] == "INTC"), None)
    summary = {
        "tracked": len(STAKES), "listed": sum(s["listed"] for s in STAKES),
        "private_or_other": sum(not s["listed"] for s in STAKES),
        "high_confidence": sum(s["confidence"] == "High" for s in STAKES),
        "compared": len(prices), "beat_spy": sum(r["beat_spy"] for r in prices),
        "as_of": prices[0]["last_date"] if prices else None,
        "newest_source": "2026-09-11",
        "intel_shares_m": round(INTEL_SHARES / 1e6, 1), "intel_paid_bn": round(INTEL_SHARES * INTEL_PRICE_PAID / 1e9, 1),
        "intel_value_bn": round(INTEL_SHARES * intel["last_price"] / 1e9, 1) if intel else None,
        "intel_last_price": intel["last_price"] if intel else None,
    }
    return {"stakes": STAKES, "prices": prices, "summary": [summary], "counts": COUNTS_DISAGREE, "skipped": skipped}


def run_govstakes() -> None:
    out = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_text(encoding="utf-8")
    meta = [{"label": "Prices", "value": "Yahoo Finance, dividend-adjusted closes (total return)"},
            {"label": "Starting point", "value": "Close on the announcement date, or the next trading day. Stakes whose "
                                                  "announcement date the sources do not give are left out."},
            {"label": "govstakes.py", "value": code, "type": "code", "lang": "python"}]
    (OUT_DIR / "stakes.json").write_text(json.dumps(out["stakes"], indent=1))
    (OUT_DIR / "counts.json").write_text(json.dumps(out["counts"], indent=1))
    (OUT_DIR / "prices.json").write_text(json.dumps({"rows": out["prices"], "meta": meta}, indent=1))
    (OUT_DIR / "summary.json").write_text(json.dumps({"rows": out["summary"], "meta": meta}, indent=1))
    s = out["summary"][0]
    print(f"{s['tracked']} stakes tracked ({s['listed']} in listed companies); prices as of {s['as_of']}.")
    print(f"{s['beat_spy']} of {s['compared']} compared stocks beat SPY since their announcement dates.")
    print(pd.DataFrame(out["prices"])[["ticker", "ref_date", "ref_price", "last_price", "return_pct", "spy_return_pct",
                                       "excess_pct"]].to_string(index=False))
    if s["intel_value_bn"]:
        print(f"Intel stake: ${s['intel_paid_bn']}B paid, ${s['intel_value_bn']}B at ${s['intel_last_price']}.")
    if out["skipped"]:
        print("Skipped (no usable price history):", out["skipped"])
    print(f"Files written to {OUT_DIR}")
