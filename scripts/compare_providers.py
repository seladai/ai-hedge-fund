"""Compare data providers for completeness and reliability.

Runs each available provider (financialdatasets [free 5 tickers], yfinance, sec,
and fmp if FMP_API_KEY is set) over a set of tickers and reports:

  COMPLETENESS  - periods returned, history depth, required-field coverage
  RELIABILITY   - cross-source agreement on raw figures (revenue/net income/equity)
  USABILITY     - whether the Buffett agent's moat/consistency/intrinsic-value
                  functions actually compute on each source

Usage:
  poetry run python compare_providers.py                 # AAPL, KO
  poetry run python compare_providers.py AAPL MSFT KO    # custom tickers
"""
from __future__ import annotations

import datetime as dt
import os
import sys

# Allow running from scripts/: put the repo root (scripts/..) on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()  # pick up FMP_API_KEY / FINANCIAL_DATASETS_API_KEY from .env

from src.data.models import LineItem
from src.tools.providers import yfinance_provider, sec_provider, fmp_provider
from src.agents.warren_buffett import (
    analyze_fundamentals, analyze_consistency, analyze_moat,
    analyze_book_value_growth, calculate_intrinsic_value,
)

END = dt.date.today().isoformat()

# Raw line items the Buffett agent depends on (completeness check).
REQUIRED_LINE_ITEMS = [
    "revenue", "net_income", "shareholders_equity", "total_assets", "total_liabilities",
    "current_assets", "current_liabilities", "capital_expenditure",
    "depreciation_and_amortization", "outstanding_shares", "free_cash_flow",
]


def _fds(ticker, end, period, limit):
    """financialdatasets via the repo's native path (DATA_PROVIDER must be unset)."""
    os.environ.pop("DATA_PROVIDER", None)
    from importlib import reload
    import src.tools.api as api
    reload(api)
    return (api.get_financial_metrics(ticker, end, period, limit),
            api.search_line_items(ticker, REQUIRED_LINE_ITEMS, end, period, limit))


PROVIDERS = {
    "financialdatasets": None,  # special-cased
    "yfinance": yfinance_provider,
    "sec": sec_provider,
}
if os.environ.get("FMP_API_KEY"):
    PROVIDERS["fmp"] = fmp_provider


def _get(provider_name, module, ticker):
    if provider_name == "financialdatasets":
        return _fds(ticker, END, "ttm", 10)
    return (module.fetch_financial_metrics(ticker, END, "ttm", 10),
            module.fetch_line_items(ticker, REQUIRED_LINE_ITEMS, END, "ttm", 10))


def _field_coverage(items: list[LineItem]) -> tuple[int, int]:
    if not items:
        return 0, len(REQUIRED_LINE_ITEMS)
    latest = items[0]
    present = sum(1 for f in REQUIRED_LINE_ITEMS if getattr(latest, f, None) is not None)
    return present, len(REQUIRED_LINE_ITEMS)


def _fmt(v, pct=False):
    if v is None:
        return "  n/a"
    return f"{v*100:6.1f}%" if pct else f"{v:,.0f}"


def _print_dispersion(raw: dict):
    """Show max cross-source % deviation for each raw figure (reliability signal)."""
    for i, label in enumerate(["revenue", "net_income", "equity"]):
        vals = [v[i] for v in raw.values() if v[i] is not None]
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            spread = (hi - lo) / abs(hi) * 100 if hi else 0
            flag = "  <-- DISAGREE (likely period mismatch or mapping)" if spread > 10 else ""
            print(f"    {label:12s} cross-source spread: {spread:5.1f}%{flag}")


def analyze_ticker(ticker: str):
    print("\n" + "=" * 80)
    print(f"  {ticker}")
    print("=" * 80)
    rows = {}
    for name, module in PROVIDERS.items():
        try:
            rows[name] = _get(name, module, ticker)
        except Exception as e:
            print(f"{name:18s} ERROR: {e}")

    print("\n-- COMPLETENESS (periods, history depth, latest-period field coverage) --")
    print(f"{'provider':18s} {'periods':>7s} {'latest':>12s} {'oldest':>12s} {'fields':>8s}")
    for name, (metrics, line_items) in rows.items():
        latest = metrics[0].report_period if metrics else "-"
        oldest = metrics[-1].report_period if metrics else "-"
        cov, tot = _field_coverage(line_items)
        print(f"{name:18s} {len(metrics):7d} {latest:>12s} {oldest:>12s} {cov:>4d}/{tot:<3d}")

    print("\n-- RELIABILITY (latest-period raw figures; should agree across sources) --")
    print(f"{'provider':18s} {'period':>12s} {'revenue':>16s} {'net_income':>16s} {'equity':>16s}")
    raw = {}
    for name, (metrics, line_items) in rows.items():
        if not line_items:
            continue
        li = line_items[0]
        rev, ni, eq = (getattr(li, "revenue", None), getattr(li, "net_income", None),
                       getattr(li, "shareholders_equity", None))
        raw[name] = (rev, ni, eq)
        print(f"{name:18s} {li.report_period:>12s} {_fmt(rev):>16s} {_fmt(ni):>16s} {_fmt(eq):>16s}")
    _print_dispersion(raw)

    print("\n-- USABILITY (Buffett agent analysis per source) --")
    print(f"{'provider':18s} {'ROE':>7s} {'D/E':>7s} {'CurR':>6s} {'moat':>6s} {'consist':>8s} {'bookval':>8s} {'intrinsic_value':>18s}")
    for name, (metrics, line_items) in rows.items():
        moat = analyze_moat(metrics)
        cons = analyze_consistency(line_items)
        bv = analyze_book_value_growth(line_items)
        iv = calculate_intrinsic_value(line_items)
        latest = metrics[0] if metrics else None
        roe = _fmt(latest.return_on_equity, pct=True) if latest else "n/a"
        de = f"{latest.debt_to_equity:.2f}" if latest and latest.debt_to_equity is not None else "n/a"
        cr = f"{latest.current_ratio:.2f}" if latest and latest.current_ratio is not None else "n/a"
        ivv = iv.get("intrinsic_value")
        ivs = f"{ivv:,.0f}" if ivv else "FAILED"
        print(f"{name:18s} {roe:>7s} {de:>7s} {cr:>6s} {str(moat['score'])+'/'+str(moat['max_score']):>6s} "
              f"{str(cons['score']):>8s} {str(bv['score']):>8s} {ivs:>18s}")


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "KO"]
    print(f"Comparing providers: {', '.join(PROVIDERS)}")
    print(f"Tickers: {', '.join(tickers)} | as of {END}")
    for t in tickers:
        analyze_ticker(t)
    print("\nNote: financialdatasets is free only for AAPL/GOOGL/MSFT/NVDA/TSLA; "
          "other tickers need FINANCIAL_DATASETS_API_KEY (0 periods if unset).")
