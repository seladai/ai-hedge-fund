"""Comprehensive provider data audit across many tickers.

Compares every raw line item and every calculated metric across the available
providers (SEC EDGAR = authoritative reference; yfinance; FMP if FMP_API_KEY set;
financialdatasets on its 5 free tickers / or with a key) and CLASSIFIES each
difference so you can see whether a gap is:

  match      |delta| < 1%
  minor      1-5%
  value_diff > 5% at the SAME period       (real data disagreement)
  units      ratio ~ x1000 / /1000         (scale/units mismatch)
  sign       opposite signs
  missing    present in one provider only
  timeline   providers' latest periods differ (freshness), shown separately

Outputs:
  provider_audit_detail.csv     long format, every (ticker, period, attribute, provider, value)
  provider_audit_report.md      coverage, freshness, per-attribute reliability, taxonomy, scoreboard

Usage:
  poetry run python provider_audit.py                  # default 20 tickers
  poetry run python provider_audit.py AAPL MSFT ...    # custom
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import statistics as stats
import sys
import time

# Allow running from scripts/: put the repo root (scripts/..) on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()  # pick up FMP_API_KEY / SEC_EDGAR_USER_AGENT / FINANCIAL_DATASETS_API_KEY from .env

from src.tools.providers import yfinance_provider, sec_provider, fmp_provider

END = dt.date.today().isoformat()

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "JPM", "JNJ", "KO",
    "PG", "WMT", "XOM", "V", "MA", "HD", "CVX", "PFE", "CSCO", "PEP",
]

RAW_ATTRS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "depreciation_and_amortization", "total_assets", "total_liabilities",
    "shareholders_equity", "current_assets", "current_liabilities",
    "capital_expenditure", "operating_cash_flow", "free_cash_flow",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares", "outstanding_shares",
]
CALC_ATTRS = [
    "return_on_equity", "return_on_assets", "gross_margin", "operating_margin",
    "net_margin", "current_ratio", "debt_to_equity", "debt_to_assets",
    "asset_turnover", "book_value_per_share", "earnings_per_share",
    "free_cash_flow_per_share",
]

REF = "sec"  # authoritative reference


def _providers():
    provs = {"sec": sec_provider, "yfinance": yfinance_provider}
    if os.environ.get("FMP_API_KEY"):
        provs["fmp"] = fmp_provider
    return provs


def _fds_available(ticker):
    """financialdatasets via native path (free 5 tickers or with key)."""
    os.environ.pop("DATA_PROVIDER", None)
    from importlib import reload
    import src.tools.api as api
    reload(api)
    try:
        li = api.search_line_items(ticker, RAW_ATTRS, END, "annual", 10)
        m = api.get_financial_metrics(ticker, END, "annual", 10)
        return m, li
    except Exception:
        return [], []


def _pull(ticker):
    """Return {provider: {'raw': {period:{attr:val}}, 'metrics_by_period', 'periods'}}"""
    out = {}
    for name, mod in _providers().items():
        try:
            items = mod.fetch_line_items(ticker, RAW_ATTRS, END, "annual", 10)
            metrics = mod.fetch_financial_metrics(ticker, END, "annual", 10)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            items, metrics = [], []
        out[name] = {
            "raw": {it.report_period: {a: getattr(it, a, None) for a in RAW_ATTRS} for it in items},
            "periods": [it.report_period for it in items],
            "metrics_by_period": {m.report_period: m for m in metrics},
        }
    m, li = _fds_available(ticker)
    if li:
        out["financialdatasets"] = {
            "raw": {it.report_period: {a: getattr(it, a, None) for a in RAW_ATTRS} for it in li},
            "periods": [it.report_period for it in li],
            "metrics_by_period": {x.report_period: x for x in m},
        }
    return out


def _classify(val, ref):
    if ref is None and val is None:
        return "match"
    if ref is None or val is None:
        return "missing"
    if ref == 0:
        return "match" if val == 0 else "value_diff"
    if (val < 0) != (ref < 0) and abs(val) > 1e-9 and abs(ref) > 1e-9:
        return "sign"
    ratio = val / ref
    a = abs(ratio)
    if 0.9 < a / 1000 < 1.1 or 0.9 < a * 1000 < 1.1:
        return "units"
    d = abs(ratio - 1)
    if d < 0.01:
        return "match"
    if d < 0.05:
        return "minor"
    return "value_diff"


def _aligned_period(pulled, providers):
    """Most recent period present in ALL listed providers; None if any has no data."""
    sets = []
    for p in providers:
        if p not in pulled or not pulled[p]["raw"]:
            return None
        sets.append(set(pulled[p]["raw"].keys()))
    common_periods = set.intersection(*sets)
    return max(common_periods) if common_periods else None


def main(tickers):
    provs = list(_providers().keys())
    detail_rows = []
    raw_tally = {p: {a: {} for a in RAW_ATTRS} for p in provs if p != REF}
    calc_tally = {p: {a: {} for a in CALC_ATTRS} for p in provs if p != REF}
    raw_absdiff = {p: {a: [] for a in RAW_ATTRS} for p in provs if p != REF}
    coverage, freshness, periods_count = {}, {}, {}
    fds_seen = 0

    for t in tickers:
        print(f"[{t}] pulling...")
        pulled = _pull(t)
        time.sleep(0.3)
        if "financialdatasets" in pulled:
            fds_seen += 1

        for p, d in pulled.items():
            coverage.setdefault(p, []); freshness.setdefault(p, []); periods_count.setdefault(p, [])
            periods_count[p].append(len(d["periods"]))
            if d["periods"]:
                latest = max(d["periods"])
                freshness[p].append(latest)
                coverage[p].append(sum(1 for a in RAW_ATTRS if d["raw"][latest].get(a) is not None) / len(RAW_ATTRS))
            for period, attrs in d["raw"].items():
                for a, v in attrs.items():
                    if v is not None:
                        detail_rows.append([t, period, "raw", a, p, v])
            for period, m in d["metrics_by_period"].items():
                for a in CALC_ATTRS:
                    v = getattr(m, a, None)
                    if v is not None:
                        detail_rows.append([t, period, "calc", a, p, v])

        if REF not in pulled or not pulled[REF]["raw"]:
            continue
        for p in [x for x in provs if x != REF]:
            if p not in pulled:
                continue
            period = _aligned_period(pulled, [REF, p])
            if period is None:
                continue
            ref_raw, cmp_raw = pulled[REF]["raw"][period], pulled[p]["raw"][period]
            for a in RAW_ATTRS:
                cls = _classify(cmp_raw.get(a), ref_raw.get(a))
                raw_tally[p][a][cls] = raw_tally[p][a].get(cls, 0) + 1
                rv, cv = ref_raw.get(a), cmp_raw.get(a)
                if rv not in (None, 0) and cv is not None:
                    raw_absdiff[p][a].append(abs(cv / rv - 1))
            ref_m, cmp_m = pulled[REF]["metrics_by_period"].get(period), pulled[p]["metrics_by_period"].get(period)
            if ref_m and cmp_m:
                for a in CALC_ATTRS:
                    cls = _classify(getattr(cmp_m, a, None), getattr(ref_m, a, None))
                    calc_tally[p][a][cls] = calc_tally[p][a].get(cls, 0) + 1

    _write_csv(detail_rows)
    _write_report(tickers, provs, raw_tally, calc_tally, raw_absdiff, coverage,
                  freshness, periods_count, fds_seen)
    print("\nWrote provider_audit_detail.csv and provider_audit_report.md")


def _med(xs):
    return f"{stats.median(xs)*100:.1f}%" if xs else "-"


def _tally_str(tally):
    order = ["match", "minor", "value_diff", "units", "sign", "missing"]
    parts = [f"{k}:{tally[k]}" for k in order if tally.get(k)]
    return " ".join(parts) if parts else "-"


def _write_csv(rows):
    with open("provider_audit_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "period", "kind", "attribute", "provider", "value"])
        w.writerows(rows)


def _write_report(tickers, provs, raw_tally, calc_tally, raw_absdiff, coverage,
                  freshness, periods_count, fds_seen):
    L = []
    L.append("# Provider Data Audit\n")
    L.append(f"As of {END}. Reference provider = **SEC EDGAR** (authoritative XBRL).\n")
    L.append(f"Tickers ({len(tickers)}): {', '.join(tickers)}\n")
    L.append("Providers compared: " + ", ".join(provs)
             + (f" + financialdatasets ({fds_seen}/{len(tickers)} tickers available)\n" if fds_seen else "\n"))

    L.append("\n## 1. Coverage & Freshness (per provider)\n")
    L.append("| provider | tickers w/ data | avg raw-field coverage | median periods | typical latest period |")
    L.append("|---|---|---|---|---|")
    for p in list(coverage.keys()):
        cov = f"{stats.mean(coverage[p])*100:.0f}%" if coverage.get(p) else "-"
        npd = f"{int(stats.median(periods_count[p]))}" if periods_count.get(p) else "-"
        try:
            latest = stats.mode(freshness[p]) if freshness.get(p) else "-"
        except stats.StatisticsError:
            latest = max(freshness[p])
        L.append(f"| {p} | {len(coverage.get(p, []))}/{len(tickers)} | {cov} | {npd} | {latest} |")

    L.append("\n## 2. Raw line-item reliability vs SEC (aligned same-period)\n")
    L.append("Median |% diff| vs SEC at the most recent COMMON fiscal period, plus difference taxonomy.\n")
    for p in raw_tally:
        L.append(f"\n### {p} vs sec — raw line items\n")
        L.append("| attribute | median \\|%diff\\| | classification counts |")
        L.append("|---|---|---|")
        for a in RAW_ATTRS:
            L.append(f"| {a} | {_med(raw_absdiff[p][a])} | {_tally_str(raw_tally[p][a])} |")

    L.append("\n## 3. Calculated-metric reliability vs SEC (aligned same-period)\n")
    L.append("yfinance/fmp metrics use the SAME formulas as SEC (shared `common.py`), so diffs here come from "
             "raw-data/units differences, NOT formula definitions. financialdatasets uses its OWN definitions.\n")
    for p in calc_tally:
        L.append(f"\n### {p} vs sec — calculated metrics\n")
        L.append("| metric | classification counts |")
        L.append("|---|---|")
        for a in CALC_ATTRS:
            L.append(f"| {a} | {_tally_str(calc_tally[p][a])} |")

    L.append("\n## 4. Difference taxonomy (raw line items, aggregated)\n")
    L.append("| provider | match | minor | value_diff | units | sign | missing |")
    L.append("|---|---|---|---|---|---|---|")
    for p in raw_tally:
        agg = {}
        for a in RAW_ATTRS:
            for k, v in raw_tally[p][a].items():
                agg[k] = agg.get(k, 0) + v
        L.append(f"| {p} | {agg.get('match',0)} | {agg.get('minor',0)} | {agg.get('value_diff',0)} | "
                 f"{agg.get('units',0)} | {agg.get('sign',0)} | {agg.get('missing',0)} |")

    L.append("\n## 5. Notes on systematic differences\n")
    L.append("- **Timeline:** financialdatasets serves TTM; SEC/yfinance/FMP serve fiscal-year-end. A *timeline* "
             "difference, not unreliability — this report compares aligned periods only.\n")
    L.append("- **Definition (financialdatasets):** its `debt_to_equity` is typically lower than total-liabilities/"
             "equity; ROE/margins may use TTM numerators. Treat FDS calc metrics as a different definition.\n")
    L.append("- **Coverage gaps:** some filers omit `Liabilities` (reconstructed as assets-equity) or `gross_profit` "
             "(banks). Shown as `missing` where not derivable.\n")
    L.append("- **Freshness:** SEC is usually most current (latest 10-K) and deepest (10+ yrs); yfinance lags ~1 FY.\n")

    with open("provider_audit_report.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    tickers = sys.argv[1:] or DEFAULT_TICKERS
    main(tickers)
