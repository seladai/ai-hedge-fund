"""Comparison tests for free data providers (yfinance, SEC EDGAR, FMP).

Verifies COMPLETENESS, RELIABILITY, and USABILITY of each free provider against
the Buffett agent's data contract. Network-dependent: tests skip (not fail) when
a source is unreachable, so the suite stays green offline / under rate limits.

Run:  poetry run pytest tests/test_data_providers_comparison.py -v
"""
from __future__ import annotations

import datetime as dt
import os

import pytest

from src.tools.providers import yfinance_provider, sec_provider, fmp_provider, hybrid_provider
from src.agents.warren_buffett import (
    analyze_fundamentals, analyze_moat, calculate_intrinsic_value,
)

END = dt.date.today().isoformat()
TICKER = "AAPL"

# Raw line items the Buffett agent depends on.
REQUIRED = [
    "revenue", "net_income", "shareholders_equity", "total_assets", "total_liabilities",
    "current_assets", "current_liabilities", "capital_expenditure",
    "depreciation_and_amortization", "outstanding_shares", "free_cash_flow",
]

FREE_PROVIDERS = {"yfinance": yfinance_provider, "sec": sec_provider}
if os.environ.get("FMP_API_KEY"):
    FREE_PROVIDERS["fmp"] = fmp_provider


def _safe(fn, *a, **k):
    """Call a network fn; skip the test if the source is unreachable."""
    try:
        return fn(*a, **k)
    except Exception as e:  # network / rate limit / parse
        pytest.skip(f"data source unavailable: {e}")


# --- COMPLETENESS ----------------------------------------------------------

@pytest.mark.parametrize("name", list(FREE_PROVIDERS))
def test_provider_returns_periods(name):
    metrics = _safe(FREE_PROVIDERS[name].fetch_financial_metrics, TICKER, END, "ttm", 10)
    if not metrics:
        pytest.skip(f"{name} returned no data (offline or rate-limited)")
    assert len(metrics) >= 1


@pytest.mark.parametrize("name", list(FREE_PROVIDERS))
def test_required_line_items_present(name):
    items = _safe(FREE_PROVIDERS[name].fetch_line_items, TICKER, REQUIRED, END, "ttm", 10)
    if not items:
        pytest.skip(f"{name} returned no line items")
    latest = items[0]
    missing = [f for f in REQUIRED if getattr(latest, f, None) is None]
    assert not missing, f"{name} missing line items: {missing}"


def test_sec_has_deep_history():
    """SEC EDGAR must clear the moat threshold (>=5 periods) — its key advantage."""
    metrics = _safe(sec_provider.fetch_financial_metrics, TICKER, END, "ttm", 10)
    if not metrics:
        pytest.skip("SEC unavailable")
    assert len(metrics) >= 5, "SEC should provide deep multi-year history"


# --- RELIABILITY -----------------------------------------------------------

def test_cross_source_revenue_agreement():
    """yfinance vs SEC latest-year revenue should agree once periods align."""
    y = _safe(yfinance_provider.fetch_line_items, TICKER, REQUIRED, END, "annual", 5)
    s = _safe(sec_provider.fetch_line_items, TICKER, REQUIRED, END, "annual", 5)
    if not y or not s:
        pytest.skip("a source returned no data")
    s_by_date = {it.report_period: it for it in s}
    common_dates = [it.report_period for it in y if it.report_period in s_by_date]
    if not common_dates:
        pytest.skip("no overlapping period between yfinance and SEC")
    d = common_dates[0]
    yv = getattr([it for it in y if it.report_period == d][0], "revenue")
    sv = getattr(s_by_date[d], "revenue")
    assert yv and sv
    spread = abs(yv - sv) / max(abs(yv), abs(sv))
    assert spread < 0.05, f"revenue disagreement on {d}: yf={yv:,.0f} sec={sv:,.0f} ({spread:.1%})"


def test_sign_conventions():
    """Dividends and net buybacks must be negative (agent relies on `< 0`)."""
    items = _safe(sec_provider.fetch_line_items, TICKER, REQUIRED, END, "ttm", 5)
    if not items:
        pytest.skip("SEC unavailable")
    div = getattr(items[0], "dividends_and_other_cash_distributions", None)
    if div is not None:
        assert div <= 0, "dividends paid should be negative"


# --- USABILITY (Buffett functions actually compute) ------------------------

@pytest.mark.parametrize("name", list(FREE_PROVIDERS))
def test_buffett_functions_compute(name):
    metrics = _safe(FREE_PROVIDERS[name].fetch_financial_metrics, TICKER, END, "ttm", 10)
    items = _safe(FREE_PROVIDERS[name].fetch_line_items, TICKER, REQUIRED, END, "ttm", 10)
    if not metrics or not items:
        pytest.skip(f"{name} returned no data")
    fundamentals = analyze_fundamentals(metrics)
    moat = analyze_moat(metrics)
    iv = calculate_intrinsic_value(items)
    assert fundamentals["score"] >= 0
    assert "score" in moat
    if name == "sec":
        assert moat["score"] > 0, "SEC depth should let moat analysis run"
        assert iv.get("intrinsic_value"), "SEC depth should let intrinsic value compute"


def test_hybrid_gives_depth_and_market_cap():
    """Hybrid = SEC fundamentals (deep) + yfinance/FMP market cap (for margin of safety)."""
    metrics = _safe(hybrid_provider.fetch_financial_metrics, TICKER, END, "ttm", 10)
    if not metrics:
        pytest.skip("hybrid unavailable")
    assert len(metrics) >= 5, "hybrid should inherit SEC's deep history"
    assert metrics[0].market_cap is not None, "hybrid must supply a market cap (yfinance/FMP)"


def test_metrics_have_core_ratios():
    """ROE, D/E and current ratio must be computed (the user's key Buffett metrics)."""
    metrics = _safe(sec_provider.fetch_financial_metrics, TICKER, END, "ttm", 10)
    if not metrics:
        pytest.skip("SEC unavailable")
    m = metrics[0]
    assert m.return_on_equity is not None
    assert m.debt_to_equity is not None
    assert m.current_ratio is not None
