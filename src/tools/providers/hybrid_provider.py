"""Hybrid data provider — the recommended free setup.

Combines the strengths found in the provider audit:
  - FUNDAMENTALS (statements, line items, ratios)  -> SEC EDGAR
      authoritative, deepest history (10+ yrs), free, any US ticker.
  - MARKET CAP / PRICES                            -> yfinance, then FMP fallback
      SEC has no price data; yfinance has no daily quota (FMP free = 250/day),
      and yfinance vs FMP market cap agreed to <1% in the audit.

Select with DATA_PROVIDER=hybrid.
"""
from __future__ import annotations

from typing import Optional

from src.data.models import FinancialMetrics, LineItem, Price
from . import sec_provider, yfinance_provider, fmp_provider


def fetch_line_items(ticker, line_items, end_date, period="ttm", limit=10, api_key=None) -> list[LineItem]:
    items = sec_provider.fetch_line_items(ticker, line_items, end_date, period, limit, api_key)
    # Backfill share count for filers whose SEC share tag is dimensional/messy
    # (e.g. Berkshire's A/B classes), so intrinsic value and book-value/share work.
    if items and getattr(items[0], "outstanding_shares", None) in (None, 0):
        shares = yfinance_provider.fetch_shares(ticker)
        if shares:
            for it in items:
                if getattr(it, "outstanding_shares", None) in (None, 0):
                    setattr(it, "outstanding_shares", shares)
    return items


def fetch_market_cap(ticker, end_date, api_key=None) -> Optional[float]:
    # yfinance first (no daily quota), FMP as fallback.
    mc = yfinance_provider.fetch_market_cap(ticker, end_date, api_key)
    if mc is not None:
        return mc
    return fmp_provider.fetch_market_cap(ticker, end_date, api_key)


def fetch_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=None) -> list[FinancialMetrics]:
    metrics = sec_provider.fetch_financial_metrics(ticker, end_date, period, limit, api_key)
    mc = fetch_market_cap(ticker, end_date, api_key)
    if metrics and mc is not None:
        metrics[0].market_cap = mc
    return metrics


def fetch_prices(ticker, start_date, end_date, api_key=None) -> list[Price]:
    prices = yfinance_provider.fetch_prices(ticker, start_date, end_date, api_key)
    if prices:
        return prices
    return fmp_provider.fetch_prices(ticker, start_date, end_date, api_key)
