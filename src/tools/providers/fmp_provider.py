"""Financial Modeling Prep (FMP) data provider — uses the current `/stable/` API.

Free tier: 250 req/day, US-only, ~5 years annual / 5 quarters, needs FMP_API_KEY.
(The legacy /api/v3/ endpoints return 403 for new free keys; /stable/ is current.)
Statements are SEC-sourced and standardized, so field mapping is straightforward.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import requests

from src.data.models import FinancialMetrics, LineItem, Price
from . import common, cache_db

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/stable"

# canonical field -> FMP json key. Underscore-prefixed are combined below.
_INCOME = {
    "revenue": "revenue", "gross_profit": "grossProfit",
    "operating_income": "operatingIncome", "net_income": "netIncome",
    "interest_expense": "interestExpense",
    "depreciation_and_amortization": "depreciationAndAmortization",
    "outstanding_shares": "weightedAverageShsOutDil",
}
_BALANCE = {
    "total_assets": "totalAssets", "total_liabilities": "totalLiabilities",
    "shareholders_equity": "totalStockholdersEquity",
    "current_assets": "totalCurrentAssets", "current_liabilities": "totalCurrentLiabilities",
    "inventory": "inventory", "total_debt": "totalDebt",
    "cash_and_equivalents": "cashAndCashEquivalents",
}
_CASHFLOW = {
    "capital_expenditure": "capitalExpenditure", "free_cash_flow": "freeCashFlow",
    "operating_cash_flow": "netCashProvidedByOperatingActivities",
    "dividends_and_other_cash_distributions": "commonDividendsPaid",  # stable name (negative)
    "_repurchase": "commonStockRepurchased",  # negative magnitude
    "_issuance": "commonStockIssued",          # often absent in /stable/
}


def _api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.environ.get("FMP_API_KEY")


def _get(endpoint: str, key: str, symbol: str, period: str = None, limit: int = None) -> list[dict]:
    url = f"{_BASE}/{endpoint}?symbol={symbol}&apikey={key}"
    if period:
        url += f"&period={period}"
    if limit:
        url += f"&limit={limit}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            logger.warning("FMP %s -> HTTP %s", endpoint, r.status_code)
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("FMP request failed (%s): %s", endpoint, e)
        return []


@lru_cache(maxsize=256)
def _period_records(ticker: str, period: str, limit: int, end_date: str, api_key=None) -> list[dict]:
    cache_key = "quarterly" if period in ("quarterly", "quarter") else "annual"
    cached = cache_db.get_records("fmp", ticker, cache_key, limit, end_date)
    if cached is not None:
        return cached
    key = _api_key(api_key)
    if not key:
        logger.warning("FMP_API_KEY not set; FMP provider returns no data.")
        return []
    fmp_period = "quarter" if period in ("quarterly", "quarter") else "annual"
    # FMP free tier rejects limit > 5 with HTTP 402; cap it (override via FMP_MAX_LIMIT).
    limit = min(limit, int(os.environ.get("FMP_MAX_LIMIT", "5")))
    income = {d["date"]: d for d in _get("income-statement", key, ticker, fmp_period, limit)}
    balance = {d["date"]: d for d in _get("balance-sheet-statement", key, ticker, fmp_period, limit)}
    cash = {d["date"]: d for d in _get("cash-flow-statement", key, ticker, fmp_period, limit)}

    dates = sorted([d for d in income if d <= end_date], reverse=True)[:limit]
    records: list[dict] = []
    for date in dates:
        inc, bal, cf = income.get(date, {}), balance.get(date, {}), cash.get(date, {})
        rec = {
            "report_period": date,
            "period": "annual" if fmp_period == "annual" else "quarterly",
            "currency": inc.get("reportedCurrency", "USD"),
        }
        for field, src_key in _INCOME.items():
            if inc.get(src_key) is not None:
                rec[field] = inc[src_key]
        for field, src_key in _BALANCE.items():
            if bal.get(src_key) is not None:
                rec[field] = bal[src_key]
        for field, src_key in _CASHFLOW.items():
            if field.startswith("_"):
                continue
            if cf.get(src_key) is not None:
                rec[field] = cf[src_key]
        # Net buyback = issuance + repurchase (FMP repurchase already negative).
        issuance = cf.get(_CASHFLOW["_issuance"])
        repurchase = cf.get(_CASHFLOW["_repurchase"])
        if issuance is not None or repurchase is not None:
            rec["issuance_or_purchase_of_equity_shares"] = (issuance or 0.0) + (repurchase or 0.0)
        records.append(rec)
    cache_db.put_records("fmp", ticker, cache_key, limit, end_date, records)
    return records


def fetch_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=None) -> list[FinancialMetrics]:
    records = _period_records(ticker, period, limit, end_date, api_key)
    metrics = common.build_financial_metrics(ticker, records)
    mc = fetch_market_cap(ticker, end_date, api_key)
    if metrics and mc is not None:
        metrics[0].market_cap = mc
    return metrics


def fetch_line_items(ticker, line_items, end_date, period="ttm", limit=10, api_key=None) -> list[LineItem]:
    records = _period_records(ticker, period, limit, end_date, api_key)
    return common.build_line_items(ticker, records)


@lru_cache(maxsize=256)
def fetch_market_cap(ticker, end_date, api_key=None) -> Optional[float]:
    key = _api_key(api_key)
    if not key:
        return None
    data = _get("market-capitalization", key, ticker)
    if data and data[0].get("marketCap"):
        return float(data[0]["marketCap"])
    return None


def fetch_prices(ticker, start_date, end_date, api_key=None) -> list[Price]:
    key = _api_key(api_key)
    if not key:
        return []
    url = f"{_BASE}/historical-price-eod/full?symbol={ticker}&from={start_date}&to={end_date}&apikey={key}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        rows = r.json()
        if not isinstance(rows, list):
            rows = rows.get("historical", [])
    except Exception as e:
        logger.warning("FMP prices failed for %s: %s", ticker, e)
        return []
    prices: list[Price] = []
    for row in rows:
        prices.append(Price(
            open=float(row["open"]), close=float(row["close"]),
            high=float(row["high"]), low=float(row["low"]),
            volume=int(row.get("volume") or 0), time=row["date"],
        ))
    return prices
