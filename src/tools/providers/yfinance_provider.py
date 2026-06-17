"""Yahoo Finance (yfinance) data provider — free, any US ticker, ~4 years of statements."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

from src.data.models import FinancialMetrics, LineItem, Price
from . import common, _net, cache_db

logger = logging.getLogger(__name__)

# Candidate yfinance statement row labels for each canonical field (first match wins).
_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "EBIT"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "depreciation_and_amortization": [
        "Reconciled Depreciation", "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    ],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "shareholders_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "total_debt": ["Total Debt"],
    "cash_and_equivalents": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "outstanding_shares": ["Ordinary Shares Number", "Share Issued"],
    "capital_expenditure": ["Capital Expenditure", "Purchase Of PPE"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "free_cash_flow": ["Free Cash Flow"],
    "_dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "_repurchase": ["Repurchase Of Capital Stock", "Repurchase Of Common Stock"],
    "_issuance": ["Common Stock Issuance", "Issuance Of Capital Stock"],
}


@lru_cache(maxsize=64)
def _ticker(ticker: str):
    import yfinance as yf
    return yf.Ticker(ticker)


def _row_value(frame: pd.DataFrame, col, labels: list[str]) -> Optional[float]:
    for label in labels:
        if label in frame.index:
            try:
                val = frame.loc[label, col]
            except Exception:
                continue
            if val is not None and pd.notna(val):
                return float(val)
    return None


def _statement(t, name: str, annual: bool) -> pd.DataFrame:
    attr = {
        ("income", True): "income_stmt", ("income", False): "quarterly_income_stmt",
        ("balance", True): "balance_sheet", ("balance", False): "quarterly_balance_sheet",
        ("cash", True): "cashflow", ("cash", False): "quarterly_cashflow",
    }[(name, annual)]
    try:
        df = _net.yf_call(lambda: getattr(t, attr), _retry_empty=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        logger.warning("yfinance %s failed: %s", attr, e)
    return pd.DataFrame()


@lru_cache(maxsize=256)
def _period_records(ticker: str, period: str, limit: int, end_date: str) -> list[dict]:
    annual = period in ("annual", "ttm")
    cache_key = "annual" if annual else "quarterly"
    cached = cache_db.get_records("yfinance", ticker, cache_key, limit, end_date)
    if cached is not None:
        return cached
    t = _ticker(ticker)
    inc = _statement(t, "income", annual)
    bal = _statement(t, "balance", annual)
    cf = _statement(t, "cash", annual)
    if inc.empty and bal.empty and cf.empty:
        return []

    # Union of period-end dates (columns) across statements, newest first, <= end_date.
    cols = set()
    for df in (inc, bal, cf):
        cols.update(list(df.columns))
    dated = sorted(
        [c for c in cols if str(pd.Timestamp(c).date()) <= end_date],
        reverse=True,
    )[:limit]

    income_fields = ("revenue", "gross_profit", "operating_income", "net_income", "interest_expense")
    records: list[dict] = []
    for col in dated:
        rec = {
            "report_period": str(pd.Timestamp(col).date()),
            "period": "annual" if annual else "quarterly",
            "currency": "USD",
        }
        for field, labels in _ROWS.items():
            if field.startswith("_"):
                continue
            frames = (inc,) if field in income_fields else (inc, bal, cf)
            for df in frames:
                if df is None or df.empty:
                    continue
                val = _row_value(df, col, labels)
                if val is not None:
                    rec[field] = val
                    break

        # Net share issuance/buyback = issuance + repurchase (repurchase already negative).
        issuance = _row_value(cf, col, _ROWS["_issuance"]) if not cf.empty else None
        repurchase = _row_value(cf, col, _ROWS["_repurchase"]) if not cf.empty else None
        if issuance is not None or repurchase is not None:
            rec["issuance_or_purchase_of_equity_shares"] = (issuance or 0.0) + (repurchase or 0.0)

        dividends = _row_value(cf, col, _ROWS["_dividends_paid"]) if not cf.empty else None
        if dividends is not None:
            rec["dividends_and_other_cash_distributions"] = dividends

        # Derive FCF if Yahoo didn't provide it: OCF + capex (capex negative).
        if "free_cash_flow" not in rec and rec.get("operating_cash_flow") is not None:
            capex = rec.get("capital_expenditure") or 0.0
            rec["free_cash_flow"] = rec["operating_cash_flow"] + capex

        records.append(rec)
    cache_db.put_records("yfinance", ticker, cache_key, limit, end_date, records)
    return records


def fetch_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=None) -> list[FinancialMetrics]:
    records = _period_records(ticker, period, limit, end_date)
    metrics = common.build_financial_metrics(ticker, records)
    mc = fetch_market_cap(ticker, end_date, api_key)
    if metrics and mc is not None:
        metrics[0].market_cap = mc
    return metrics


def fetch_line_items(ticker, line_items, end_date, period="ttm", limit=10, api_key=None) -> list[LineItem]:
    records = _period_records(ticker, period, limit, end_date)
    return common.build_line_items(ticker, records)


@lru_cache(maxsize=256)
def fetch_market_cap(ticker, end_date, api_key=None) -> Optional[float]:
    t = _ticker(ticker)
    try:
        mc = _net.yf_call(lambda: t.fast_info.get("market_cap"))
        if mc:
            return float(mc)
    except Exception:
        pass
    try:
        mc = _net.yf_call(lambda: t.info.get("marketCap"))
        return float(mc) if mc else None
    except Exception:
        return None


@lru_cache(maxsize=256)
def fetch_shares(ticker) -> Optional[float]:
    """Current shares outstanding (for backfilling filers whose SEC share tag is messy)."""
    t = _ticker(ticker)
    try:
        sh = _net.yf_call(lambda: t.fast_info.get("shares"))
        if sh:
            return float(sh)
    except Exception:
        pass
    try:
        sh = _net.yf_call(lambda: t.info.get("sharesOutstanding"))
        return float(sh) if sh else None
    except Exception:
        return None


def fetch_prices(ticker, start_date, end_date, api_key=None) -> list[Price]:
    t = _ticker(ticker)
    try:
        df = _net.yf_call(lambda: t.history(start=start_date, end=end_date, auto_adjust=False), _retry_empty=True)
    except Exception as e:
        logger.warning("yfinance history failed for %s: %s", ticker, e)
        return []
    if df is None:
        return []
    prices: list[Price] = []
    for idx, row in df.iterrows():
        prices.append(Price(
            open=float(row["Open"]), close=float(row["Close"]),
            high=float(row["High"]), low=float(row["Low"]),
            volume=int(row["Volume"]), time=str(pd.Timestamp(idx).date()),
        ))
    return prices
