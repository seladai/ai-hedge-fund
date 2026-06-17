"""SEC EDGAR data provider — free, any US ticker, deep multi-year (10-30yr) history.

Uses the authoritative public endpoints (no API key):
  - https://www.sec.gov/files/company_tickers.json   (ticker -> CIK)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  (all XBRL facts)

SEC requires a descriptive User-Agent. Set SEC_EDGAR_USER_AGENT to your own
"Name email" string; a sensible default is used otherwise.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import requests

from src.data.models import FinancialMetrics, LineItem, Price
from . import common, _net, cache_db

logger = logging.getLogger(__name__)

_UA = os.environ.get("SEC_EDGAR_USER_AGENT", "ai-hedge-fund research contact@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}

# canonical field -> candidate us-gaap concept tags (first with data wins), and sign.
# sign = +1 keep as reported, -1 negate (SEC reports outflows as positive).
_CONCEPTS = {
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], 1),
    "gross_profit": (["GrossProfit"], 1),
    "operating_income": (["OperatingIncomeLoss"], 1),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], 1),
    "interest_expense": (["InterestExpense", "InterestExpenseNonoperating"], 1),
    "depreciation_and_amortization": (["DepreciationDepletionAndAmortization",
                                       "DepreciationAmortizationAndAccretionNet",
                                       "DepreciationAndAmortization"], 1),
    "total_assets": (["Assets"], 1),
    "total_liabilities": (["Liabilities"], 1),
    "shareholders_equity": (["StockholdersEquity",
                             "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], 1),
    "current_assets": (["AssetsCurrent"], 1),
    "current_liabilities": (["LiabilitiesCurrent"], 1),
    "inventory": (["InventoryNet"], 1),
    "cash_and_equivalents": (["CashAndCashEquivalentsAtCarryingValue"], 1),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], 1),
    "capital_expenditure": (["PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets"], -1),  # outflow -> negative
    "_dividends": (["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"], -1),  # paid -> negative
    "_repurchase": (["PaymentsForRepurchaseOfCommonStock"], 1),  # positive magnitude
    "_issuance": (["ProceedsFromIssuanceOfCommonStock"], 1),
}
_SHARE_CONCEPTS = ["WeightedAverageNumberOfDilutedSharesOutstanding",
                   "WeightedAverageNumberOfSharesOutstandingBasic",
                   "CommonStockSharesOutstanding"]


@lru_cache(maxsize=1)
def _ticker_to_cik() -> dict:
    r = _net.sec_get("https://www.sec.gov/files/company_tickers.json", _HEADERS)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


@lru_cache(maxsize=64)
def _company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = _net.sec_get(url, _HEADERS)
    r.raise_for_status()
    return r.json()


def _annual_by_end(facts: dict, concepts: list[str], unit: str = "USD") -> dict[str, float]:
    """Return {end_date: value} for annual 10-K entries, first concept with data.

    Keyed by period-end date (not fiscal year): a single 10-K tags its comparative
    prior-year figures with the SAME fy, so keying by fy would collapse/drop years.
    Keying by `end` preserves every distinct fiscal-year-end across all filings.
    """
    from datetime import date
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    for concept in concepts:
        node = usgaap.get(concept) or dei.get(concept)
        if not node:
            continue
        units = node.get("units", {}).get(unit)
        if not units:
            continue
        by_end: dict[str, tuple[float, str]] = {}  # end -> (val, filed)
        for e in units:
            if e.get("form") not in ("10-K", "10-K/A") or e.get("fp") != "FY":
                continue
            end = e.get("end")
            if not end:
                continue
            # Flow items carry a start date; keep only ~annual spans (drop quarters).
            start = e.get("start")
            if start:
                try:
                    if (date.fromisoformat(end) - date.fromisoformat(start)).days < 300:
                        continue
                except ValueError:
                    pass
            prev = by_end.get(end)
            if prev is None or e.get("filed", "") > prev[1]:
                by_end[end] = (e["val"], e.get("filed", ""))
        if by_end:
            return {end: val for end, (val, _f) in by_end.items()}
    return {}


@lru_cache(maxsize=256)
def _period_records(ticker: str, limit: int, end_date: str) -> list[dict]:
    cached = cache_db.get_records("sec", ticker, "annual", limit, end_date)
    if cached is not None:
        return cached
    cik = _ticker_to_cik().get(ticker.upper())
    if not cik:
        logger.warning("SEC: no CIK for %s", ticker)
        return []
    try:
        facts = _company_facts(cik)
    except Exception as e:
        logger.warning("SEC companyfacts failed for %s: %s", ticker, e)
        return []

    # Map each canonical field to {end_date: signed value}; assemble per period-end.
    field_maps: dict[str, dict[str, float]] = {}
    for field, (concepts, sign) in _CONCEPTS.items():
        field_maps[field] = {end: val * sign for end, val in _annual_by_end(facts, concepts).items()}
    shares_map = _annual_by_end(facts, _SHARE_CONCEPTS, unit="shares")

    all_ends = set()
    for m in field_maps.values():
        all_ends.update(m.keys())
    all_ends.update(shares_map.keys())

    records: list[dict] = []
    for end in sorted(all_ends, reverse=True):
        if end > end_date:
            continue
        rec = {"report_period": end, "period": "annual", "currency": "USD"}
        for field in _CONCEPTS:
            if field.startswith("_"):
                continue
            if end in field_maps[field]:
                rec[field] = field_maps[field][end]
        if end in shares_map:
            rec["outstanding_shares"] = shares_map[end]

        # Net buyback = issuance - repurchase (both positive magnitudes); negative => buyback.
        issuance = field_maps["_issuance"].get(end)
        repurchase = field_maps["_repurchase"].get(end)
        if issuance is not None or repurchase is not None:
            rec["issuance_or_purchase_of_equity_shares"] = (issuance or 0.0) - (repurchase or 0.0)
        if end in field_maps["_dividends"]:
            rec["dividends_and_other_cash_distributions"] = field_maps["_dividends"][end]

        # FCF = operating cash flow - |capex| (capex already negative here).
        if rec.get("operating_cash_flow") is not None:
            rec["free_cash_flow"] = rec["operating_cash_flow"] + (rec.get("capital_expenditure") or 0.0)

        records.append(rec)
        if len(records) >= limit:
            break
    cache_db.put_records("sec", ticker, "annual", limit, end_date, records)
    return records


def fetch_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=None) -> list[FinancialMetrics]:
    records = _period_records(ticker, limit, end_date)
    return common.build_financial_metrics(ticker, records)


def fetch_line_items(ticker, line_items, end_date, period="ttm", limit=10, api_key=None) -> list[LineItem]:
    records = _period_records(ticker, limit, end_date)
    return common.build_line_items(ticker, records)


def fetch_market_cap(ticker, end_date, api_key=None) -> Optional[float]:
    # SEC has no market cap (price data). Caller may fall back to yfinance.
    return None


def fetch_prices(ticker, start_date, end_date, api_key=None) -> list[Price]:
    # SEC EDGAR does not provide market prices.
    return []
