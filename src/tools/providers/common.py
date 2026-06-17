"""Shared helpers for free data providers.

Providers normalize their raw source data into a list of *period records*
(plain dicts, newest first) using the canonical keys below, then call
build_line_items() / build_financial_metrics() here so all ratio math and
sign conventions live in ONE place.

Canonical period-record keys (all optional, float unless noted):
  report_period (str 'YYYY-MM-DD'), period (str), currency (str)
  revenue, gross_profit, operating_income, net_income
  depreciation_and_amortization, interest_expense
  total_assets, total_liabilities, shareholders_equity
  current_assets, current_liabilities, inventory, total_debt, cash_and_equivalents
  capital_expenditure, operating_cash_flow, free_cash_flow
  dividends_and_other_cash_distributions   (negative = cash paid out)
  issuance_or_purchase_of_equity_shares    (negative = net buyback)
  outstanding_shares

Sign conventions (match financialdatasets so agent logic is unchanged):
  - dividends_and_other_cash_distributions < 0  => dividends were paid
  - issuance_or_purchase_of_equity_shares  < 0  => net share buyback
  - capital_expenditure < 0                     => cash outflow
"""
from __future__ import annotations

from typing import Optional

from src.data.models import FinancialMetrics, LineItem

# Line-item fields the agents may request; attached to every LineItem when present.
LINE_ITEM_FIELDS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "depreciation_and_amortization", "interest_expense",
    "total_assets", "total_liabilities", "shareholders_equity",
    "current_assets", "current_liabilities", "inventory",
    "total_debt", "cash_and_equivalents",
    "capital_expenditure", "operating_cash_flow", "free_cash_flow",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares",
    "outstanding_shares",
]


def _num(record: dict, key: str) -> Optional[float]:
    v = record.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _enriched(r: dict) -> dict:
    """Fill derived balance-sheet fields via the accounting identity A = L + E.

    Some filers (e.g. Coca-Cola) omit a single us-gaap:Liabilities total, so
    total_liabilities is reconstructed from total_assets - equity when absent.
    """
    r = dict(r)
    ta, eq, tl = _num(r, "total_assets"), _num(r, "shareholders_equity"), _num(r, "total_liabilities")
    if tl is None and ta is not None and eq is not None:
        r["total_liabilities"] = ta - eq
    if eq is None and ta is not None and tl is not None:
        r["shareholders_equity"] = ta - tl
    return r


def build_line_items(ticker: str, records: list[dict]) -> list[LineItem]:
    """Convert canonical period records into LineItem objects (extra fields allowed)."""
    records = [_enriched(r) for r in records]
    items: list[LineItem] = []
    for r in records:
        payload = {
            "ticker": ticker,
            "report_period": r.get("report_period", ""),
            "period": r.get("period", "annual"),
            "currency": r.get("currency", "USD"),
        }
        # Always set every field (None when absent) so attribute access matches
        # financialdatasets, which returns explicit nulls for missing line items.
        for field in LINE_ITEM_FIELDS:
            payload[field] = _num(r, field)
        items.append(LineItem(**payload))
    return items


def _metrics_for_period(ticker: str, r: dict, prev: Optional[dict]) -> FinancialMetrics:
    revenue = _num(r, "revenue")
    net_income = _num(r, "net_income")
    equity = _num(r, "shareholders_equity")
    assets = _num(r, "total_assets")
    liabilities = _num(r, "total_liabilities")
    cur_assets = _num(r, "current_assets")
    cur_liab = _num(r, "current_liabilities")
    inventory = _num(r, "inventory")
    operating_income = _num(r, "operating_income")
    gross_profit = _num(r, "gross_profit")
    interest_expense = _num(r, "interest_expense")
    fcf = _num(r, "free_cash_flow")
    shares = _num(r, "outstanding_shares")
    dividends = _num(r, "dividends_and_other_cash_distributions")
    market_cap = _num(r, "market_cap")

    quick_assets = None
    if cur_assets is not None and inventory is not None:
        quick_assets = cur_assets - inventory

    return FinancialMetrics(
        ticker=ticker,
        report_period=r.get("report_period", ""),
        period=r.get("period", "annual"),
        currency=r.get("currency", "USD"),
        market_cap=market_cap,
        enterprise_value=None,
        price_to_earnings_ratio=None,
        price_to_book_ratio=None,
        price_to_sales_ratio=None,
        enterprise_value_to_ebitda_ratio=None,
        enterprise_value_to_revenue_ratio=None,
        free_cash_flow_yield=None,
        peg_ratio=None,
        gross_margin=_safe_div(gross_profit, revenue),
        operating_margin=_safe_div(operating_income, revenue),
        net_margin=_safe_div(net_income, revenue),
        return_on_equity=_safe_div(net_income, equity),
        return_on_assets=_safe_div(net_income, assets),
        return_on_invested_capital=None,
        asset_turnover=_safe_div(revenue, assets),
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=_safe_div(cur_assets, cur_liab),
        quick_ratio=_safe_div(quick_assets, cur_liab),
        cash_ratio=_safe_div(_num(r, "cash_and_equivalents"), cur_liab),
        operating_cash_flow_ratio=_safe_div(_num(r, "operating_cash_flow"), cur_liab),
        debt_to_equity=_safe_div(liabilities, equity),
        debt_to_assets=_safe_div(liabilities, assets),
        interest_coverage=_safe_div(operating_income, abs(interest_expense)) if interest_expense else None,
        revenue_growth=_growth(revenue, _num(prev, "revenue") if prev else None),
        earnings_growth=_growth(net_income, _num(prev, "net_income") if prev else None),
        book_value_growth=_growth(equity, _num(prev, "shareholders_equity") if prev else None),
        earnings_per_share_growth=None,
        free_cash_flow_growth=_growth(fcf, _num(prev, "free_cash_flow") if prev else None),
        operating_income_growth=_growth(operating_income, _num(prev, "operating_income") if prev else None),
        ebitda_growth=None,
        payout_ratio=_safe_div(-dividends, net_income) if (dividends is not None and net_income) else None,
        earnings_per_share=_safe_div(net_income, shares),
        book_value_per_share=_safe_div(equity, shares),
        free_cash_flow_per_share=_safe_div(fcf, shares),
    )


def build_financial_metrics(ticker: str, records: list[dict]) -> list[FinancialMetrics]:
    """Convert canonical period records (newest first) into FinancialMetrics with computed ratios."""
    records = [_enriched(r) for r in records]
    out: list[FinancialMetrics] = []
    for i, r in enumerate(records):
        prev = records[i + 1] if i + 1 < len(records) else None
        out.append(_metrics_for_period(ticker, r, prev))
    return out
