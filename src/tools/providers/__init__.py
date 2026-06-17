"""Alternative free data providers for the AI hedge fund.

Selectable via the DATA_PROVIDER environment variable:
  - financialdatasets (default; handled in src/tools/api.py)
  - yfinance
  - sec
  - fmp

Each provider exposes the same four functions consumed by src/tools/api.py:
  fetch_financial_metrics, fetch_line_items, fetch_market_cap, fetch_prices
"""
from __future__ import annotations

import os


def active_provider() -> str:
    """Return the normalized DATA_PROVIDER value (default 'financialdatasets')."""
    return os.environ.get("DATA_PROVIDER", "financialdatasets").strip().lower()


def get_provider(name: str | None = None):
    """Return the provider module for `name` (or the active one). None => financialdatasets."""
    name = (name or active_provider()).lower()
    if name in ("financialdatasets", "financial_datasets", "fds", ""):
        return None
    if name in ("yfinance", "yahoo", "yf"):
        from . import yfinance_provider
        return yfinance_provider
    if name in ("sec", "edgar", "sec_edgar"):
        from . import sec_provider
        return sec_provider
    if name in ("fmp", "financialmodelingprep", "financial_modeling_prep"):
        from . import fmp_provider
        return fmp_provider
    if name in ("hybrid", "sec+yfinance", "recommended"):
        from . import hybrid_provider
        return hybrid_provider
    raise ValueError(f"Unknown DATA_PROVIDER: {name!r}")
