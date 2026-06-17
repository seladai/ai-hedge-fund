"""Shared rate limiting + retry helpers for free data providers.

  - SEC EDGAR: official limit is 10 requests/second. We pace to ~8/s and retry
    on 429/503 honoring Retry-After.
  - yfinance: Yahoo has no published quota but throttles bursts with 429/errors.
    We pace calls and retry with backoff.

Throttles are process-global and lock-guarded so concurrent agents on the same
ticker still respect the limits.
"""
from __future__ import annotations

import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)


class RateLimiter:
    """Minimum-interval throttle (process-global, thread-safe)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


# ~8 req/s leaves headroom under SEC's 10/s ceiling; ~4 req/s pacing for Yahoo.
SEC_LIMITER = RateLimiter(0.125)
YF_LIMITER = RateLimiter(0.25)


def sec_get(url: str, headers: dict, timeout: int = 30, max_retries: int = 4) -> requests.Response:
    """GET against data.sec.gov with pacing + retry on 429/503."""
    resp = None
    for attempt in range(max_retries + 1):
        SEC_LIMITER.wait()
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code in (429, 503) and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 8)
            logger.warning("SEC %s on attempt %d; waiting %.1fs", resp.status_code, attempt + 1, delay)
            time.sleep(delay)
            continue
        return resp
    return resp


def yf_call(fn, *args, _retries: int = 3, _retry_empty: bool = False, **kwargs):
    """Call a yfinance function with pacing + backoff retry.

    Retries on exceptions; if _retry_empty, also retries when the result is
    empty/falsey (helps when Yahoo soft-throttles with blank payloads).
    """
    last_exc = None
    for attempt in range(_retries):
        YF_LIMITER.wait()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - yfinance raises many types
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
            continue
        empty = result is None or (hasattr(result, "empty") and getattr(result, "empty"))
        if _retry_empty and empty and attempt < _retries - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        return result
    if last_exc:
        raise last_exc
    return None
