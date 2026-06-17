"""Local SQLite cache + structured fundamentals store for the data providers.

Two tables in one SQLite DB (default: <repo>/provider_cache.db, override with
PROVIDER_CACHE_DB):

  provider_cache   raw cache of each provider fetch (JSON payload), keyed by
                   (provider, ticker, period, limit_n, end_date), with a TTL so
                   repeated runs across processes reuse data without refetching.

  financial_data   long-format structured store mirroring provider_audit_detail.csv
                   columns: ticker, period, kind(raw|calc), attribute, provider,
                   value, fetched_at  -- one row per data point, upserted.

TTL via PROVIDER_CACHE_TTL_HOURS (default 24). Disable entirely with
PROVIDER_CACHE_DISABLE=1.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import threading

_LOCK = threading.Lock()
_META = {"report_period", "period", "currency", "ticker"}


def _db_path() -> str:
    env = os.environ.get("PROVIDER_CACHE_DB")
    if env:
        return env
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, "provider_cache.db")


def _disabled() -> bool:
    return os.environ.get("PROVIDER_CACHE_DISABLE", "") not in ("", "0", "false", "False")


def _ttl_hours() -> float:
    try:
        return float(os.environ.get("PROVIDER_CACHE_TTL_HOURS", "24"))
    except ValueError:
        return 24.0


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _LOCK, _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS provider_cache (
                provider TEXT, ticker TEXT, period TEXT, limit_n INTEGER, end_date TEXT,
                payload TEXT, fetched_at TEXT,
                PRIMARY KEY (provider, ticker, period, limit_n, end_date)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS financial_data (
                ticker TEXT, period TEXT, kind TEXT, attribute TEXT, provider TEXT,
                value REAL, fetched_at TEXT,
                PRIMARY KEY (ticker, period, kind, attribute, provider)
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_ticker ON financial_data(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_attr ON financial_data(attribute)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                run_time TEXT,        -- when the analysis was executed (ISO)
                tickers TEXT,         -- JSON list
                agents TEXT,          -- JSON list of analysts
                model TEXT,
                model_provider TEXT,
                data_provider TEXT,   -- hybrid / sec / yfinance / fmp / financialdatasets
                period_basis TEXT,    -- annual / quarterly / ttm (basis of the data used)
                start_date TEXT,
                end_date TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS analysis_results (
                run_id TEXT, ticker TEXT, agent TEXT,
                signal TEXT, confidence REAL, reasoning TEXT,
                intrinsic_value REAL, market_cap REAL, margin_of_safety REAL,
                data_as_of TEXT,      -- latest report_period of the data used
                data_periods INTEGER, -- how many periods were available
                period_basis TEXT,    -- annual / quarterly / ttm
                data_provider TEXT,
                run_time TEXT,
                metadata TEXT,        -- JSON (decision, etc.)
                PRIMARY KEY (run_id, ticker, agent)
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ar_ticker ON analysis_results(ticker)")


def _expired(fetched_at: str) -> bool:
    try:
        ts = dt.datetime.fromisoformat(fetched_at)
    except ValueError:
        return True
    return (dt.datetime.now() - ts) > dt.timedelta(hours=_ttl_hours())


def get_records(provider, ticker, period, limit, end_date):
    """Return cached list[dict] of period records, or None if absent/stale/disabled."""
    if _disabled():
        return None
    try:
        with _LOCK, _connect() as conn:
            row = conn.execute(
                "SELECT payload, fetched_at FROM provider_cache "
                "WHERE provider=? AND ticker=? AND period=? AND limit_n=? AND end_date=?",
                (provider, ticker, period, limit, end_date),
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row or _expired(row[1]):
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def put_records(provider, ticker, period, limit, end_date, records) -> None:
    """Cache the raw payload and explode it into the structured financial_data table."""
    if _disabled() or not records:
        return
    now = dt.datetime.now().isoformat(timespec="seconds")
    from . import common  # local import avoids any import-time cycle

    fd_rows = []
    for r in records:
        rp = r.get("report_period")
        for attr, val in r.items():
            if attr in _META or isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            fd_rows.append((ticker, rp, "raw", attr, provider, float(val), now))
    for m in common.build_financial_metrics(ticker, records):
        d = m.model_dump()
        rp = d.get("report_period")
        for attr, val in d.items():
            if attr in _META or val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            fd_rows.append((ticker, rp, "calc", attr, provider, float(val), now))

    try:
        with _LOCK, _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO provider_cache "
                "(provider, ticker, period, limit_n, end_date, payload, fetched_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (provider, ticker, period, limit, end_date, json.dumps(records), now),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO financial_data "
                "(ticker, period, kind, attribute, provider, value, fetched_at) "
                "VALUES (?,?,?,?,?,?,?)",
                fd_rows,
            )
    except sqlite3.Error:
        pass


def save_analysis_run(run: dict, results: list[dict]) -> str:
    """Persist an analysis run (run-level metadata + per-ticker/agent verdicts).

    `run` keys: run_id, run_time, tickers(list), agents(list), model,
                model_provider, data_provider, period_basis, start_date, end_date
    each `results` row: run_id, ticker, agent, signal, confidence, reasoning,
                intrinsic_value, market_cap, margin_of_safety, data_as_of,
                data_periods, period_basis, data_provider, run_time, metadata
    """
    init_db()
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analysis_runs "
            "(run_id, run_time, tickers, agents, model, model_provider, data_provider, "
            "period_basis, start_date, end_date) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run["run_id"], run["run_time"],
                json.dumps(run.get("tickers")), json.dumps(run.get("agents")),
                run.get("model"), run.get("model_provider"), run.get("data_provider"),
                run.get("period_basis"), run.get("start_date"), run.get("end_date"),
            ),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO analysis_results "
            "(run_id, ticker, agent, signal, confidence, reasoning, intrinsic_value, "
            "market_cap, margin_of_safety, data_as_of, data_periods, period_basis, "
            "data_provider, run_time, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["run_id"], r["ticker"], r["agent"], r.get("signal"), r.get("confidence"),
                    r.get("reasoning"), r.get("intrinsic_value"), r.get("market_cap"),
                    r.get("margin_of_safety"), r.get("data_as_of"), r.get("data_periods"),
                    r.get("period_basis"), r.get("data_provider"), r.get("run_time"),
                    r.get("metadata"),
                )
                for r in results
            ],
        )
    return run["run_id"]


init_db()
