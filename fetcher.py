"""AKShare data fetching with SQLite caching."""

import os
import sqlite3
import json
import re
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import akshare as ak
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# The DB lives on the WSL-native filesystem (ext4), NOT the project dir: the
# project sits on /mnt/c where every SQLite I/O crosses the 9p protocol —
# 10-100x slower, which made each Streamlit rerun spend ~10s just opening and
# reading the cache. Override with FUND_ANALYZER_DATA if needed.
_DATA_DIR = os.environ.get("FUND_ANALYZER_DATA") or os.path.join(
    os.path.expanduser("~"), ".local", "share", "fund-analyzer")
os.makedirs(_DATA_DIR, exist_ok=True)
CACHE_DB = os.path.join(_DATA_DIR, "fund_cache.db")
_LEGACY_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fund_cache.db")


def _migrate_db_location():
    """One-time move of an existing fund_cache.db out of the project dir.

    Uses the SQLite backup API (not a file copy) so pending WAL content is
    carried over intact; the legacy file is then removed.
    """
    if os.path.exists(CACHE_DB) or not os.path.exists(_LEGACY_DB):
        return
    logger.warning("migrating fund_cache.db to %s (one-time, ~373MB)", _DATA_DIR)
    src = sqlite3.connect(_LEGACY_DB)
    dst = sqlite3.connect(CACHE_DB)
    with dst:
        src.backup(dst)
    dst.close()
    src.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(_LEGACY_DB + suffix)
        except OSError:
            pass
FUND_LIST_TTL = 3600    # 1 hour
NAV_TTL = 86400         # 24 hours
NAV_START = "2025-01-01"  # NAV history is kept from this date onward
MAX_WORKERS = 8
RISK_FREE_RATE = 0.0113  # fallback 1-year China gov bond yield (see get_risk_free_rate)
RF_TTL = 30 * 86400      # auto-refresh the risk-free rate ~monthly
HOLDINGS_START_YEAR = 2025     # top holdings are shown from 2025Q1 onward
HOLDINGS_TTL = 7 * 86400       # current year re-checked weekly for new quarterly reports
HOLDINGS_TTL_PAST = 30 * 86400 # past years' disclosures barely change


# ── DB helpers ───────────────────────────────────────────────────────────────

def _conn():
    # timeout guards the threaded backfill: concurrent writers wait for the
    # lock instead of failing with "database is locked".
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    _migrate_db_location()
    conn = _conn()
    # WAL lets the app keep reading while the pipeline writes (and vice versa).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fund_list (
            id       INTEGER PRIMARY KEY,
            data     TEXT    NOT NULL,
            saved_at REAL    NOT NULL
        );
        -- One row per fund per day: appends are single-row INSERTs instead of
        -- rewriting a whole per-fund JSON blob (the old fund_nav design, which
        -- churned ~20KB of freelist pages per fund per update).
        CREATE TABLE IF NOT EXISTS fund_nav_daily (
            code          TEXT NOT NULL,
            date          TEXT NOT NULL,    -- ISO yyyy-mm-dd
            nav           REAL,
            daily_ret_pct REAL,
            acc_nav       REAL,
            PRIMARY KEY (code, date)
        ) WITHOUT ROWID;
        -- Per-fund freshness + newest stored date, so gap detection and TTL
        -- checks never have to scan fund_nav_daily.
        CREATE TABLE IF NOT EXISTS fund_nav_meta (
            code      TEXT PRIMARY KEY,
            saved_at  REAL NOT NULL,
            last_date TEXT
        );
        CREATE TABLE IF NOT EXISTS fund_sharpe (
            code        TEXT PRIMARY KEY,
            ann_return  REAL,
            volatility  REAL,
            sharpe      REAL,
            data_points INTEGER,
            saved_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_meta (
            key      TEXT PRIMARY KEY,
            value    REAL,
            saved_at REAL NOT NULL
        );
        -- Quarterly top-10 holdings (stocks + bonds) per fund, one year of
        -- quarters per row, stored as normalized JSON records.
        CREATE TABLE IF NOT EXISTS fund_holdings (
            code     TEXT NOT NULL,
            year     TEXT NOT NULL,
            data     TEXT NOT NULL,
            saved_at REAL NOT NULL,
            PRIMARY KEY (code, year)
        ) WITHOUT ROWID;
    """)
    # Add per-period max-drawdown and Sharpe columns (migration for existing DBs).
    for col in ("mdd_1m", "mdd_3m", "mdd_6m", "mdd_1y", "sharpe_6m", "sharpe_1y"):
        try:
            conn.execute(f"ALTER TABLE fund_sharpe ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
    _migrate_nav_blobs(conn)
    conn.commit()
    conn.close()


def _migrate_nav_blobs(conn):
    """One-time migration: legacy fund_nav JSON blobs → fund_nav_daily rows."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fund_nav'"
    ).fetchone():
        return
    for r in conn.execute("SELECT code, data, saved_at FROM fund_nav").fetchall():
        df = _nav_from_json(r["data"])
        if not df.empty and "date" in df.columns:
            _write_nav_rows(conn, r["code"], df, saved_at=r["saved_at"])
    conn.execute("DROP TABLE fund_nav")
    logger.info("migrated legacy fund_nav JSON blobs to fund_nav_daily")


# ── Risk-free rate ───────────────────────────────────────────────────────────
# The 1-year China government bond yield, fetched automatically and cached for a
# month, so nobody has to keep a number up to date by hand.

def _get_meta(key: str):
    conn = _conn()
    row = conn.execute("SELECT value, saved_at FROM app_meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return (row["value"], row["saved_at"]) if row else (None, None)


def _set_meta(key: str, value: float):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO app_meta (key, value, saved_at) VALUES (?, ?, ?)",
        (key, value, time.time()),
    )
    conn.commit()
    conn.close()


def _fetch_treasury_1y() -> Optional[float]:
    """Latest 1-year China government bond yield as a decimal (e.g. 0.0113)."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    df = ak.bond_china_yield(start_date=start, end_date=end)
    df = df[df["曲线名称"] == "中债国债收益率曲线"].sort_values("日期")
    val = pd.to_numeric(df["1年"], errors="coerce").dropna()
    return float(val.iloc[-1]) / 100.0 if not val.empty else None


def get_risk_free_rate(force_refresh: bool = False) -> float:
    """1-year China treasury yield as the risk-free rate, cached ~monthly.

    Falls back to the last cached value, then RISK_FREE_RATE, if the fetch fails.
    """
    value, saved_at = _get_meta("rf_rate")
    if not force_refresh and value is not None and (time.time() - saved_at) < RF_TTL:
        return value
    try:
        rf = _fetch_treasury_1y()
        if rf is not None and 0 < rf < 0.2:   # sanity bound
            _set_meta("rf_rate", rf)
            return rf
    except Exception as e:
        logger.debug("risk-free rate fetch failed: %s", e)
    return value if value is not None else RISK_FREE_RATE


def clear_all_caches():
    """Wipe every cache table: fund list, NAV history, computed Sharpe/drawdown.

    Used by the sidebar "清空所有缓存" button so a code/口径 change can be picked
    up cleanly — afterwards the list re-fetches and Sharpe/drawdown recompute on
    the next ⚡ run rather than being served stale from cache.
    """
    conn = _conn()
    conn.execute("DELETE FROM fund_list")
    conn.execute("DELETE FROM fund_nav_daily")
    conn.execute("DELETE FROM fund_nav_meta")
    conn.execute("DELETE FROM fund_sharpe")
    conn.execute("DELETE FROM fund_holdings")
    conn.commit()
    conn.close()


# Look-back windows in CALENDAR days, matching how EastMoney defines 近1月/3月/
# 6月/1年 (date-to-date from the latest NAV date), so the computed drawdown and
# Sharpe cover the same period as the 近X 收益率 columns shown alongside them.
DRAWDOWN_DAYS = {"mdd_1m": 30, "mdd_3m": 91, "mdd_6m": 182, "mdd_1y": 365}

# Sharpe is only computed for longer windows (short windows are too noisy).
SHARPE_DAYS = {"sharpe_6m": 182, "sharpe_1y": 365}

# Period returns (%), matching the rank list's 近1月/3月/6月/1年 columns; used
# when metrics are recomputed as of a past date and the list values don't apply.
RETURN_DAYS = {"ret_1m": 30, "ret_3m": 91, "ret_6m": 182, "ret_1y": 365}


# A window anchor may miss by a few days when the ideal start lands in a
# holiday gap or just before the stored history begins (data starts NAV_START,
# but 01-01 itself is a holiday). Accept the earliest NAV as anchor when it is
# at most this many days late; beyond that the fund is genuinely too young.
ANCHOR_GRACE_DAYS = 10


def _window_by_date(df: pd.DataFrame, days_back: int) -> Optional[pd.DataFrame]:
    """Rows from the anchor through the latest NAV, for a trailing date window.

    The anchor is the last NAV on or before (latest_date - days_back); it is the
    base point one period ago (e.g. the NAV "one year ago"). It is kept in the
    slice so it can serve as the drawdown peak candidate and as the base for the
    first in-window daily return. Returns None when the fund has no NAV old
    enough to anchor the window (e.g. a fund younger than the period).

    `df` must be sorted ascending by `date` with a 0..n-1 RangeIndex.
    """
    end_date = df["date"].max()
    start_date = end_date - timedelta(days=days_back)
    older = df[df["date"] <= start_date]
    if older.empty:
        first = df["date"].iloc[0]
        if (first - start_date).days <= ANCHOR_GRACE_DAYS:
            return df
        return None
    return df.loc[older.index[-1]:]


def _max_drawdown(nav: pd.Series) -> Optional[float]:
    """Max drawdown magnitude (positive fraction) for an ascending NAV series."""
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2:
        return None
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    return float(-dd.min())


def _annualized(r: pd.Series, span_days: int, rf: float):
    """(annual_return, annual_vol, sharpe) from a daily-return series.

    Everything is derived from the window itself — no fixed trading-day constant
    — so it adapts to A-shares, QDII/US funds, HK, etc. automatically:
      • return: the actual compounded return over the window, annualized by its
        real calendar span (a full year keeps its real return, no inflation);
      • volatility: daily σ × √(observations per year), where observations-per-
        year is *measured* from how many NAV points actually fell in the window
        rather than assumed to be 252.
    None if degenerate.
    """
    r = r.dropna()
    n = len(r)
    if n < 2:
        return None
    std_daily = r.std(ddof=1)
    if std_daily == 0 or np.isnan(std_daily):
        return None
    span = max(span_days, 1)
    total_growth = float((1.0 + r).prod())          # e.g. 4.38 = +338%
    ann_return = total_growth ** (365.0 / span) - 1.0
    obs_per_year = n * 365.0 / span                  # measured, not the 252 convention
    ann_vol = std_daily * np.sqrt(obs_per_year)
    return ann_return, ann_vol, (ann_return - rf) / ann_vol


def _period_sharpe(df: pd.DataFrame, days_back: int, rf: float) -> Optional[float]:
    """Annualized Sharpe over the trailing `days_back` calendar-day window.

    Returns None when the fund lacks enough history to cover the window.
    """
    window = _window_by_date(df, days_back)
    if window is None:
        return None
    # Drop the anchor's own return: it happened the day before the window opens.
    r = window["r"].iloc[1:].dropna()
    if len(r) < int(days_back / 365 * 200):
        return None
    span = (window["date"].iloc[-1] - window["date"].iloc[0]).days
    res = _annualized(r, span, rf)
    return float(res[2]) if res else None


def _period_return(df: pd.DataFrame, days_back: int) -> Optional[float]:
    """Compounded % return over the trailing `days_back` calendar-day window
    (daily growth rates multiplied up, so dividends are handled). None when
    the fund lacks history old enough to anchor the window."""
    window = _window_by_date(df, days_back)
    if window is None:
        return None
    r = window["r"].iloc[1:].dropna()
    if r.empty:
        return None
    return float(((1.0 + r).prod() - 1.0) * 100.0)


def _period_mdd(df: pd.DataFrame, days_back: int) -> Optional[float]:
    """Max drawdown over the trailing `days_back` window, on accumulated NAV."""
    window = _window_by_date(df, days_back)
    if window is None:
        return None
    return _max_drawdown(window["acc_nav"])


# ── Fund list ────────────────────────────────────────────────────────────────

# The fund list is a single cached snapshot (always read/written whole), so it
# lives in one fixed row (id=1) and is upserted in place.
def _load_fund_list_cache() -> Optional[pd.DataFrame]:
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM fund_list WHERE id = 1"
    ).fetchone()
    conn.close()
    if row and (time.time() - row["saved_at"]) < FUND_LIST_TTL:
        return pd.DataFrame(json.loads(row["data"]))
    return None


def _save_fund_list_cache(df: pd.DataFrame):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_list (id, data, saved_at) VALUES (1, ?, ?)",
        (df.to_json(orient="records", force_ascii=False), time.time()),
    )
    conn.commit()
    conn.close()


def fetch_fund_list(force_refresh: bool = False) -> pd.DataFrame:
    """Return all open-end funds with basic performance from EastMoney.

    Uses ak.fund_open_fund_rank_em(symbol='全部') which returns:
    序号, 基金代码, 基金简称, 日期, 单位净值, 累计净值, 日增长率,
    近1周, 近1月, 近3月, 近6月, 近1年, 近2年, 近3年, 今年来, 成立来, 手续费
    """
    if not force_refresh:
        cached = _load_fund_list_cache()
        if cached is not None:
            return cached

    df = ak.fund_open_fund_rank_em(symbol="全部")
    df = df.rename(columns={
        "基金代码": "code",
        "基金简称": "name",
        "日期": "nav_date",
        "单位净值": "nav",
        "累计净值": "acc_nav",
        "日增长率": "daily_ret",
        "近1周": "ret_1w",
        "近1月": "ret_1m",
        "近3月": "ret_3m",
        "近6月": "ret_6m",
        "近1年": "ret_1y",
        "近2年": "ret_2y",
        "近3年": "ret_3y",
        "今年来": "ret_ytd",
        "成立来": "ret_inception",
        "手续费": "fee",
    })

    # fund_open_fund_rank_em doesn't include type; merge with fund_name_em
    try:
        name_df = ak.fund_name_em()[["基金代码", "基金类型"]].rename(
            columns={"基金代码": "code", "基金类型": "type"}
        )
        df = df.merge(name_df, on="code", how="left")
    except Exception:
        df["type"] = "未知"

    df["ret_1y_pct"] = pd.to_numeric(df["ret_1y"], errors="coerce")

    _save_fund_list_cache(df)
    return df


# ── NAV history ──────────────────────────────────────────────────────────────

def _nav_from_json(blob: str) -> pd.DataFrame:
    """Rebuild a NAV DataFrame from a legacy fund_nav JSON blob (migration only).

    df.to_json serialized datetimes as epoch-millisecond ints, which
    pd.to_datetime would otherwise misread as nanoseconds (everything → 1970).
    Newer rows were stored ISO-formatted; handle both shapes.
    """
    df = pd.DataFrame(json.loads(blob))
    if not df.empty and "date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="ms")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _write_nav_rows(conn, code: str, df: pd.DataFrame,
                    saved_at: Optional[float] = None):
    """Upsert NAV rows for one fund and refresh its meta row. No commit —
    the caller owns the transaction."""
    df = df.dropna(subset=["date"])
    if df.empty:
        return
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    nav = pd.to_numeric(df["nav"], errors="coerce")
    ret = pd.to_numeric(df["daily_ret_pct"], errors="coerce") \
        if "daily_ret_pct" in df.columns else pd.Series(np.nan, index=df.index)
    acc = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(nav) \
        if "acc_nav" in df.columns else nav
    rows = [
        (code, d,
         None if pd.isna(n) else float(n),
         None if pd.isna(r) else float(r),
         None if pd.isna(a) else float(a))
        for d, n, r, a in zip(dates, nav, ret, acc)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO fund_nav_daily "
        "(code, date, nav, daily_ret_pct, acc_nav) VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_meta (code, saved_at, last_date) "
        "VALUES (?, ?, (SELECT MAX(date) FROM fund_nav_daily WHERE code=?))",
        (code, saved_at if saved_at is not None else time.time(), code))


def _load_nav_df(code: str, conn=None) -> pd.DataFrame:
    """Stored NAV history for one fund, ascending by date.

    Columns: date (datetime64), nav, daily_ret_pct, acc_nav.
    """
    own = conn is None
    if own:
        conn = _conn()
    df = pd.read_sql_query(
        "SELECT date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
        "WHERE code = ? ORDER BY date", conn, params=(code,))
    if own:
        conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


_EM_PZD_URL = "https://fund.eastmoney.com/pingzhongdata/{code}.js"


def _fetch_nav_full(code: str) -> Optional[pd.DataFrame]:
    """NAV history since NAV_START in a single request via pingzhongdata.

    The one js blob carries unit NAV, daily growth AND accumulated NAV, so this
    replaces akshare's fund_open_fund_info_em (which downloads the same blob
    once per indicator and parses it with a JS engine) — the two series are
    pulled out with a regex + json.loads instead.
    Columns: date, nav, daily_ret_pct, acc_nav (ascending). None on failure.
    """
    def _dates(ms):
        # timestamps are midnight Beijing time; naive UTC parse would land on
        # the previous day, so convert before dropping the timezone
        s = pd.to_datetime(ms, unit="ms", utc=True)
        return s.dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None)

    try:
        r = requests.get(_EM_PZD_URL.format(code=code),
                         headers=_EM_LSJZ_HEADERS, timeout=20)
        m = re.search(r"var Data_netWorthTrend\s*=\s*(\[.*?\])\s*;", r.text)
        if not m:
            return None
        unit = pd.DataFrame(json.loads(m.group(1)))
        if unit.empty:
            return None
        df = pd.DataFrame({
            "date": _dates(unit["x"]),
            "nav": pd.to_numeric(unit["y"], errors="coerce"),
            "daily_ret_pct": pd.to_numeric(
                unit.get("equityReturn"), errors="coerce"),
        })

        # Accumulated (dividend-reinvested) NAV for drawdown; fall back to unit
        # NAV where the accumulated series is missing.
        m = re.search(r"var Data_ACWorthTrend\s*=\s*(\[.*?\])\s*;", r.text)
        acc_raw = json.loads(m.group(1)) if m else []
        if acc_raw:
            acc = pd.DataFrame(acc_raw, columns=["x", "acc_nav"])
            acc["date"] = _dates(acc["x"])
            df = df.merge(acc[["date", "acc_nav"]], on="date", how="left")
        else:
            df["acc_nav"] = df["nav"]
        df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(df["nav"])

        df = df[df["date"] >= pd.Timestamp(NAV_START)]
        return df.sort_values("date").reset_index(drop=True)

    except Exception as e:
        logger.debug("full NAV fetch failed for %s: %s", code, e)
        return None


def fetch_nav(code: str) -> Optional[pd.DataFrame]:
    """Return NAV history since NAV_START for a single fund (cache-first).

    Serves stored rows while fresh (< NAV_TTL); otherwise downloads the whole
    history in one pingzhongdata request and stores it row-per-day.
    Columns: date, nav, daily_ret_pct, acc_nav
    """
    conn = _conn()
    meta = conn.execute(
        "SELECT saved_at FROM fund_nav_meta WHERE code = ?", (code,)).fetchone()
    if meta and (time.time() - meta["saved_at"]) < NAV_TTL:
        df = _load_nav_df(code, conn)
        conn.close()
        return df if len(df) >= 20 else None
    conn.close()

    df = _fetch_nav_full(code)
    if df is None or len(df) < 20:
        return None
    conn = _conn()
    _write_nav_rows(conn, code, df)
    conn.commit()
    conn.close()
    return df


# ── Sharpe calculation ────────────────────────────────────────────────────────

def _save_sharpe(code: str, ann_return: float, volatility: float, sharpe: float,
                 n: int, mdd: dict, psharpe: dict):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_sharpe "
        "(code, ann_return, volatility, sharpe, data_points, "
        " mdd_1m, mdd_3m, mdd_6m, mdd_1y, sharpe_6m, sharpe_1y, saved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (code, ann_return, volatility, sharpe, n,
         mdd.get("mdd_1m"), mdd.get("mdd_3m"), mdd.get("mdd_6m"), mdd.get("mdd_1y"),
         psharpe.get("sharpe_6m"), psharpe.get("sharpe_1y"),
         time.time()),
    )
    conn.commit()
    conn.close()


def _metrics_from_nav(nav_df: pd.DataFrame, rf: float) -> Optional[dict]:
    """Compute annualized return / volatility / Sharpe + per-period Sharpe and
    max-drawdown from an already-fetched NAV DataFrame. Pure (no I/O)."""
    df = nav_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    # daily_ret_pct is percentage; convert to decimal
    if "daily_ret_pct" in df.columns:
        df["r"] = pd.to_numeric(df["daily_ret_pct"], errors="coerce") / 100.0
    else:
        df["r"] = df["nav"].pct_change()
    if "acc_nav" in df.columns:
        df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(df["nav"])
    else:
        df["acc_nav"] = df["nav"]
    df = df.sort_values("date").reset_index(drop=True)

    returns = df["r"].dropna()
    if len(returns) < 20:
        return None

    n = len(returns)
    span = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    res = _annualized(returns, span, rf)
    if res is None:
        return None
    ann_return, ann_vol, sharpe = res

    # Per-period Sharpe and max drawdown over trailing calendar-day windows,
    # date-aligned with EastMoney's 近X 收益率. A fund younger than a window
    # gets None for it (no anchor) rather than a misleadingly short reading.
    # Drawdown runs on accumulated NAV so dividends aren't mistaken for drops.
    psharpe = {key: _period_sharpe(df, days, rf) for key, days in SHARPE_DAYS.items()}
    mdd = {key: _period_mdd(df, days) for key, days in DRAWDOWN_DAYS.items()}
    rets = {key: _period_return(df, days) for key, days in RETURN_DAYS.items()}

    return {
        "ann_return": ann_return, "volatility": ann_vol, "sharpe": sharpe,
        "data_points": n, **mdd, **psharpe, **rets,
    }


def _save_metrics(code: str, m: dict):
    _save_sharpe(
        code, m["ann_return"], m["volatility"], m["sharpe"], m["data_points"],
        {k: m[k] for k in ("mdd_1m", "mdd_3m", "mdd_6m", "mdd_1y")},
        {k: m[k] for k in ("sharpe_6m", "sharpe_1y")},
    )


def compute_sharpe_for_fund(code: str, rf: float = RISK_FREE_RATE) -> Optional[dict]:
    """Fetch NAV (network/cache), compute metrics, persist, and return them."""
    nav_df = fetch_nav(code)
    if nav_df is None:
        return None
    m = _metrics_from_nav(nav_df, rf)
    if m is not None:
        _save_metrics(code, m)
    return m


# ── Quarterly top holdings ───────────────────────────────────────────────────
# EastMoney F10 discloses each fund's top-10 stock/bond holdings per quarter.
# Fetched one year at a time (the API's granularity) and cached per (code, year).

_HOLDINGS_COLS = ["quarter", "kind", "代码", "名称", "占净值比例", "持股数", "持仓市值"]


def _fetch_holdings_year(code: str, year: str) -> Optional[pd.DataFrame]:
    """One year's quarterly top holdings (stocks + bonds), normalized.

    Returns an empty DataFrame when the fund disclosed nothing that year, or
    None when both requests failed (network error — caller keeps stale cache).
    """
    frames, failures = [], 0
    for kind, fn, code_col, name_col in (
        ("股票", ak.fund_portfolio_hold_em, "股票代码", "股票名称"),
        ("债券", ak.fund_portfolio_bond_hold_em, "债券代码", "债券名称"),
    ):
        try:
            raw = fn(symbol=code, date=year)
        except Exception as e:
            logger.debug("holdings fetch failed %s %s %s: %s", code, year, kind, e)
            failures += 1
            continue
        if raw is None or raw.empty:
            continue
        # 季度 looks like "2025年1季度股票投资明细" → "2025Q1"
        q = raw["季度"].astype(str).str.extract(r"(\d{4})年(\d)季度")
        df = pd.DataFrame({
            "quarter": q[0] + "Q" + q[1],
            "kind": kind,
            "代码": raw[code_col].astype(str),
            "名称": raw[name_col].astype(str),
            "占净值比例": pd.to_numeric(raw["占净值比例"], errors="coerce"),
            "持股数": pd.to_numeric(raw["持股数"], errors="coerce")
                if "持股数" in raw.columns else np.nan,
            "持仓市值": pd.to_numeric(raw["持仓市值"], errors="coerce"),
        })
        frames.append(df.dropna(subset=["quarter"]))
    if failures == 2:
        return None
    if not frames:
        return pd.DataFrame(columns=_HOLDINGS_COLS)
    return pd.concat(frames, ignore_index=True)


def fetch_holdings(code: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Quarterly top holdings from 2025Q1 (HOLDINGS_START_YEAR) to the latest
    disclosed quarter, cache-first.

    Columns: quarter ("2025Q1"), kind (股票/债券), 代码, 名称, 占净值比例(%),
    持股数(万股, stocks only), 持仓市值(万元). Sorted newest quarter first,
    biggest position first within a quarter. Returns None only when nothing
    could be fetched and no cache exists.
    """
    years = [str(y) for y in range(HOLDINGS_START_YEAR, datetime.now().year + 1)]
    frames, any_data = [], False
    conn = _conn()
    for year in years:
        row = conn.execute(
            "SELECT data, saved_at FROM fund_holdings WHERE code=? AND year=?",
            (code, year)).fetchone()
        ttl = HOLDINGS_TTL if year == str(datetime.now().year) else HOLDINGS_TTL_PAST
        if row and not force_refresh and (time.time() - row["saved_at"]) < ttl:
            df = pd.DataFrame(json.loads(row["data"]), columns=_HOLDINGS_COLS)
        else:
            df = _fetch_holdings_year(code, year)
            if df is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO fund_holdings "
                    "(code, year, data, saved_at) VALUES (?, ?, ?, ?)",
                    (code, year, df.to_json(orient="records", force_ascii=False),
                     time.time()))
                conn.commit()
            elif row:   # fetch failed → serve the stale cache rather than nothing
                df = pd.DataFrame(json.loads(row["data"]), columns=_HOLDINGS_COLS)
        if df is not None:
            any_data = True
            if not df.empty:
                frames.append(df)
    conn.close()
    if not any_data:
        return None
    if not frames:
        return pd.DataFrame(columns=_HOLDINGS_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["quarter", "kind", "占净值比例"], ascending=[False, True, False]
    ).reset_index(drop=True)


# ── Daily-batch pipeline ──────────────────────────────────────────────────────
# Used by update_daily.py: backfill once, then each day append the latest NAV
# point (from the bulk fund-list call) and recompute Sharpe/drawdown for all.


def list_nav_codes() -> set:
    """Codes that already have a stored NAV history."""
    conn = _conn()
    rows = conn.execute("SELECT code FROM fund_nav_meta").fetchall()
    conn.close()
    return {r["code"] for r in rows}


_EM_LSJZ_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_EM_LSJZ_HEADERS = {"Referer": "https://fundf10.eastmoney.com/"}


def _fetch_nav_range(code: str, start: datetime,
                     end: datetime) -> Optional[pd.DataFrame]:
    """NAV rows in [start, end] via EastMoney's date-ranged 历史净值 API.

    One paged JSON request (~KB) carries unit NAV, accumulated NAV and daily
    growth together — unlike ak.fund_open_fund_info_em, which downloads the
    fund's entire since-inception pingzhongdata blob (~MB) once per indicator.
    Returns an empty DataFrame when the range has no rows, None on failure.
    Columns: date, nav, daily_ret_pct, acc_nav (ascending by date).
    """
    rows, page = [], 1
    try:
        while True:
            resp = requests.get(_EM_LSJZ_URL, headers=_EM_LSJZ_HEADERS, timeout=15,
                                params={
                                    "fundCode": code,
                                    "pageIndex": page,
                                    "pageSize": 49,
                                    "startDate": start.strftime("%Y-%m-%d"),
                                    "endDate": end.strftime("%Y-%m-%d"),
                                })
            payload = resp.json()
            data = payload.get("Data")
            if not isinstance(data, dict):   # ErrCode -999 etc.
                return None
            batch = data.get("LSJZList") or []
            rows.extend(batch)
            if not batch or len(rows) >= (payload.get("TotalCount") or 0):
                break
            page += 1
    except Exception as e:
        logger.debug("ranged NAV fetch failed for %s: %s", code, e)
        return None

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = pd.DataFrame({
        "date": pd.to_datetime(df["FSRQ"], errors="coerce"),
        "nav": pd.to_numeric(df["DWJZ"], errors="coerce"),
        "daily_ret_pct": pd.to_numeric(df["JZZZL"], errors="coerce"),
        "acc_nav": pd.to_numeric(df["LJJZ"], errors="coerce"),
    })
    df = df.dropna(subset=["date", "nav"])
    df["acc_nav"] = df["acc_nav"].fillna(df["nav"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_nav_incremental(code: str, after_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch NAV rows newer than `after_date` for a single fund.

    Asks the ranged API for just the missing span; falls back to the full
    single-request download only if the ranged API fails.
    """
    start = max(after_date + timedelta(days=1), pd.Timestamp(NAV_START))
    df = _fetch_nav_range(code, start, datetime.now())
    if df is None:
        df = _fetch_nav_full(code)
    if df is None:
        return None
    df = df[df["date"] > after_date].reset_index(drop=True)
    return df if not df.empty else None


def _backfill_incremental(codes: list, workers: int = MAX_WORKERS,
                          progress: Optional[Callable] = None) -> int:
    """Incrementally fill NAV gaps for `codes` (threaded).

    For each code, reads the last stored date, fetches only newer rows, and
    inserts them.  Returns the count of codes that got new data.
    """
    total, done, patched = len(codes), 0, 0
    if not codes:
        return 0

    def _patch_one(code: str) -> bool:
        conn = _conn()
        meta = conn.execute(
            "SELECT last_date FROM fund_nav_meta WHERE code=?", (code,)).fetchone()
        conn.close()
        if not meta or not meta["last_date"]:
            return False

        new_rows = _fetch_nav_incremental(code, pd.to_datetime(meta["last_date"]))
        if new_rows is None or new_rows.empty:
            return False

        conn = _conn()
        _write_nav_rows(conn, code, new_rows)
        conn.commit()
        conn.close()
        return True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_patch_one, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                if fut.result():
                    patched += 1
            except Exception:
                pass
            if progress and (done % 50 == 0 or done == total):
                progress(done, total)
    return patched


def _only_weekends_between(a: datetime, b: datetime) -> bool:
    """True if every calendar day strictly between a and b is a Sat/Sun.

    Used to tell "the fund list's latest NAV is the only missing point"
    (consecutive trading days, possibly across a weekend) from a real
    multi-day gap. Holidays make this return False, which safely falls
    through to the ranged fetch.
    """
    d = pd.Timestamp(a).normalize() + timedelta(days=1)
    end = pd.Timestamp(b).normalize()
    while d < end:
        if d.weekday() < 5:
            return False
        d += timedelta(days=1)
    return True


def _append_nav_point(conn, code: str, date, nav: float,
                      acc_nav: Optional[float], ret_pct: Optional[float]) -> bool:
    """Append one NAV row (from the bulk fund-list call) — a single INSERT,
    zero network. The caller owns the transaction (no commit here)."""
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    if ret_pct is None:
        prev = conn.execute(
            "SELECT nav FROM fund_nav_daily WHERE code=? AND date<? "
            "ORDER BY date DESC LIMIT 1", (code, d)).fetchone()
        if prev and prev["nav"]:
            ret_pct = (nav / prev["nav"] - 1.0) * 100.0
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_daily "
        "(code, date, nav, daily_ret_pct, acc_nav) VALUES (?, ?, ?, ?, ?)",
        (code, d, nav, ret_pct, acc_nav if acc_nav is not None else nav))
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav_meta (code, saved_at, last_date) "
        "VALUES (?, ?, ?)", (code, time.time(), d))
    return True


def append_incremental(list_df: pd.DataFrame,
                       progress: Optional[Callable] = None) -> dict:
    """Bring every stored NAV history up to the fund list's latest date.

    Two tiers, cheapest first:
      • gap of exactly one trading day (only weekends in between) — the bulk
        fund-list call already carries that day's nav/acc_nav/daily_ret, so
        the point is appended directly, zero extra network;
      • bigger gap (holidays, many days since last run) — fetch just the
        missing date span via the ranged 历史净值 API (threaded).

    Gap detection reads fund_nav_meta.last_date — one small table scan.
    `progress(phase, done, total)` as in run_pipeline. Returns a summary dict
    with counts (failed = gapped funds whose ranged fetch returned nothing).
    """
    def _p(phase, done, total):
        if progress:
            progress(phase, done, total)

    latest = list_df.dropna(subset=["code"]).drop_duplicates("code").set_index("code")

    conn = _conn()
    stored = {r["code"]: r["last_date"]
              for r in conn.execute("SELECT code, last_date FROM fund_nav_meta")}

    appended = skipped = 0
    gapped = []
    total = len(stored)
    for i, (code, last_iso) in enumerate(stored.items()):
        if i % 500 == 0 or i == total - 1:
            _p("追加当日净值", i + 1, total)
        if not last_iso or code not in latest.index:
            skipped += 1
            continue
        row = latest.loc[code]
        new_date = pd.to_datetime(row["nav_date"], errors="coerce")
        nav = pd.to_numeric(row["nav"], errors="coerce")
        if pd.isna(new_date) or pd.isna(nav):
            skipped += 1
            continue
        last_date = pd.to_datetime(last_iso)
        if new_date <= last_date:
            skipped += 1                       # already up-to-date
            continue

        if _only_weekends_between(last_date, new_date):
            acc = pd.to_numeric(row.get("acc_nav"), errors="coerce")
            ret = pd.to_numeric(row.get("daily_ret"), errors="coerce")
            if _append_nav_point(conn, code, new_date, float(nav),
                                 None if pd.isna(acc) else float(acc),
                                 None if pd.isna(ret) else float(ret)):
                appended += 1
                continue
        gapped.append(code)                    # real gap → ranged fetch
    conn.commit()
    conn.close()

    patched = 0
    if gapped:
        patched = _backfill_incremental(
            gapped, MAX_WORKERS, lambda d, t: _p("增量补缺口", d, t)
        )

    return {"appended": appended, "patched": patched, "skipped": skipped,
            "gap_codes": len(gapped), "failed": len(gapped) - patched}


def recompute_all(rf: Optional[float] = None,
                  progress_callback: Optional[Callable] = None) -> int:
    """Recompute Sharpe + drawdown for every stored fund from its stored NAV.

    No network for NAV — pure CPU over cached NAV. `rf` defaults to the
    auto-fetched (monthly-cached) risk-free rate. All reads/writes share one
    connection and a single commit, so 20k funds finish in seconds.
    """
    if rf is None:
        rf = get_risk_free_rate()
    conn = _conn()
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav_meta").fetchall()]
    total, done, saved = len(codes), 0, 0
    for code in codes:
        done += 1
        try:
            nav_df = _load_nav_df(code, conn)
            m = _metrics_from_nav(nav_df, rf) if not nav_df.empty else None
        except Exception as e:
            logger.debug("recompute parse error %s: %s", code, e)
            m = None
        if m is not None:
            conn.execute(
                "INSERT OR REPLACE INTO fund_sharpe "
                "(code, ann_return, volatility, sharpe, data_points, "
                " mdd_1m, mdd_3m, mdd_6m, mdd_1y, sharpe_6m, sharpe_1y, saved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (code, m["ann_return"], m["volatility"], m["sharpe"], m["data_points"],
                 m["mdd_1m"], m["mdd_3m"], m["mdd_6m"], m["mdd_1y"],
                 m["sharpe_6m"], m["sharpe_1y"], time.time()),
            )
            saved += 1
        if progress_callback and (done % 500 == 0 or done == total):
            progress_callback(done, total)
    conn.commit()
    conn.close()
    return saved


def compute_metrics_asof(asof: str, rf: Optional[float] = None,
                         progress_callback: Optional[Callable] = None) -> dict:
    """Every fund's metrics as they stood on a past date, from stored NAV only.

    Truncates each fund's history to rows on/before `asof` (ISO yyyy-mm-dd) and
    recomputes period returns, Sharpe and drawdown over the same trailing
    windows — no network, nothing persisted. Funds with under 20 NAV points by
    that date are omitted. Returns {code: metrics-dict}.
    """
    if rf is None:
        rf = get_risk_free_rate()
    conn = _conn()
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav_meta")]
    out, total = {}, len(codes)
    for i, code in enumerate(codes):
        df = pd.read_sql_query(
            "SELECT date, nav, daily_ret_pct, acc_nav FROM fund_nav_daily "
            "WHERE code = ? AND date <= ? ORDER BY date",
            conn, params=(code, asof))
        if len(df) >= 20:
            df["date"] = pd.to_datetime(df["date"])
            try:
                m = _metrics_from_nav(df, rf)
            except Exception as e:
                logger.debug("asof metrics failed %s: %s", code, e)
                m = None
            if m is not None:
                out[code] = m
        if progress_callback and ((i + 1) % 500 == 0 or i + 1 == total):
            progress_callback(i + 1, total)
    conn.close()
    return out


def load_all_precomputed() -> dict:
    """Every precomputed Sharpe/drawdown row, keyed by code, ignoring TTL.

    Lets the app show metrics instantly on load; freshness is the daily
    pipeline's responsibility (surfaced via last_update_time()).
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT code, ann_return, volatility, sharpe, data_points, "
        "mdd_1m, mdd_3m, mdd_6m, mdd_1y, sharpe_6m, sharpe_1y FROM fund_sharpe"
    ).fetchall()
    conn.close()
    return {r["code"]: {k: r[k] for k in r.keys() if k != "code"} for r in rows}


def last_update_time() -> Optional[float]:
    """Unix time of the most recent precomputed metric, or None if empty."""
    conn = _conn()
    row = conn.execute("SELECT MAX(saved_at) AS t FROM fund_sharpe").fetchone()
    conn.close()
    return row["t"] if row and row["t"] else None


def _backfill_codes(codes: list, workers: int = MAX_WORKERS,
                    progress: Optional[Callable] = None) -> int:
    """Download full ~1y NAV history for `codes` (threaded). Returns success count."""
    total, done, ok = len(codes), 0, 0
    if not codes:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_nav, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                if fut.result() is not None:
                    ok += 1
            except Exception:
                pass
            if progress and (done % 50 == 0 or done == total):
                progress(done, total)
    return ok


def run_pipeline(progress: Optional[Callable] = None, do_backfill: bool = True,
                 rf: Optional[float] = None, workers: int = MAX_WORKERS) -> dict:
    """Full daily pipeline shared by the CLI script and the in-app button:
    ① refresh fund list → ② backfill funds missing history → ③ append the
    list's latest NAV point / range-fetch bigger gaps → ④ recompute
    Sharpe/drawdown for all.

    `rf` defaults to the auto-fetched (monthly-cached) risk-free rate.
    `progress(phase, done, total)` is invoked throughout for a UI bar / logging.
    Returns a summary dict.
    """
    if rf is None:
        rf = get_risk_free_rate()

    def _p(phase, done, total):
        if progress:
            progress(phase, done, total)

    _p("拉取基金列表", 0, 1)
    list_df = fetch_fund_list(force_refresh=True)
    all_codes = list_df["code"].dropna().unique().tolist()
    _p("拉取基金列表", 1, 1)

    backfilled = 0
    if do_backfill:
        have = list_nav_codes()
        todo = [c for c in all_codes if c not in have]
        backfilled = _backfill_codes(
            todo, workers, lambda d, t: _p("回填缺失历史", d, t)
        )

    res = append_incremental(list_df, progress=_p)

    saved = recompute_all(rf=rf, progress_callback=lambda d, t: _p("重算指标", d, t))

    return {
        "funds": len(all_codes),
        "backfilled": backfilled,
        "appended": res["appended"],
        "patched": res["patched"],
        "failed": res["failed"],
        "recomputed": saved,
        "rf": rf,
    }
