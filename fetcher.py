"""AKShare data fetching with SQLite caching."""

import sqlite3
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DB = "fund_cache.db"
FUND_LIST_TTL = 3600    # 1 hour
NAV_TTL = 86400         # 24 hours
MAX_WORKERS = 8
RISK_FREE_RATE = 0.0113  # 1-year China gov bond yield as of 2026-06


# ── DB helpers ───────────────────────────────────────────────────────────────

def _conn():
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fund_list (
            id       INTEGER PRIMARY KEY,
            data     TEXT    NOT NULL,
            saved_at REAL    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fund_nav (
            code     TEXT PRIMARY KEY,
            data     TEXT NOT NULL,
            saved_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fund_sharpe (
            code        TEXT PRIMARY KEY,
            ann_return  REAL,
            volatility  REAL,
            sharpe      REAL,
            data_points INTEGER,
            saved_at    REAL NOT NULL
        );
    """)
    # Add per-period max-drawdown and Sharpe columns (migration for existing DBs).
    for col in ("mdd_1m", "mdd_3m", "mdd_6m", "mdd_1y", "sharpe_6m", "sharpe_1y"):
        try:
            conn.execute(f"ALTER TABLE fund_sharpe ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


# Approximate trading days per look-back window, used for max-drawdown slices.
DRAWDOWN_WINDOWS = {"mdd_1m": 21, "mdd_3m": 63, "mdd_6m": 126, "mdd_1y": 252}

# Sharpe is only computed for longer windows (short windows are too noisy).
SHARPE_WINDOWS = {"sharpe_6m": 126, "sharpe_1y": 252}

TRADING_DAYS = 252


def _max_drawdown(nav: pd.Series) -> Optional[float]:
    """Max drawdown magnitude (positive fraction) for an ascending NAV series."""
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2:
        return None
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    return float(-dd.min())


def _period_sharpe(r: pd.Series, days: int, rf: float) -> Optional[float]:
    """Annualized Sharpe over the most recent `days` daily returns.

    Returns None when the fund lacks enough history to cover the window.
    """
    r = r.tail(days)
    if len(r) < days * 0.8:
        return None
    mean_daily = r.mean()
    std_daily = r.std(ddof=1)
    if std_daily == 0 or np.isnan(std_daily):
        return None
    ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
    ann_vol = std_daily * np.sqrt(TRADING_DAYS)
    return float((ann_return - rf) / ann_vol)


# ── Fund list ────────────────────────────────────────────────────────────────

def _load_fund_list_cache() -> Optional[pd.DataFrame]:
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM fund_list ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row and (time.time() - row["saved_at"]) < FUND_LIST_TTL:
        return pd.DataFrame(json.loads(row["data"]))
    return None


def _save_fund_list_cache(df: pd.DataFrame):
    conn = _conn()
    conn.execute("DELETE FROM fund_list")
    conn.execute(
        "INSERT INTO fund_list (data, saved_at) VALUES (?, ?)",
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

def _load_nav_cache(code: str) -> Optional[pd.DataFrame]:
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM fund_nav WHERE code = ?", (code,)
    ).fetchone()
    conn.close()
    if row and (time.time() - row["saved_at"]) < NAV_TTL:
        return pd.DataFrame(json.loads(row["data"]))
    return None


def _save_nav_cache(code: str, df: pd.DataFrame):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav (code, data, saved_at) VALUES (?, ?, ?)",
        (code, df.to_json(orient="records", force_ascii=False), time.time()),
    )
    conn.commit()
    conn.close()


def fetch_nav(code: str) -> Optional[pd.DataFrame]:
    """Return ~1-year NAV history for a single fund.

    Uses ak.fund_open_fund_info_em(symbol, indicator='单位净值走势').
    Columns returned: 净值日期, 单位净值, 日增长率
    """
    cached = _load_nav_cache(code)
    if cached is not None:
        return cached

    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "净值日期": "date",
            "单位净值": "nav",
            "日增长率": "daily_ret_pct",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Filter to last 365 + buffer days
        cutoff = datetime.now() - timedelta(days=380)
        df = df[df["date"] >= cutoff].copy()

        if len(df) < 20:
            return None

        _save_nav_cache(code, df)
        return df

    except Exception as e:
        logger.debug("NAV fetch failed for %s: %s", code, e)
        return None


# ── Sharpe calculation ────────────────────────────────────────────────────────

def _load_sharpe_cache(codes: list) -> dict:
    if not codes:
        return {}
    conn = _conn()
    placeholders = ",".join("?" * len(codes))
    rows = conn.execute(
        f"SELECT code, ann_return, volatility, sharpe, data_points, "
        f"mdd_1m, mdd_3m, mdd_6m, mdd_1y, sharpe_6m, sharpe_1y, saved_at "
        f"FROM fund_sharpe WHERE code IN ({placeholders})",
        codes,
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        # Skip stale rows, and rows predating the per-period columns so they
        # get recomputed (sharpe_1y is populated once those exist).
        if (time.time() - r["saved_at"]) < NAV_TTL and r["sharpe_1y"] is not None:
            result[r["code"]] = {
                "ann_return": r["ann_return"],
                "volatility": r["volatility"],
                "sharpe": r["sharpe"],
                "data_points": r["data_points"],
                "mdd_1m": r["mdd_1m"],
                "mdd_3m": r["mdd_3m"],
                "mdd_6m": r["mdd_6m"],
                "mdd_1y": r["mdd_1y"],
                "sharpe_6m": r["sharpe_6m"],
                "sharpe_1y": r["sharpe_1y"],
            }
    return result


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


def compute_sharpe_for_fund(code: str, rf: float = RISK_FREE_RATE) -> Optional[dict]:
    """Fetch NAV, compute annualized return / volatility / Sharpe."""
    nav_df = fetch_nav(code)
    if nav_df is None:
        return None

    df = nav_df.copy()
    # daily_ret_pct is percentage; convert to decimal
    if "daily_ret_pct" in df.columns:
        df["r"] = pd.to_numeric(df["daily_ret_pct"], errors="coerce") / 100.0
    elif "nav" in df.columns:
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["r"] = df["nav"].pct_change()
    else:
        return None

    df = df.dropna(subset=["r"])
    if len(df) < 20:
        return None

    n = len(df)
    mean_daily = df["r"].mean()
    std_daily = df["r"].std(ddof=1)

    if std_daily == 0 or np.isnan(std_daily):
        return None

    ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
    ann_vol = std_daily * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol

    # Per-period Sharpe (only longer windows; short ones are too noisy).
    psharpe = {key: _period_sharpe(df["r"], days, rf) for key, days in SHARPE_WINDOWS.items()}

    # Per-period max drawdown from the (ascending) NAV history. Only compute a
    # window when the fund actually has enough history to (mostly) cover it,
    # so e.g. a 1-month-old fund gets no "1-year" drawdown rather than a
    # misleadingly small one.
    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce").dropna() \
        if "nav" in nav_df.columns else None
    mdd = {}
    if nav_series is not None:
        n_nav = len(nav_series)
        for key, days in DRAWDOWN_WINDOWS.items():
            mdd[key] = _max_drawdown(nav_series.tail(days)) if n_nav >= days * 0.8 else None

    _save_sharpe(code, ann_return, ann_vol, sharpe, n, mdd, psharpe)
    return {
        "ann_return": ann_return, "volatility": ann_vol, "sharpe": sharpe,
        "data_points": n, **mdd, **psharpe,
    }


def batch_compute_sharpe(
    codes: list,
    rf: float = RISK_FREE_RATE,
    progress_callback: Optional[Callable] = None,
    workers: int = MAX_WORKERS,
) -> dict:
    """Compute Sharpe for a list of fund codes using cache + thread pool."""
    cached = _load_sharpe_cache(codes)
    to_fetch = [c for c in codes if c not in cached]
    results = dict(cached)

    total = len(codes)
    done_count = len(cached)

    if not to_fetch:
        if progress_callback:
            progress_callback(total, total)
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(compute_sharpe_for_fund, c, rf): c for c in to_fetch}
        for future in as_completed(futures):
            code = futures[future]
            try:
                res = future.result()
                if res:
                    results[code] = res
            except Exception as e:
                logger.debug("Sharpe compute error %s: %s", code, e)
            done_count += 1
            if progress_callback:
                progress_callback(done_count, total)

    return results
