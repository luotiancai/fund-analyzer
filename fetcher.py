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
RISK_FREE_RATE = 0.0113  # fallback 1-year China gov bond yield (see get_risk_free_rate)
RF_TTL = 30 * 86400      # auto-refresh the risk-free rate ~monthly


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
        CREATE TABLE IF NOT EXISTS app_meta (
            key      TEXT PRIMARY KEY,
            value    REAL,
            saved_at REAL NOT NULL
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
    conn.execute("DELETE FROM fund_nav")
    conn.execute("DELETE FROM fund_sharpe")
    conn.commit()
    conn.close()


# Look-back windows in CALENDAR days, matching how EastMoney defines 近1月/3月/
# 6月/1年 (date-to-date from the latest NAV date), so the computed drawdown and
# Sharpe cover the same period as the 近X 收益率 columns shown alongside them.
DRAWDOWN_DAYS = {"mdd_1m": 30, "mdd_3m": 91, "mdd_6m": 182, "mdd_1y": 365}

# Sharpe is only computed for longer windows (short windows are too noisy).
SHARPE_DAYS = {"sharpe_6m": 182, "sharpe_1y": 365}


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
    """Rebuild a NAV DataFrame from stored JSON, restoring the `date` column.

    df.to_json serializes datetimes as epoch-millisecond ints, which
    pd.to_datetime would otherwise misread as nanoseconds (everything → 1970).
    Newer rows are stored ISO-formatted; handle both shapes.
    """
    df = pd.DataFrame(json.loads(blob))
    if not df.empty and "date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="ms")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _load_nav_cache(code: str) -> Optional[pd.DataFrame]:
    conn = _conn()
    row = conn.execute(
        "SELECT data, saved_at FROM fund_nav WHERE code = ?", (code,)
    ).fetchone()
    conn.close()
    if row and (time.time() - row["saved_at"]) < NAV_TTL:
        return _nav_from_json(row["data"])
    return None


def _save_nav_cache(code: str, df: pd.DataFrame):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO fund_nav (code, data, saved_at) VALUES (?, ?, ?)",
        (code, df.to_json(orient="records", force_ascii=False, date_format="iso"),
         time.time()),
    )
    conn.commit()
    conn.close()


def fetch_nav(code: str) -> Optional[pd.DataFrame]:
    """Return ~1-year NAV history for a single fund.

    Pulls both 单位净值走势 (unit NAV + daily growth) and 累计净值走势 (accumulated
    NAV). Drawdown is later computed on the accumulated NAV so that dividend
    distributions — which drop the *unit* NAV on the ex-date — don't register as
    spurious drawdowns.
    Columns: date, nav, daily_ret_pct, acc_nav
    """
    cached = _load_nav_cache(code)
    if cached is not None and "acc_nav" in cached.columns:
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

        # Accumulated (dividend-reinvested) NAV for drawdown; fall back to unit
        # NAV if the accumulated series can't be fetched or has gaps.
        try:
            acc = ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
            acc = acc.rename(columns={"净值日期": "date", "累计净值": "acc_nav"})
            acc["date"] = pd.to_datetime(acc["date"])
            df = df.merge(acc[["date", "acc_nav"]], on="date", how="left")
        except Exception:
            df["acc_nav"] = df["nav"]
        df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce").fillna(
            pd.to_numeric(df["nav"], errors="coerce")
        )

        df = df.sort_values("date")

        # Filter to last 365 + buffer days; the buffer guarantees a NAV old
        # enough to anchor the trailing-1-year window even across holiday gaps.
        cutoff = datetime.now() - timedelta(days=400)
        df = df[df["date"] >= cutoff].copy().reset_index(drop=True)

        if len(df) < 20:
            return None

        _save_nav_cache(code, df)
        return df

    except Exception as e:
        logger.debug("NAV fetch failed for %s: %s", code, e)
        return None


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

    return {
        "ann_return": ann_return, "volatility": ann_vol, "sharpe": sharpe,
        "data_points": n, **mdd, **psharpe,
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


# ── Daily-batch pipeline ──────────────────────────────────────────────────────
# Used by update_daily.py: backfill once, then each day append the latest NAV
# point (from the bulk fund-list call) and recompute Sharpe/drawdown for all.

NAV_HISTORY_DAYS = 400  # rolling window kept per fund (covers the 1y look-back)


def list_nav_codes() -> set:
    """Codes that already have a stored NAV history."""
    conn = _conn()
    rows = conn.execute("SELECT code FROM fund_nav").fetchall()
    conn.close()
    return {r["code"] for r in rows}


def append_latest_nav(list_df: pd.DataFrame) -> dict:
    """Append today's NAV point to every stored fund from the bulk fund list.

    `list_df` is fetch_fund_list() output (columns: code, nav_date, nav, acc_nav,
    daily_ret). Appends one row per fund when the list's date is newer than the
    stored last date, trims to NAV_HISTORY_DAYS, and writes everything in one
    transaction. Funds whose gap to the latest date exceeds a week are returned
    in `gapped` for the caller to fully re-backfill (the bulk call only carries
    the single latest point, so it can't fill multi-day gaps).
    """
    latest = list_df.dropna(subset=["code"]).drop_duplicates("code").set_index("code")
    cutoff = datetime.now() - timedelta(days=NAV_HISTORY_DAYS)
    now = time.time()

    conn = _conn()
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav").fetchall()]
    updated, skipped, gapped = 0, 0, []
    for code in codes:
        if code not in latest.index:
            skipped += 1
            continue
        row = latest.loc[code]
        new_date = pd.to_datetime(row["nav_date"], errors="coerce")
        nav = pd.to_numeric(row["nav"], errors="coerce")
        if pd.isna(new_date) or pd.isna(nav):
            skipped += 1
            continue

        cur = conn.execute("SELECT data FROM fund_nav WHERE code=?", (code,)).fetchone()
        hist = _nav_from_json(cur["data"])
        if hist.empty:
            gapped.append(code)
            continue
        last_date = hist["date"].max()

        if new_date <= last_date:
            skipped += 1                         # already have this day
            continue
        if (new_date - last_date).days > 7:
            gapped.append(code)                  # missed too many days; refetch
            continue

        acc = pd.to_numeric(row.get("acc_nav"), errors="coerce")
        new_row = {
            "date": new_date,
            "nav": float(nav),
            "daily_ret_pct": pd.to_numeric(row.get("daily_ret"), errors="coerce"),
            "acc_nav": float(acc) if not pd.isna(acc) else float(nav),
        }
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist = hist[hist["date"] >= cutoff].sort_values("date").reset_index(drop=True)
        conn.execute(
            "UPDATE fund_nav SET data=?, saved_at=? WHERE code=?",
            (hist.to_json(orient="records", force_ascii=False), now, code),
        )
        updated += 1

    conn.commit()
    conn.close()
    return {"updated": updated, "skipped": skipped, "gapped": gapped}


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
    codes = [r["code"] for r in conn.execute("SELECT code FROM fund_nav").fetchall()]
    total, done, saved = len(codes), 0, 0
    for code in codes:
        row = conn.execute("SELECT data FROM fund_nav WHERE code=?", (code,)).fetchone()
        done += 1
        try:
            nav_df = _nav_from_json(row["data"]) if row else None
            m = _metrics_from_nav(nav_df, rf) if nav_df is not None and not nav_df.empty else None
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
    ① refresh fund list → ② backfill funds missing history → ③ append today's
    NAV point → ④ recompute Sharpe/drawdown for all.

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
        todo = [c for c in all_codes if c not in list_nav_codes()]
        backfilled = _backfill_codes(
            todo, workers, lambda d, t: _p("回填缺失历史", d, t)
        )

    _p("追加当日净值", 0, 1)
    res = append_latest_nav(list_df)
    _p("追加当日净值", 1, 1)
    if res["gapped"]:
        _backfill_codes(res["gapped"], workers, lambda d, t: _p("补净值缺口", d, t))

    saved = recompute_all(rf=rf, progress_callback=lambda d, t: _p("重算指标", d, t))

    return {
        "funds": len(all_codes),
        "backfilled": backfilled,
        "appended": res["updated"],
        "recomputed": saved,
        "rf": rf,
    }
