"""Paper-trading simulator (模拟盘) backed by the cached NAV store.

State model: the trade log (sim_trades) is the single source of truth. Cash,
holdings and P&L are always replayed from it plus NAV history, so 回退一天 is
simply "delete the departing day's trades and step the date back" — derived
state can never drift out of sync.

Trades execute at the fund's latest unit NAV on/before the simulated date
(same-day EOD fill, no fees) — a deliberate simplification for strategy
testing, not a broker emulation.
"""

import json
import logging
import time
from typing import Optional, Tuple

import pandas as pd

import fetcher

logger = logging.getLogger(__name__)

SIM_START = "2026-01-01"
INITIAL_CAPITAL = 1_000_000.0


def init_sim_db():
    conn = fetcher._conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sim_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS sim_trades (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            date   TEXT NOT NULL,
            code   TEXT NOT NULL,
            action TEXT NOT NULL,      -- 'buy' | 'sell'
            shares REAL NOT NULL,
            nav    REAL NOT NULL,      -- execution unit NAV
            amount REAL NOT NULL       -- buy: cash spent; sell: cash received
        );
        CREATE TABLE IF NOT EXISTS sim_archives (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            saved_at     REAL NOT NULL,
            current_date TEXT,
            trades       TEXT NOT NULL  -- JSON dump of sim_trades rows
        );
    """)
    # The trading calendar (MIN/MAX/DISTINCT over date) needs this; one-time build.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_date ON fund_nav_daily(date)")
    conn.commit()
    conn.close()


# ── Trading calendar ─────────────────────────────────────────────────────────
# A "trading day" is any date with at least one stored NAV row.

def first_trading_day() -> Optional[str]:
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT MIN(date) AS d FROM fund_nav_daily WHERE date >= ?",
        (SIM_START,)).fetchone()
    conn.close()
    return row["d"] if row else None


def latest_trading_day() -> Optional[str]:
    conn = fetcher._conn()
    row = conn.execute("SELECT MAX(date) AS d FROM fund_nav_daily").fetchone()
    conn.close()
    return row["d"] if row else None


def trading_days(start: str, end: str) -> list:
    conn = fetcher._conn()
    rows = conn.execute(
        "SELECT DISTINCT date FROM fund_nav_daily WHERE date >= ? AND date <= ? "
        "ORDER BY date", (start, end)).fetchall()
    conn.close()
    return [r["date"] for r in rows]


# ── Current simulated date ───────────────────────────────────────────────────

def _set_current_date(d: str):
    conn = fetcher._conn()
    conn.execute(
        "INSERT OR REPLACE INTO sim_meta (key, value) VALUES ('current_date', ?)",
        (d,))
    conn.commit()
    conn.close()


def get_current_date() -> Optional[str]:
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT value FROM sim_meta WHERE key='current_date'").fetchone()
    conn.close()
    if row:
        return row["value"]
    d = first_trading_day()
    if d:
        _set_current_date(d)
    return d


def advance_day() -> Tuple[str, bool]:
    """Step to the next trading day. Returns (date, moved)."""
    cur = get_current_date()
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT MIN(date) AS d FROM fund_nav_daily WHERE date > ?",
        (cur,)).fetchone()
    conn.close()
    if row and row["d"]:
        _set_current_date(row["d"])
        return row["d"], True
    return cur, False


def rollback_day() -> Tuple[str, bool]:
    """Step back one trading day, discarding the departing day's trades.

    Returns (date, moved). Never goes before the first trading day.
    """
    cur = get_current_date()
    first = first_trading_day()
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT MAX(date) AS d FROM fund_nav_daily WHERE date < ? AND date >= ?",
        (cur, first)).fetchone()
    prev = row["d"] if row else None
    if not prev:
        conn.close()
        return cur, False
    conn.execute("DELETE FROM sim_trades WHERE date >= ?", (cur,))
    conn.execute(
        "INSERT OR REPLACE INTO sim_meta (key, value) VALUES ('current_date', ?)",
        (prev,))
    conn.commit()
    conn.close()
    return prev, True


def reset():
    """Wipe all trades and restart from the first trading day."""
    conn = fetcher._conn()
    conn.execute("DELETE FROM sim_trades")
    conn.execute("DELETE FROM sim_meta WHERE key='current_date'")
    conn.commit()
    conn.close()


# ── Archives (saved simulator runs) ──────────────────────────────────────────
# A snapshot of the live state (trade log + simulated date). Loading one
# replaces the live state wholesale, so the run continues exactly where it
# was archived; derived state is still replayed from the restored trades.

def save_archive(name: str) -> Optional[str]:
    """Snapshot current trades + date under `name`. Error string or None."""
    d = get_current_date()
    conn = fetcher._conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT date, code, action, shares, nav, amount "
        "FROM sim_trades ORDER BY id")]
    if not rows:
        conn.close()
        return "当前没有任何交易，无需存档"
    conn.execute(
        "INSERT INTO sim_archives (name, saved_at, current_date, trades) "
        "VALUES (?, ?, ?, ?)",
        (name.strip() or f"存档 {time.strftime('%m-%d %H:%M')}",
         time.time(), d, json.dumps(rows, ensure_ascii=False)))
    conn.commit()
    conn.close()
    return None


def list_archives() -> pd.DataFrame:
    """All archives, newest first: id, name, saved_at, current_date, n_trades."""
    conn = fetcher._conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, name, saved_at, [current_date], trades FROM sim_archives "
        "ORDER BY id DESC")]
    conn.close()
    for r in rows:
        r["n_trades"] = len(json.loads(r.pop("trades")))
    return pd.DataFrame(rows, columns=["id", "name", "saved_at",
                                       "current_date", "n_trades"])


def load_archive(archive_id: int) -> Optional[str]:
    """Replace the live simulator state with an archive's. Error or None."""
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT [current_date], trades FROM sim_archives WHERE id=?",
        (archive_id,)).fetchone()
    if not row:
        conn.close()
        return "存档不存在"
    trades = json.loads(row["trades"])
    conn.execute("DELETE FROM sim_trades")
    conn.executemany(
        "INSERT INTO sim_trades (date, code, action, shares, nav, amount) "
        "VALUES (:date, :code, :action, :shares, :nav, :amount)", trades)
    conn.execute(
        "INSERT OR REPLACE INTO sim_meta (key, value) VALUES ('current_date', ?)",
        (row["current_date"],))
    conn.commit()
    conn.close()
    return None


def copy_archive(archive_id: int) -> Optional[str]:
    """Duplicate an archive (name + ' 副本') so a strategy can be branched
    and modified without touching the original. Error string or None."""
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT name, [current_date], trades FROM sim_archives WHERE id=?",
        (archive_id,)).fetchone()
    if not row:
        conn.close()
        return "存档不存在"
    conn.execute(
        "INSERT INTO sim_archives (name, saved_at, current_date, trades) "
        "VALUES (?, ?, ?, ?)",
        (f"{row['name']} 副本", time.time(), row["current_date"], row["trades"]))
    conn.commit()
    conn.close()
    return None


def delete_archive(archive_id: int):
    conn = fetcher._conn()
    conn.execute("DELETE FROM sim_archives WHERE id=?", (archive_id,))
    conn.commit()
    conn.close()


# ── Portfolio state (replayed from the trade log) ────────────────────────────

def nav_asof(code: str, date: str) -> Tuple[Optional[str], Optional[float]]:
    """Latest (nav_date, nav) on/before `date` — never looks into the future."""
    conn = fetcher._conn()
    row = conn.execute(
        "SELECT date, nav FROM fund_nav_daily "
        "WHERE code = ? AND date <= ? AND nav IS NOT NULL "
        "ORDER BY date DESC LIMIT 1", (code, date)).fetchone()
    conn.close()
    return (row["date"], row["nav"]) if row else (None, None)


def _load_trades(upto: str) -> list:
    conn = fetcher._conn()
    rows = conn.execute(
        "SELECT id, date, code, action, shares, nav, amount FROM sim_trades "
        "WHERE date <= ? ORDER BY id", (upto,)).fetchall()
    conn.close()
    return rows


def _replay(trades) -> Tuple[dict, float]:
    """Replay trades → ({code: [shares, cost, open_date, open_nav]}, cash).

    Average-cost basis. open_date/open_nav are from the trade that opened the
    current position (a full exit clears them; re-buying starts a new lot),
    so charts can show 累计收益率 since the actual entry point.
    """
    cash = INITIAL_CAPITAL
    pos: dict = {}
    for t in trades:
        if t["action"] == "buy":
            cash -= t["amount"]
            p = pos.setdefault(t["code"], [0.0, 0.0, t["date"], t["nav"]])
            p[0] += t["shares"]
            p[1] += t["amount"]
        else:
            cash += t["amount"]
            p = pos.get(t["code"])
            if p:
                frac = min(t["shares"] / p[0], 1.0) if p[0] > 0 else 1.0
                p[1] *= (1.0 - frac)
                p[0] -= t["shares"]
                if p[0] <= 1e-9:
                    pos.pop(t["code"])
    return pos, cash


def holdings_and_cash(asof: str) -> Tuple[dict, float]:
    return _replay(_load_trades(asof))


def portfolio_value(asof: str) -> float:
    pos, cash = holdings_and_cash(asof)
    total = cash
    for code, p in pos.items():
        shares, cost = p[0], p[1]
        _, nav = nav_asof(code, asof)
        total += shares * nav if nav is not None else cost
    return total


# ── Trading ──────────────────────────────────────────────────────────────────
# Both return an error string, or None on success.

def buy(code: str, amount: float) -> Optional[str]:
    d = get_current_date()
    if amount <= 0:
        return "买入金额需大于 0"
    _, cash = holdings_and_cash(d)
    if amount > cash + 1e-6:
        return f"现金不足（可用 ¥{cash:,.2f}）"
    nav_date, nav = nav_asof(code, d)
    if nav is None:
        return "该基金在当前模拟日期之前没有净值数据"
    shares = amount / nav
    conn = fetcher._conn()
    conn.execute(
        "INSERT INTO sim_trades (date, code, action, shares, nav, amount) "
        "VALUES (?, ?, 'buy', ?, ?, ?)", (d, code, shares, nav, amount))
    conn.commit()
    conn.close()
    return None


def sell(code: str, shares: Optional[float] = None) -> Optional[str]:
    """Sell `shares` of `code` (None = everything held)."""
    d = get_current_date()
    pos, _ = holdings_and_cash(d)
    held = pos.get(code)
    if not held or held[0] <= 0:
        return "当前未持有该基金"
    if shares is None:
        shares = held[0]
    if shares <= 0:
        return "卖出份额需大于 0"
    if shares > held[0] + 1e-6:
        return f"持有份额不足（持有 {held[0]:,.2f} 份）"
    shares = min(shares, held[0])
    nav_date, nav = nav_asof(code, d)
    if nav is None:
        return "该基金在当前模拟日期之前没有净值数据"
    conn = fetcher._conn()
    conn.execute(
        "INSERT INTO sim_trades (date, code, action, shares, nav, amount) "
        "VALUES (?, ?, 'sell', ?, ?, ?)", (d, code, shares, nav, shares * nav))
    conn.commit()
    conn.close()
    return None


# ── Reporting ────────────────────────────────────────────────────────────────

def nav_series(code: str, start: str, end: str) -> pd.DataFrame:
    """NAV + daily-return history for one fund within [start, end] (ascending).

    Columns: date, nav, daily_ret_pct. `end` should be the current simulated
    date so charts never show the future to the strategy being tested.
    """
    conn = fetcher._conn()
    df = pd.read_sql_query(
        "SELECT date, nav, daily_ret_pct FROM fund_nav_daily "
        "WHERE code = ? AND date >= ? AND date <= ? AND nav IS NOT NULL "
        "ORDER BY date", conn, params=(code, start, end))
    conn.close()
    return df



def holdings_table(asof: str) -> pd.DataFrame:
    """Current positions valued as of `asof`: one row per held fund."""
    pos, _ = holdings_and_cash(asof)
    conn = fetcher._conn()
    rows = []
    for code, p in pos.items():
        shares, cost, open_date, open_nav = p[0], p[1], p[2], p[3]
        r = conn.execute(
            "SELECT date, nav, daily_ret_pct FROM fund_nav_daily "
            "WHERE code = ? AND date <= ? AND nav IS NOT NULL "
            "ORDER BY date DESC LIMIT 1", (code, asof)).fetchone()
        nav_date, nav, day_ret = (
            (r["date"], r["nav"], r["daily_ret_pct"]) if r else (None, None, None))
        value = shares * nav if nav is not None else cost
        rows.append({
            "code": code, "shares": shares, "cost": cost,
            "open_date": open_date, "open_nav": open_nav,
            "nav": nav, "nav_date": nav_date, "day_ret": day_ret,
            "value": value,
            "pnl": value - cost,
            "pnl_pct": (value / cost - 1.0) * 100.0 if cost > 0 else 0.0,
        })
    conn.close()
    df = pd.DataFrame(rows)
    return df.sort_values("value", ascending=False) if not df.empty else df


def trades_table(upto: str) -> pd.DataFrame:
    rows = _load_trades(upto)
    return pd.DataFrame([dict(r) for r in rows])


def equity_curve() -> pd.DataFrame:
    """Portfolio value on every trading day from SIM_START to the current
    simulated date. Columns: date, value."""
    cur = get_current_date()
    first = first_trading_day()
    if not cur or not first:
        return pd.DataFrame(columns=["date", "value"])
    days = trading_days(first, cur)
    trades = _load_trades(cur)

    # One NAV series per traded fund, forward-filled onto the sim calendar so
    # per-day valuation never looks past the day (or misses a fund holiday).
    codes = {t["code"] for t in trades}
    conn = fetcher._conn()
    nav_ff = {}
    for c in codes:
        s = pd.read_sql_query(
            "SELECT date, nav FROM fund_nav_daily "
            "WHERE code = ? AND date <= ? AND nav IS NOT NULL ORDER BY date",
            conn, params=(c, cur)).set_index("date")["nav"]
        idx = sorted(set(s.index) | set(days))
        nav_ff[c] = s.reindex(idx).ffill()
    conn.close()

    cash = INITIAL_CAPITAL
    pos: dict = {}
    ti = 0
    out = []
    for d in days:
        while ti < len(trades) and trades[ti]["date"] <= d:
            t = trades[ti]
            if t["action"] == "buy":
                cash -= t["amount"]
                p = pos.setdefault(t["code"], [0.0, 0.0])
                p[0] += t["shares"]
                p[1] += t["amount"]
            else:
                cash += t["amount"]
                p = pos.get(t["code"])
                if p:
                    frac = min(t["shares"] / p[0], 1.0) if p[0] > 0 else 1.0
                    p[1] *= (1.0 - frac)
                    p[0] -= t["shares"]
                    if p[0] <= 1e-9:
                        pos.pop(t["code"])
            ti += 1
        total = cash
        for c, (shares, cost) in pos.items():
            nav = nav_ff[c].get(d)
            total += shares * nav if nav is not None and pd.notna(nav) else cost
        out.append({"date": d, "value": total})
    return pd.DataFrame(out)
