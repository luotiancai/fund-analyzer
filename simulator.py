"""Paper-trading simulator (模拟盘) backed by the cached NAV store.

State model: the trade log (sim_trades) is the single source of truth. Cash,
holdings and P&L are always replayed from it plus NAV history, so 回退一天 is
simply "delete the departing day's trades and step the date back" — derived
state can never drift out of sync.

Trades execute at the fund's latest unit NAV on/before the simulated date
(same-day EOD fill, no fees) — a deliberate simplification for strategy
testing, not a broker emulation.
"""

import logging
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
    """Replay trades → ({code: [shares, cost]}, cash). Average-cost basis."""
    cash = INITIAL_CAPITAL
    pos: dict = {}
    for t in trades:
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
    return pos, cash


def holdings_and_cash(asof: str) -> Tuple[dict, float]:
    return _replay(_load_trades(asof))


def portfolio_value(asof: str) -> float:
    pos, cash = holdings_and_cash(asof)
    total = cash
    for code, (shares, cost) in pos.items():
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

def holdings_table(asof: str) -> pd.DataFrame:
    """Current positions valued as of `asof`: one row per held fund."""
    pos, _ = holdings_and_cash(asof)
    rows = []
    for code, (shares, cost) in pos.items():
        nav_date, nav = nav_asof(code, asof)
        value = shares * nav if nav is not None else cost
        rows.append({
            "code": code, "shares": shares, "cost": cost,
            "nav": nav, "nav_date": nav_date, "value": value,
            "pnl": value - cost,
            "pnl_pct": (value / cost - 1.0) * 100.0 if cost > 0 else 0.0,
        })
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
