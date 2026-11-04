"""Disk caches shared by workers and the HTTP reader; no analysis imports."""
import contextlib
import fcntl
import json
from pathlib import Path
import sqlite3

from web_common import CACHE, FILES

LOOKUP_TABLES = {
    "fund_holdings": ("code,year,data,saved_at", "code,year", "data,saved_at"),
    "etf_target_map": ("code,target_code,target_name,saved_at", "code", "target_code,target_name,saved_at"),
    "fund_index_code": ("code,index_code,saved_at", "code", "index_code,saved_at"),
}


def readonly(path):
    return sqlite3.connect(Path(path).as_uri() + "?mode=ro&immutable=1", uri=True)


def lookup_cache(snapshot):
    """Keep upstream caches across code/data versions; seed prior overlays once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "lookups.db"
    with (CACHE / ".lookup-init.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with contextlib.closing(sqlite3.connect(dest, timeout=15)) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS fund_holdings (
                    code TEXT, year TEXT, data TEXT NOT NULL, saved_at REAL NOT NULL,
                    PRIMARY KEY(code,year)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS etf_target_map (
                    code TEXT PRIMARY KEY,target_code TEXT NOT NULL,
                    target_name TEXT NOT NULL,saved_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS fund_index_code (
                    code TEXT PRIMARY KEY,index_code TEXT NOT NULL,saved_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS lookup_meta (key TEXT PRIMARY KEY,value TEXT);
            """)
            if c.execute("SELECT 1 FROM lookup_meta WHERE key='seeded'").fetchone():
                return dest
            sources = [snapshot / FILES["rank"], snapshot / FILES["scale"]]
            for folder in (CACHE / "runtime").glob("*"):
                sources.extend(folder / FILES[name] for name in ("rank", "scale"))
            for source in sources:
                if not source.exists():
                    continue
                with contextlib.closing(readonly(source)) as old:
                    tables = {r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for table, (cols, key, updated) in LOOKUP_TABLES.items():
                        if table not in tables:
                            continue
                        rows = old.execute(f"SELECT {cols} FROM {table}")
                        updates = ",".join(f"{col}=excluded.{col}" for col in updated.split(","))
                        marks = ",".join("?" for _ in cols.split(","))
                        c.executemany(f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
                                      f"ON CONFLICT({key}) DO UPDATE SET {updates} "
                                      f"WHERE excluded.saved_at>{table}.saved_at", rows)
            c.execute("INSERT INTO lookup_meta VALUES ('seeded','1')")
            c.commit()
    return dest


def analysis_folder(snapshot, code_version):
    return CACHE / "runtime" / (snapshot.name + "-" + code_version)


def prepared_result(snapshot, code_version, operation, args):
    if operation not in ("pick", "probe"):
        return None
    folder = analysis_folder(snapshot, code_version)
    asof = args["asof"]
    if operation == "pick":
        path = folder / ("pick-" + asof + ".json")
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        result["chosen_name"] = result.get("names", {}).get(result.get("chosen"))
        return {k: v for k, v in result.items() if k not in ("names", "cand", "ranks")}
    path = folder / ("probe-" + asof + ".db")
    if not path.exists():
        return None
    with contextlib.closing(readonly(path)) as c:
        n_drop = c.execute("SELECT value FROM meta WHERE key='n_drop'").fetchone()[0]
        rows = []
        for code in args["codes"]:
            found = c.execute("SELECT data FROM probes WHERE code=?", (code,)).fetchone()
            rows.append(json.loads(found[0]) if found else dict(
                code=code, name=code, ret3m=None, beta=None, aum=None,
                why="榜单里没有这个代码(不是C类/已清盘/代码敲错)", role="",
                win_start=None, win_end=None, rank=None))
    return dict(rows=rows, asof=asof, n_drop=int(n_drop))
