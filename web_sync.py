#!/usr/bin/env python3
"""Publish one validated, immutable generation for the web consumer.

GitHub Actions remains the writer of rank/nav/scale/market; the existing QVIX
job remains its sole writer. This command downloads releases and never uploads.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import urllib.request
import uuid

from web_common import FILES, ROOT, snapshot, today, write_json

REPO = "luotiancai/fund-analyzer"
log = logging.getLogger("web-sync")


def assets():
    raw = subprocess.check_output(
        ["gh", "release", "view", "data", "--repo", REPO, "--json", "assets"], timeout=45)
    wanted = {filename + ".gz" for filename in FILES.values()}
    result = {r["name"]: {"updated": r["updatedAt"], "size": r["size"]}
              for r in json.loads(raw)["assets"] if r["name"] in wanted}
    if set(result) != wanted:
        raise RuntimeError("Release 缺少必要数据库，保留当前快照")
    return result


def connect(path):
    c = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    return c


def build_meta(stage):
    """Scan NAV once at synchronization time, never on a screening request."""
    dest = stage / "web_meta.db"
    if dest.exists():
        return
    tmp = stage / "web_meta.db.tmp"
    with contextlib.closing(sqlite3.connect(tmp)) as out:
        out.execute("CREATE TABLE nav_dates (code TEXT PRIMARY KEY,first_date TEXT,last_date TEXT) WITHOUT ROWID")
        with contextlib.closing(connect(stage / FILES["nav"])) as source:
            rows = source.execute("SELECT code,MIN(date),MAX(date) FROM fund_nav_daily GROUP BY code")
            out.executemany("INSERT INTO nav_dates VALUES (?,?,?)", rows)
        out.commit()
    tmp.chmod(0o444)
    os.replace(tmp, dest)


def validate(stage):
    with contextlib.closing(connect(stage / FILES["rank"])) as c:
        row = c.execute("SELECT MAX(last_date),COUNT(*) FROM fund_nav_meta").fetchone()
        nav_date, count = row
        if not nav_date or not count:
            raise RuntimeError("净值元数据为空")
        version = c.execute("SELECT value FROM app_meta WHERE key='metrics_nav_date'").fetchone()
        if not version or int(version[0]) != int(nav_date.replace("-", "")):
            raise RuntimeError("净值与指标日期不一致")
        data = json.loads(c.execute("SELECT data FROM fund_list WHERE id=1").fetchone()[0])
        probes = c.execute("SELECT code,last_date FROM fund_nav_meta ORDER BY code").fetchall()[::300]
    with contextlib.closing(connect(stage / FILES["nav"])) as c:
        for code, expected in probes:
            actual = c.execute("SELECT MAX(date) FROM fund_nav_daily WHERE code=?", (code,)).fetchone()[0]
            if actual != expected:
                raise RuntimeError(f"榜单与净值快照不一致: {code}，稍后重试")
    with contextlib.closing(connect(stage / FILES["market"])) as c:
        raw = c.execute("SELECT data FROM index_daily_cache WHERE key='sse'").fetchone()
        sse = json.loads(raw[0]) if raw else []
        # index_daily_cache uses pandas orient=split, not records.
        if isinstance(sse, dict) and "columns" in sse:
            idx = sse["columns"].index("date")
            sse_date = max(str(r[idx])[:10] for r in sse["data"])
        else:
            sse_date = max(str(r["date"])[:10] for r in sse)
    with contextlib.closing(connect(stage / FILES["qvix"])) as c:
        qdate = c.execute("SELECT MAX(date) FROM qvix_self_history").fetchone()[0]
    return dict(nav_date=nav_date, sse_date=sse_date, qvix_date=qdate,
                nav_funds=count, funds=len(data), synced_at=time.time())


def sync():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "snapshots").mkdir(exist_ok=True)
    with (ROOT / ".sync.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("已有同步任务，跳过")
            return False
        remote = assets()
        generation = hashlib.sha256(json.dumps(remote, sort_keys=True).encode()).hexdigest()[:20]
        previous, old = None, {}
        try:
            previous, old = snapshot()
        except FileNotFoundError:
            pass
        if old.get("version") == generation:
            build_meta(previous)
            log.info("远端未变，当前 %s", generation)
            return False
        stage = ROOT / "snapshots" / (".staging-" + uuid.uuid4().hex)
        dest = ROOT / "snapshots" / generation
        stage.mkdir(mode=0o755)
        try:
            for filename in FILES.values():
                asset = filename + ".gz"
                target = stage / filename
                if previous and old.get("assets", {}).get(asset) == remote[asset]:
                    os.link(previous / filename, target)
                    continue
                log.info("下载 %s (%.1f MB)", asset, remote[asset]["size"] / 1048576)
                archive = stage / asset
                subprocess.run(["gh", "release", "download", "data", "--repo", REPO,
                                "--pattern", asset, "--output", str(archive)],
                               check=True, timeout=600)
                with gzip.open(archive, "rb") as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                archive.unlink()
                with contextlib.closing(connect(target)) as c:
                    result = c.execute("PRAGMA quick_check").fetchone()[0]
                    if result != "ok":
                        raise RuntimeError(f"{filename} 完整性检查失败: {result}")
                target.chmod(0o444)
            if assets() != remote:
                raise RuntimeError("下载期间 Release 已更新，保留原快照，下一次重试")
            if (previous and (previous / "web_meta.db").exists()
                    and old["assets"]["fund_nav.db.gz"] == remote["fund_nav.db.gz"]):
                os.link(previous / "web_meta.db", stage / "web_meta.db")
            else:
                build_meta(stage)
            summary = validate(stage)
            with contextlib.closing(connect(stage / "web_meta.db")) as meta:
                actual = meta.execute("SELECT MAX(last_date) FROM nav_dates").fetchone()[0]
                if actual != summary["nav_date"]:
                    raise RuntimeError("净值实存日期与榜单日期不一致，保留原快照")
            manifest = dict(version=generation, assets=remote, summary=summary)
            write_json(stage / "manifest.json", manifest)
            if dest.exists():
                shutil.rmtree(stage)
            else:
                os.replace(stage, dest)
            link = ROOT / (".current-" + uuid.uuid4().hex)
            link.symlink_to(Path("snapshots") / generation)
            os.replace(link, ROOT / "current")
            log.info("切换至 %s，净值 %s", generation, summary["nav_date"])
            # Keep recent generations for active jobs and rollback. All cleanup
            # is confined to directories created by this consumer.
            generations = sorted((p for p in (ROOT / "snapshots").iterdir()
                                  if p.is_dir() and not p.name.startswith(".")),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            for p in generations[4:]:
                if p != dest and time.time() - p.stat().st_mtime > 86400:
                    shutil.rmtree(p)
            return True
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def warm():
    request = urllib.request.Request(
        "http://127.0.0.1:8010/api/v1/jobs",
        data=json.dumps({"operation": "prepare", "asof": today().isoformat()}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            log.info("今日候选预热: %s", json.load(response)["status"])
    except Exception as exc:
        log.warning("预热暂未提交: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true")
    args = ap.parse_args()
    sync()
    if args.warm:
        warm()
