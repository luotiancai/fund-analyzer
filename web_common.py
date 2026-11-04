"""Shared paths for the persistent VPS web service (no pandas imports)."""
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

CST = ZoneInfo("Asia/Shanghai")
ROOT = Path(os.environ.get("FUND_WEB_ROOT", "/var/lib/fund-web"))
CACHE = Path(os.environ.get("FUND_WEB_CACHE", str(ROOT / "cache")))
FILES = {
    "rank": "fund_rank.db", "nav": "fund_nav.db", "scale": "fund_scale.db",
    "market": "market.db", "qvix": "qvix.db",
}


def today():
    return dt.datetime.now(CST).date()


def snapshot():
    p = (ROOT / "current").resolve(strict=True)
    if p.parent != (ROOT / "snapshots").resolve():
        raise RuntimeError("无效的数据快照路径")
    manifest = json.loads((p / "manifest.json").read_text())
    return p, manifest


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
