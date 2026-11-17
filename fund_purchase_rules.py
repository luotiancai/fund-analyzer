"""Audited subscription restrictions for the historical comparison pool.

These are dated facts, not a permanent blacklist of North Exchange funds.
Only reviewed buy dates may be used; a missing review must not silently turn
into an unrestricted pool. Unlisted funds are not certified unrestricted.
"""
import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

RULES_FILE = Path(__file__).with_name("fund_purchase_limits.json")


@lru_cache(maxsize=1)
def purchase_rules():
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def restricted_codes_on(asof):
    date = str(asof)[:10]
    dt.date.fromisoformat(date)
    snapshot = purchase_rules()["by_date"].get(date)
    if snapshot is None:
        raise ValueError(f"{date} 的历史限购尚未核实，请先补充公告再计算可买池。")
    return frozenset(snapshot)
