"""Manual QVIX lookup, isolated from analysis jobs and authoritative writers."""
import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys
import time

import qvix_core
from web_common import CACHE, FILES, write_json

log = logging.getLogger("fund-web.qvix")
TIMEOUT = 25
CACHE_VERSION = hashlib.sha256(
    Path(__file__).read_bytes() + Path(qvix_core.__file__).read_bytes()).hexdigest()[:16]


def connection(path, name):
    return sqlite3.connect((path / FILES[name]).as_uri() + "?mode=ro&immutable=1", uri=True)


def quote_date():
    # The actual 50ETF quote date also detects weekday holidays, without
    # installing akshare/a JavaScript runtime in the lightweight web service.
    text = qvix_core._get(qvix_core._HQ + "sh510050")
    match = re.search(r'hq_str_sh510050="([^"]+)"', text)
    fields = match.group(1).split(",") if match else []
    if len(fields) < 32:
        raise ValueError("未取得有效的 50ETF 行情日期，请稍后重试。")
    try:
        return dt.date.fromisoformat(fields[30])
    except ValueError:
        raise ValueError("50ETF 行情日期无效，请稍后重试。") from None


def calculate(path, now=None):
    now = now or dt.datetime.now(qvix_core.CST)
    phase, as_of = qvix_core.market_phase(now, now.weekday() < 5)
    quoted = None
    if phase != "prev":
        quoted = quote_date()
        if quoted > now.date():
            raise ValueError("上游行情日期异常，请稍后重试。")
        phase, as_of = qvix_core.market_phase(now, quoted == now.date())
    if phase == "auction":
        raise ValueError("收盘集合竞价中（14:57–15:00），请在 15:00 后查询。")

    with contextlib.closing(connection(path, "qvix")) as c:
        row = c.execute("SELECT date,qvix,threshold FROM qvix_self_history "
                        "WHERE date < ? ORDER BY date DESC LIMIT 1",
                        (now.date().isoformat(),)).fetchone()
    if row is None:
        raise ValueError("没有可用的 QVIX 历史，请等待数据同步。")
    day, historical, threshold = row
    if phase == "prev":
        if quoted and day != quoted.isoformat():
            raise ValueError("行情日期与 QVIX 历史不一致，请等待数据同步后重试。")
        if historical is None:
            raise ValueError("最近交易日的 QVIX 缺失，请等待数据同步。")
        value, date, clock = historical, day, "15:00:00"
    else:
        with contextlib.closing(connection(path, "rank")) as c:
            rf = c.execute("SELECT value FROM app_meta WHERE key='rf_rate'").fetchone()
        fallback_rate = rf[0] if rf and rf[0] is not None else 0.0113
        result = qvix_core.compute_qvix(as_of=as_of, fallback_rate=fallback_rate)
        if result is None or result[0] is None:
            raise ValueError("实时 QVIX 计算失败：期权报价不完整或上游连接失败，请重试。")
        value, clock = result
        date = now.date().isoformat()
    return dict(qvix=value, date=date, time=clock, phase=phase, threshold=threshold,
                threshold_date=day, gap=round(value - threshold, 2) if threshold is not None else None,
                hit=value >= threshold if threshold is not None else None,
                checked_at=time.time())


class LiveQvix:
    """Only button clicks start work; concurrent clicks share one small process."""
    def __init__(self, cache_dir=None):
        self.pending = None
        self.pending_key = None
        self.cache_file = (Path(cache_dir) if cache_dir is not None else CACHE / "qvix") / "frozen.json"

    @staticmethod
    def cacheable(key, result):
        return (key[-1] in ("noon", "close") and result.get("phase") == key[-1]
                and result.get("date") == key[-2] and result.get("qvix") is not None)

    async def get(self, path, now=None):
        clock = now or dt.datetime.now(qvix_core.CST)
        phase, _ = qvix_core.market_phase(clock, clock.weekday() < 5)
        key = (CACHE_VERSION, str(path), clock.date().isoformat(), phase)
        if phase in ("noon", "close"):
            try:
                cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
                if (cached.get("key") == list(key) and isinstance(cached.get("result"), dict)
                        and self.cacheable(key, cached["result"])):
                    return {**cached["result"], "cached": True}
            except (OSError, ValueError, AttributeError):
                pass
        if self.pending and not self.pending.done() and self.pending_key != key:
            # A request from another session/data generation cannot reuse the
            # old result, but still waits so only one lookup process runs.
            with contextlib.suppress(ValueError):
                await asyncio.shield(self.pending)
            return await self.get(path, now=now)
        if self.pending is None or self.pending.done():
            self.pending_key = key
            self.pending = asyncio.create_task(self.query(path, key))
            # A disconnected caller must not leave an unobserved exception.
            self.pending.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return await asyncio.shield(self.pending)

    async def query(self, path, key):
        result = await self.run(path)
        # Failed queries and a result that crossed a session boundary never
        # enter the cache. The date/session key expires lunch at 13:00 and
        # the closing value at midnight; snapshot/code changes also expire it.
        if self.cacheable(key, result):
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                write_json(self.cache_file, {"key": key, "result": result})
            except OSError:
                log.warning("Unable to save QVIX cache", exc_info=True)
        return {**result, "cached": False}

    async def run(self, path):
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), TIMEOUT)
            except asyncio.TimeoutError:
                raise ValueError("实时 QVIX 查询超时，请稍后重试。")
            if process.returncode:
                log.warning("QVIX worker: %s", stderr.decode(errors="replace")[-2000:])
                try:
                    message = json.loads(stdout)["error"]
                except (ValueError, KeyError):
                    message = "实时 QVIX 查询失败，请稍后重试。"
                raise ValueError(message)
            return json.loads(stdout)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def close(self):
        if self.pending and not self.pending.done():
            self.pending.cancel()
            await asyncio.gather(self.pending, return_exceptions=True)


if __name__ == "__main__":
    try:
        print(json.dumps(calculate(Path(sys.argv[1])), ensure_ascii=False, allow_nan=False))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    except Exception:
        logging.exception("QVIX lookup failed")
        print(json.dumps({"error": "实时 QVIX 上游连接失败，请稍后重试。"}, ensure_ascii=False))
        sys.exit(1)
