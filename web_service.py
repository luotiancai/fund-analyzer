"""Small HTTP service: static UI, disk caches, one bounded analysis subprocess."""
import asyncio
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from web_common import CACHE, CST, FILES, snapshot, today, write_json
from web_cache import prepared_result
from fund_rules import is_fof

log = logging.getLogger("fund-web")
HERE = Path(__file__).resolve().parent
WORKER = HERE / "web_tasks.py"
CODE_VERSION = hashlib.sha256(b"".join((HERE / name).read_bytes() for name in (
    "web_service.py", "web_tasks.py", "web_cache.py", "fetcher.py", "today_pick.py", "backtest_qvix.py",
    "fund_rules.py"))).hexdigest()[:16]
JOB_TIMEOUT = int(os.environ.get("FUND_WEB_JOB_TIMEOUT", "240"))


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    operation: Literal["pick", "probe", "filter", "detail", "holdings", "prepare"]
    asof: Optional[dt.date] = None
    code: Optional[str] = Field(default=None, pattern=r"^[0-9]{6}$")
    codes: list[str] = Field(default_factory=list, max_length=20)
    period: Literal["1m", "3m", "6m", "1y"] = "1y"
    types: list[str] = Field(default_factory=list, max_length=30)
    min_ret: Optional[float] = Field(default=0, ge=-100, le=100000)
    max_dd: Optional[float] = Field(default=20, ge=0, le=100)
    min_aum: float = Field(default=0, ge=0, le=100000)
    start_year: Optional[int] = Field(default=None, ge=2020)

    @model_validator(mode="after")
    def valid(self):
        if self.asof and not (dt.date(2020, 1, 1) <= self.asof <= today()):
            raise ValueError("日期须在 2020-01-01 至今天之间")
        if self.start_year and self.start_year > today().year:
            raise ValueError("持仓年份不能晚于今年")
        if self.operation in ("detail", "holdings") and not self.code:
            raise ValueError("请输入六位基金代码")
        if self.operation == "probe" and (not self.codes or any(
                len(c) != 6 or not c.isascii() or not c.isdigit() for c in self.codes)):
            raise ValueError("请输入最多 20 个六位基金代码")
        if any(len(t) > 80 for t in self.types):
            raise ValueError("基金类型过长")
        return self

    def arguments(self):
        if self.operation in ("detail", "holdings"):
            return {"code": self.code, **({"start_year": self.start_year or today().year}
                                         if self.operation == "holdings" else {})}
        if self.operation in ("pick", "probe", "prepare"):
            args = {"asof": (self.asof or today()).isoformat()}
            if self.operation == "probe":
                args["codes"] = list(dict.fromkeys(self.codes))
            return args
        return dict(asof=(self.asof or today()).isoformat(),
                    period=self.period, types=sorted(set(self.types)),
                    min_ret=self.min_ret, max_dd=self.max_dd, min_aum=self.min_aum)


class Jobs:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=8)
        self.network_queue = asyncio.Queue(maxsize=4)
        self.active = set()
        self.lock = asyncio.Lock()
        self.current = None
        self.network_current = None
        self.runner = None
        self.network_runner = None

    def folder(self, job_id):
        return CACHE / "jobs" / job_id

    def state(self, job_id):
        try:
            return json.loads((self.folder(job_id) / "state.json").read_text())
        except FileNotFoundError:
            raise HTTPException(404, "任务不存在，请重新提交")

    def update(self, job_id, **fields):
        path = self.folder(job_id) / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {"id": job_id}
        state.update(fields, updated_at=time.time())
        write_json(path, state)
        return state

    async def submit(self, task):
        try:
            path, manifest = snapshot()
        except (FileNotFoundError, RuntimeError):
            raise HTTPException(503, "数据正在首次同步，请稍后再试")
        args = task.arguments()
        identity = [CODE_VERSION, manifest["version"], task.operation, args]
        job_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        async with self.lock:
            folder = self.folder(job_id)
            if (folder / "state.json").exists():
                state = self.state(job_id)
                if state["status"] == "completed" and (folder / "result.json").exists():
                    return {**state, "cached": True}
                if job_id in self.active:
                    return {**state, "cached": False}
                if (state["status"] == "error" and time.time() - state["updated_at"] < 10
                        and not (state.get("error") or "").startswith("服务已")):
                    return state
            cached = prepared_result(path, CODE_VERSION, task.operation, args)
            if cached is not None:
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "result.json", cached)
                state = self.update(job_id, status="completed", operation=task.operation,
                                    generation=manifest["version"], error=None, finished_at=time.time())
                return {**state, "cached": True}
            queue = self.network_queue if task.operation == "holdings" else self.queue
            if queue.full():
                raise HTTPException(429, "计算队列已满，请稍后重试", headers={"Retry-After": "10"})
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "request.json", {"snapshot": str(path),
                        "code_version": CODE_VERSION, "operation": task.operation, "args": args})
            state = self.update(job_id, status="queued", operation=task.operation,
                                generation=manifest["version"], error=None)
            self.active.add(job_id)
            queue.put_nowait(job_id)
            return {**state, "cached": False}

    async def work(self, network=False):
        queue = self.network_queue if network else self.queue
        while True:
            job_id = await queue.get()
            folder = self.folder(job_id)
            process = None
            try:
                if self.state(job_id)["status"] == "completed":
                    continue
                self.update(job_id, status="running", started_at=time.time())
                with (folder / "stderr.log").open("wb") as stderr:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, str(WORKER), str(folder / "request.json"),
                        str(folder / "result.json"), stdout=asyncio.subprocess.DEVNULL,
                        stderr=stderr,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
                    if network:
                        self.network_current = process
                    else:
                        self.current = process
                    try:
                        await asyncio.wait_for(process.wait(), timeout=JOB_TIMEOUT)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        raise RuntimeError("计算超时，请缩小查询范围后重试")
                if process.returncode != 0:
                    stderr = (folder / "stderr.log").read_text(errors="replace")[-8000:]
                    log.error("job %s failed: %s", job_id, stderr)
                    last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
                    # Expected data validation errors are safe to show. Internal
                    # exceptions stay in journald, without exposing paths/SQL.
                    expected_errors = ("ValueError: ", "today_pick.IncompleteNavError: ")
                    message = (last.split(": ", 1)[1] if last.startswith(expected_errors)
                               else "计算失败，请稍后重试")
                    raise RuntimeError(message)
                if not (folder / "result.json").exists():
                    raise RuntimeError("计算未返回结果，请重试")
                self.update(job_id, status="completed", finished_at=time.time())
            except asyncio.CancelledError:
                self.update(job_id, status="error", error="服务已重启，请重新提交")
                raise
            except Exception as exc:
                log.exception("job %s", job_id)
                self.update(job_id, status="error", error=str(exc))
            finally:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
                if network:
                    self.network_current = None
                else:
                    self.current = None
                self.active.discard(job_id)
                queue.task_done()
                if not self.active:
                    try:
                        self.prune()
                    except OSError:
                        log.exception("cache cleanup failed")

    def prune(self):
        """Bound disposable results; never touch authority data or active jobs."""
        folders = []
        for folder in (CACHE / "jobs").iterdir():
            if folder.is_dir() and folder.name not in self.active:
                size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                folders.append((folder.stat().st_mtime, size, folder))
        total = sum(size for _, size, _ in folders)
        import shutil
        for modified, size, folder in sorted(folders):
            if time.time() - modified > 7 * 86400 or total > 128 * 1048576:
                # Keep at least a minute for the polling client to fetch/export.
                if time.time() - modified > 60:
                    shutil.rmtree(folder)
                    total -= size
        runtime = CACHE / "runtime"
        if runtime.exists():
            from web_common import ROOT
            for folder in runtime.iterdir():
                if folder.is_dir() and not (ROOT / "snapshots" / folder.name.split("-", 1)[0]).exists():
                    if time.time() - folder.stat().st_mtime > 86400:
                        shutil.rmtree(folder)


jobs = None


@contextlib.asynccontextmanager
async def lifespan(app):
    global jobs
    jobs = Jobs()
    CACHE.mkdir(parents=True, exist_ok=True)
    # In-flight tasks belong to the previous process. Polling clients must get
    # an explicit retry rather than wait forever on an abandoned queued state.
    for path in (CACHE / "jobs").glob("*/state.json"):
        state = json.loads(path.read_text())
        if state.get("status") in ("queued", "running"):
            jobs.update(path.parent.name, status="error", error="服务已更新，请重新提交查询")
    jobs.runner = asyncio.create_task(jobs.work())
    jobs.network_runner = asyncio.create_task(jobs.work(network=True))
    try:
        yield
    finally:
        jobs.runner.cancel()
        jobs.network_runner.cancel()
        await asyncio.gather(jobs.runner, jobs.network_runner, return_exceptions=True)


app = FastAPI(title="基金分析仪", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


def read_connection(path, name):
    c = sqlite3.connect((path / FILES[name]).as_uri() + "?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    return c


_status_cache = None


@app.get("/api/v1/status")
def status():
    global _status_cache
    try:
        path, manifest = snapshot()
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(503, "数据正在首次同步")
    if _status_cache is None or _status_cache[0] != path.name:
        with contextlib.closing(read_connection(path, "rank")) as c:
            data = json.loads(c.execute("SELECT data FROM fund_list WHERE id=1").fetchone()[0])
        import re
        types = sorted({r.get("type", "") for r in data if r.get("type")
                        and not is_fof(r["type"], r.get("name"))
                        and not re.search("债|固收", r["type"])})
        _status_cache = (path.name, {**manifest["summary"], "types": types,
                                    "version": manifest["version"], "code_version": CODE_VERSION})
    return {**_status_cache[1], "today": today().isoformat(),
            "queued": jobs.queue.qsize() + jobs.network_queue.qsize(),
            "computing": bool(jobs.current or jobs.network_current)}


@app.get("/api/v1/health")
def health():
    try:
        path, manifest = snapshot()
        ready = (all((path / filename).is_file() for filename in FILES.values())
                 and jobs is not None and jobs.runner is not None and not jobs.runner.done()
                 and jobs.network_runner is not None and not jobs.network_runner.done())
    except (FileNotFoundError, RuntimeError):
        ready = False
    return Response(content='{"ok":' + str(ready).lower() + '}',
                    media_type="application/json", status_code=200 if ready else 503)


@app.post("/api/v1/jobs", status_code=202)
async def submit(task: Task):
    return await jobs.submit(task)


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str, page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500),
            sort: Literal["code", "name", "type", "aum", "ret", "mdd"] = "ret",
            descending: bool = True):
    if len(job_id) != 64 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(404, "任务不存在")
    state = jobs.state(job_id)
    if state["status"] in ("queued", "running") and state["operation"] in ("pick", "probe"):
        request = json.loads((jobs.folder(job_id) / "request.json").read_text())
        result = prepared_result(Path(request["snapshot"]), request["code_version"],
                                 state["operation"], request["args"])
        if result is not None:
            return {**state, "status": "completed", "cached": True, "result": result}
    if state["status"] == "completed":
        result = json.loads((jobs.folder(job_id) / "result.json").read_text())
        if state["operation"] == "filter":
            rows = result["rows"]
            rows = sorted((r for r in rows if r.get(sort) is not None),
                          key=lambda r: r[sort], reverse=descending) + [
                              r for r in rows if r.get(sort) is None]
            result["rows"] = rows[(page - 1) * page_size:page * page_size]
            result.update(page=page, page_size=page_size)
        state["result"] = result
    return state


@app.get("/api/v1/jobs/{job_id}/download")
def download(job_id: str):
    state = get_job(job_id, page=1, page_size=1)
    if state["status"] != "completed" or state["operation"] != "filter":
        raise HTTPException(409, "筛选尚未完成")
    result = json.loads((jobs.folder(job_id) / "result.json").read_text())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["基金代码", "基金名称", "类型", "规模(亿)", "收益率(%)", "最大回撤(%)"])
    for row in result["rows"]:
        # Preserve leading zeroes in spreadsheet applications, without formulas.
        writer.writerow(["\t" + row["code"], row["name"], row["type"], row["aum"], row["ret"], row["mdd"]])
    return Response(content="\ufeff" + stream.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="funds.csv"'})


app.mount("/", StaticFiles(directory=HERE / "web", html=True), name="web")
