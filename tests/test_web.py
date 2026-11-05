"""Migration checks: unchanged financial math, immutable data, real HTTP jobs."""
import contextlib
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="fund-web-test-")
ROOT = Path(TMP.name)
os.environ["FUND_WEB_ROOT"] = str(ROOT)
os.environ["FUND_WEB_CACHE"] = str(ROOT / "cache")
os.environ["FUND_ANALYZER_DATA"] = str(ROOT / "snapshots" / "fixture")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import fetcher
from web_common import FILES, write_json, today

# Other test modules may have imported fetcher before this module set its env.
# Explicit paths keep the fixture isolated regardless of discovery order.
Path(os.environ["FUND_ANALYZER_DATA"]).mkdir(parents=True, exist_ok=True)
fetcher._DATA_DIR = os.environ["FUND_ANALYZER_DATA"]
fetcher.DB_PATH = {name: str(Path(fetcher._DATA_DIR) / filename)
                   for name, filename, _ in fetcher.DB_LAYOUT}
fetcher._schema_ready = False


def fixture():
    fetcher.init_db()
    dates = pd.bdate_range(end=pd.Timestamp(today()) - pd.Timedelta(days=1), periods=650)
    funds = []
    c = fetcher._conn()
    for i, code in enumerate(("000002", "000003", "000004", "000005")):
        days = dates if code != "000005" else dates[-30:]
        returns = 0.004 * np.sin(np.arange(len(days)) / 5) - (0.001 if i % 2 == 0 else -0.001)
        nav = np.cumprod(1 + returns)
        rows = [(code, d.strftime("%Y-%m-%d"), float(n), float(r * 100), float(n))
                for d, n, r in zip(days, nav, returns)]
        c.executemany("INSERT INTO fund_nav_daily VALUES (?,?,?,?,?)", rows)
        c.execute("INSERT INTO fund_nav_meta VALUES (?,?,?)", (code, time.time(), days[-1].strftime("%Y-%m-%d")))
        funds.append(dict(code=code, name=f"测试{i}混合C", type="债券型" if i == 2 else "混合型",
                          nav=float(nav[-1]), nav_date=dates[-1].strftime("%Y-%m-%d"),
                          ret_1m=2, ret_3m=3, ret_6m=4, ret_1y=5))
        c.execute("INSERT INTO fund_scale_hist VALUES (?,?,?,?,?)", (code, "2025-12-31", 5, "2026-01-22", time.time()))
    c.execute("INSERT INTO fund_list VALUES (1,?,?)", (json.dumps(funds), time.time()))
    sse = pd.DataFrame(dict(date=dates.strftime("%Y-%m-%d"),
                            close=3000 * np.cumprod(1 + .002 * np.sin(np.arange(len(dates)) / 5)),
                            pct=.2 * np.sin(np.arange(len(dates)) / 5)))
    c.execute("INSERT INTO index_daily_cache VALUES ('sse',?,?)", (sse.to_json(orient="split"), time.time()))
    c.executemany("INSERT INTO qvix_self_history(date,qvix,note) VALUES (?,?,?)",
                  [(d.strftime("%Y-%m-%d"), 20.0 + float(np.sin(i)), "ok") for i, d in enumerate(dates)])
    c.commit()
    c.close()
    fetcher.recompute_all()
    path = ROOT / "snapshots" / "fixture"
    write_json(path / "manifest.json", dict(version="fixture", summary=dict(
        nav_date=dates[-1].strftime("%Y-%m-%d"), sse_date=dates[-1].strftime("%Y-%m-%d"),
        qvix_date=dates[-1].strftime("%Y-%m-%d"), nav_funds=4, funds=4, synced_at=time.time())))
    (ROOT / "current").symlink_to("snapshots/fixture")
    return path


PATH = fixture()


class CalculationTests(unittest.TestCase):
    def test_upstream_holdings_cache_survives_code_and_data_generations(self):
        from web_cache import lookup_cache
        cached = lookup_cache(PATH)
        row = {"quarter": f"{today().year}Q1", "kind": "股票", "代码": "600001", "名称": "测试持仓",
               "占净值比例": 5., "持股数": 1., "持仓市值": 2.}
        with contextlib.closing(sqlite3.connect(cached)) as c:
            c.execute("INSERT OR REPLACE INTO fund_holdings VALUES (?,?,?,?)",
                      ("000002", str(today().year), json.dumps([row]), time.time()))
            c.commit()
        for version in ("first-code", "second-code"):
            request = ROOT / f"holding-{version}.json"
            output = ROOT / f"holding-{version}-result.json"
            write_json(request, dict(snapshot=str(PATH), code_version=version, operation="holdings",
                                     args=dict(code="000002", start_year=today().year)))
            subprocess.run([sys.executable, "web_tasks.py", str(request), str(output)],
                           check=True, capture_output=True, timeout=15,
                           cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(json.loads(output.read_text())["rows"][0]["名称"], "测试持仓")
        self.assertEqual(lookup_cache(ROOT / "snapshots" / "new-generation"), cached)
    def test_low_memory_matches_original_returns_and_boundaries(self):
        for asof in (today().isoformat(), "2026-06-15", "2025-01-01"):
            for cols in ({"ret_3m"}, {"ret_1m", "ret_1y"}):
                vector = fetcher.compute_metrics_asof(asof, cols=cols)
                sequential = fetcher.compute_metrics_asof(asof, cols=cols, low_memory=True)
                self.assertEqual(set(vector), set(sequential))
                for code in vector:
                    for col in cols:
                        a, b = vector[code].get(col), sequential[code].get(col)
                        if a is None or b is None:
                            self.assertIs(a, b)
                        else:
                            self.assertAlmostEqual(a, b, places=9)

    def test_worker_never_writes_authoritative_snapshots(self):
        before = {name: hashlib.sha256((PATH / filename).read_bytes()).hexdigest()
                  for name, filename in FILES.items()}
        request = ROOT / "detail-request.json"
        output = ROOT / "detail-result.json"
        write_json(request, dict(snapshot=str(PATH), operation="detail", args={"code": "000002"}))
        subprocess.run([sys.executable, "web_tasks.py", str(request), str(output)],
                       check=True, capture_output=True, timeout=15,
                       cwd=Path(__file__).resolve().parents[1])
        result = json.loads(output.read_text())
        self.assertEqual(result["info"]["code"], "000002")
        self.assertEqual(len(result["nav"]), 650)
        for name, filename in FILES.items():
            self.assertEqual(before[name], hashlib.sha256((PATH / filename).read_bytes()).hexdigest())

    def test_precomputed_date_index_preserves_freshness_and_first_dates(self):
        import web_sync
        import today_pick
        web_sync.build_meta(PATH)
        with contextlib.closing(sqlite3.connect(PATH / "web_meta.db")) as meta:
            indexed = meta.execute("SELECT code,first_date FROM nav_dates ORDER BY code").fetchall()
            last = meta.execute("SELECT MAX(last_date) FROM nav_dates").fetchone()[0]
        original = fetcher.nav_first_dates().sort_values("code")
        self.assertEqual(indexed, [(r.code, r.first_nav_date.strftime("%Y-%m-%d")) for r in original.itertuples()])
        with contextlib.closing(fetcher._conn()) as c:
            before = today_pick.data_freshness(c, today().isoformat())
            after = today_pick.data_freshness(c, today().isoformat(), nav_max=last)
        self.assertEqual(before, after)

    def test_stale_nav_is_rejected_instead_of_calculated(self):
        stale = ROOT / "snapshots" / "stale"
        stale.mkdir()
        for filename in FILES.values():
            shutil.copyfile(PATH / filename, stale / filename)
        c = sqlite3.connect(stale / FILES["nav"])
        date = c.execute("SELECT DISTINCT date FROM fund_nav_daily ORDER BY date DESC LIMIT 1 OFFSET 2").fetchone()[0]
        c.execute("DELETE FROM fund_nav_daily WHERE date>?", (date,))
        c.commit()
        c.close()
        request = ROOT / "stale-request.json"
        output = ROOT / "stale-result.json"
        write_json(request, dict(snapshot=str(stale), operation="pick", args={"asof": today().isoformat()}))
        process = subprocess.run([sys.executable, "web_tasks.py", str(request), str(output)],
                                 capture_output=True, text=True, timeout=15,
                                 cwd=Path(__file__).resolve().parents[1])
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("数据尚未同步", process.stderr)
        self.assertFalse(output.exists())

    def test_download_failure_preserves_current_generation(self):
        import web_sync
        current = (ROOT / "current").resolve()
        remote = {f + ".gz": {"updated": "new", "size": 1024} for f in FILES.values()}
        failure = subprocess.CalledProcessError(1, ["gh"])
        with patch.object(web_sync, "assets", return_value=remote), \
                patch.object(web_sync.subprocess, "run", side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError):
                web_sync.sync()
        self.assertEqual((ROOT / "current").resolve(), current)
        self.assertFalse(list((ROOT / "snapshots").glob(".staging-*")))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import web_service
        abandoned = ROOT / "cache" / "jobs" / ("0" * 64)
        abandoned.mkdir(parents=True)
        write_json(abandoned / "state.json", dict(id="0" * 64, status="queued",
                   operation="pick", updated_at=time.time()))
        cls.client = TestClient(web_service.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def wait(self, task):
        r = self.client.post("/api/v1/jobs", json=task)
        self.assertEqual(r.status_code, 202, r.text)
        job_id = r.json()["id"]
        for _ in range(100):
            r = self.client.get("/api/v1/jobs/" + job_id)
            state = r.json()
            if state["status"] in ("completed", "error"):
                self.assertEqual(state["status"], "completed", state)
                return job_id, state["result"]
            time.sleep(.05)
        self.fail("job did not finish")

    def test_static_status_and_validation(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/health").json(), {"ok": True})
        self.assertEqual(self.client.get("/api/v1/status").json()["nav_funds"], 4)
        for task in [dict(operation="detail", code="' OR 1"),
                     dict(operation="pick", asof="2099-01-01"),
                     dict(operation="probe", codes=["bad"]),
                     dict(operation="filter", min_ret="nan")]:
            self.assertEqual(self.client.post("/api/v1/jobs", json=task).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/jobs/anything").status_code, 404)

    def test_restart_does_not_leave_pollers_pending(self):
        state = self.client.get("/api/v1/jobs/" + "0" * 64).json()
        self.assertEqual(state["status"], "error")
        self.assertIn("服务已更新", state["error"])

    def test_filter_pagination_cache_and_csv(self):
        task = dict(operation="filter", period="3m", min_ret=None, max_dd=None)
        job_id, result = self.wait(task)
        # Bond funds and the 30-day-old fund cannot pass the 3-month screen.
        self.assertEqual({r["code"] for r in result["rows"]}, {"000002", "000003"})
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}?page_size=1&page=2").json()["result"]["rows"][0]["code"], result["rows"][1]["code"])
        again = self.client.post("/api/v1/jobs", json=task).json()
        self.assertTrue(again["cached"])
        self.assertEqual(again["id"], job_id)
        csv = self.client.get(f"/api/v1/jobs/{job_id}/download")
        self.assertIn("000002", csv.text)
        self.assertEqual(csv.status_code, 200)

    def test_pick_and_probe_share_the_same_calculation(self):
        _, selected = self.wait(dict(operation="pick"))
        _, probe = self.wait(dict(operation="probe", codes=["000002"]))
        row = next(r for r in selected["rows"] if r["code"] == "000002")
        self.assertAlmostEqual(probe["rows"][0]["ret3m"], row["ret3m"], places=9)

    def test_prepared_probes_bypass_full_analysis_queue_and_slow_holdings(self):
        import web_service
        self.wait(dict(operation="prepare"))
        before = web_service.jobs.queue.qsize()
        with patch.object(web_service.jobs.queue, "full", return_value=True), \
                patch.object(web_service.jobs, "network_current", object()):
            r = self.client.post("/api/v1/jobs", json=dict(
                operation="probe", codes=["000003", "000002", "999999"]))
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["status"], "completed")
        self.assertTrue(r.json()["cached"])
        result = self.client.get("/api/v1/jobs/" + r.json()["id"]).json()["result"]
        self.assertEqual([row["code"] for row in result["rows"]], ["000003", "000002", "999999"])
        self.assertIn("榜单里没有", result["rows"][2]["why"])
        self.assertEqual(web_service.jobs.queue.qsize(), before)

    def test_holding_request_uses_separate_queue_when_analysis_is_full(self):
        import web_service
        async def submit():
            isolated = web_service.Jobs()
            for i in range(8):
                isolated.queue.put_nowait(str(i))
            result = await isolated.submit(web_service.Task(operation="holdings", code="000003"))
            self.assertEqual(isolated.network_queue.qsize(), 1)
            self.assertEqual(isolated.queue.qsize(), 8)
            self.assertEqual(result["status"], "queued")
        asyncio.run(submit())

    def test_new_generation_invalidates_completed_cache(self):
        task = dict(operation="detail", code="000002")
        old_id, _ = self.wait(task)
        second = ROOT / "snapshots" / "second"
        second.mkdir()
        for filename in FILES.values():
            shutil.copyfile(PATH / filename, second / filename)
        manifest = json.loads((PATH / "manifest.json").read_text())
        manifest["version"] = "second"
        write_json(second / "manifest.json", manifest)
        (ROOT / "next").symlink_to("snapshots/second")
        os.replace(ROOT / "next", ROOT / "current")
        try:
            new_id, _ = self.wait(task)
            self.assertNotEqual(old_id, new_id)
            self.assertEqual(self.client.get("/api/v1/status").json()["version"], "second")
        finally:
            (ROOT / "restore").symlink_to("snapshots/fixture")
            os.replace(ROOT / "restore", ROOT / "current")


if __name__ == "__main__":
    unittest.main()
