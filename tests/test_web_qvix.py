"""Real-time query boundaries, immutable snapshots and isolated execution."""
import asyncio
import datetime as dt
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import qvix_core
import web_qvix


class QvixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="web-qvix-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        with sqlite3.connect(self.path / "qvix.db") as c:
            c.execute("CREATE TABLE qvix_self_history (date TEXT,qvix REAL,threshold REAL)")
            c.executemany("INSERT INTO qvix_self_history VALUES (?,?,?)", [
                ("2026-11-10", 18, 21), ("2026-11-11", 19, 22), ("2026-11-12", 20, 23)])
        with sqlite3.connect(self.path / "fund_rank.db") as c:
            c.execute("CREATE TABLE app_meta (key TEXT,value REAL)")
            c.execute("INSERT INTO app_meta VALUES ('rf_rate',0.012)")

    def now(self, clock="14:40", day="2026-11-12"):
        return dt.datetime.fromisoformat(day + "T" + clock).replace(tzinfo=qvix_core.CST)

    def test_session_boundaries(self):
        for clock, phase, frozen in [
            ("09:29:59", "prev", None), ("09:30", "live", None),
            ("11:30", "noon", "11:30"), ("12:59:59", "noon", "11:30"),
            ("13:00", "live", None), ("14:56:59", "live", None),
            ("14:57", "auction", None), ("15:00", "close", "15:00")]:
            with self.subTest(clock=clock):
                result, asof = qvix_core.market_phase(self.now(clock), True)
                self.assertEqual(result, phase)
                self.assertEqual(asof.strftime("%H:%M") if asof else None, frozen)
        self.assertEqual(qvix_core.market_phase(self.now(), False), ("prev", None))

    def test_live_uses_prior_threshold_and_preserves_databases(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in self.path.iterdir()}
        with patch.object(web_qvix, "quote_date", return_value=dt.date(2026, 11, 12)), \
                patch.object(qvix_core, "compute_qvix", return_value=(22.5, "14:40:00")) as compute:
            result = web_qvix.calculate(self.path, self.now())
        self.assertEqual(result["qvix"], 22.5)
        self.assertEqual(result["threshold_date"], "2026-11-11")
        self.assertEqual(result["threshold"], 22)
        self.assertTrue(result["hit"])
        self.assertEqual(result["gap"], .5)
        compute.assert_called_once_with(as_of=None, fallback_rate=.012)
        for name, digest in before.items():
            self.assertEqual(hashlib.sha256((self.path / name).read_bytes()).digest(), digest)

    def test_noon_and_close_freeze_time(self):
        with patch.object(web_qvix, "quote_date", return_value=dt.date(2026, 11, 12)), \
                patch.object(qvix_core, "compute_qvix", return_value=(20, "15:00:00")) as compute:
            for clock, phase, frozen in [("12:20", "noon", "11:30"), ("19:00", "close", "15:00")]:
                result = web_qvix.calculate(self.path, self.now(clock))
                self.assertEqual(result["phase"], phase)
                self.assertEqual(compute.call_args.kwargs["as_of"], self.now(frozen))

    def test_auction_and_failed_quotes_do_not_use_history_as_live_value(self):
        with patch.object(web_qvix, "quote_date", return_value=dt.date(2026, 11, 12)), \
                patch.object(qvix_core, "compute_qvix", return_value=None) as compute:
            with self.assertRaisesRegex(ValueError, "集合竞价"):
                web_qvix.calculate(self.path, self.now("14:58"))
            compute.assert_not_called()
            with self.assertRaisesRegex(ValueError, "计算失败"):
                web_qvix.calculate(self.path, self.now())

    def test_weekday_holiday_and_preopen_read_historical_value(self):
        with patch.object(web_qvix, "quote_date", return_value=dt.date(2026, 11, 12)) as quote, \
                patch.object(qvix_core, "compute_qvix") as compute:
            result = web_qvix.calculate(self.path, self.now("14:40", "2026-11-13"))
            self.assertEqual((result["phase"], result["date"], result["qvix"]),
                             ("prev", "2026-11-12", 20))
            self.assertEqual(result["threshold"], 23)
            quote.reset_mock()
            result = web_qvix.calculate(self.path, self.now("09:15"))
            self.assertEqual(result["phase"], "prev")
            self.assertEqual(result["date"], "2026-11-11")
            quote.assert_not_called()
            compute.assert_not_called()

    def test_invalid_quote_date_is_rejected(self):
        with patch.object(qvix_core, "_get", return_value='var hq_str_sh510050="";'):
            with self.assertRaises(ValueError):
                web_qvix.quote_date()

    def test_stale_upstream_quote_cannot_select_older_history(self):
        with patch.object(web_qvix, "quote_date", return_value=dt.date(2026, 11, 10)):
            with self.assertRaisesRegex(ValueError, "日期.*不一致"):
                web_qvix.calculate(self.path, self.now())


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_automatic_query_and_concurrent_clicks_share_one_worker(self):
        lookup = web_qvix.LiveQvix()
        self.assertIsNone(lookup.pending)
        started, release = asyncio.Event(), asyncio.Event()

        async def run(path):
            started.set()
            await release.wait()
            return {"qvix": 22}

        with patch.object(lookup, "run", side_effect=run) as worker:
            a = asyncio.create_task(lookup.get(Path("/snapshot")))
            await started.wait()
            b = asyncio.create_task(lookup.get(Path("/snapshot")))
            await asyncio.sleep(0)
            release.set()
            self.assertEqual(await a, await b)
            self.assertEqual(worker.call_count, 1)
            await lookup.get(Path("/snapshot"))
            self.assertEqual(worker.call_count, 2)

    async def test_timeout_kills_worker(self):
        process = AsyncMock()
        process.returncode = None
        process.kill = unittest.mock.Mock()
        process.communicate.side_effect = asyncio.TimeoutError
        with patch.object(asyncio, "create_subprocess_exec", return_value=process):
            with self.assertRaisesRegex(ValueError, "超时"):
                await web_qvix.LiveQvix().run(Path("/snapshot"))
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
