"""Batch storage omits annualization; detail calculations keep it on demand."""
import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fetcher


class DailyMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fund-daily-metrics-")
        self.addCleanup(self.tmp.cleanup)
        paths = {name: str(Path(self.tmp.name) / filename)
                 for name, filename, _ in fetcher.DB_LAYOUT}
        for name, value in (("DB_PATH", paths), ("_schema_ready", False)):
            p = patch.object(fetcher, name, value)
            p.start()
            self.addCleanup(p.stop)
        fetcher.init_db()
        dates = pd.bdate_range("2023-01-02", periods=500)
        growth = .003 * np.sin(np.arange(len(dates)) / 7)
        self.nav = pd.DataFrame(dict(date=dates, nav=np.cumprod(1 + growth),
                                     acc_nav=np.cumprod(1 + growth), daily_ret_pct=growth * 100))
        self.flat = pd.DataFrame(dict(date=dates, nav=1., acc_nav=1., daily_ret_pct=0.))
        with contextlib.closing(fetcher._conn()) as c:
            fetcher._write_nav_rows(c, "000001", self.nav)
            fetcher._write_nav_rows(c, "000002", self.flat)
            c.commit()

    def test_daily_pipeline_skips_new_and_stored_fof_funds(self):
        end = self.nav["date"].iloc[-1]
        funds = pd.DataFrame([
            dict(code="000001", name="普通混合C", type="混合型"),
            dict(code="000002", name="嘉实领航资产配置混合C", type="FOF-稳健型"),
            dict(code="000003", name="新增资产配置C", type="FOF-混合型"),
            dict(code="000004", name="新增混合(FOF-LOF)C", type="未知"),
            dict(code="000005", name="新增普通混合C", type="混合型"),
        ])
        funds["nav_date"] = end + pd.offsets.BDay(1)
        funds["nav"] = funds["acc_nav"] = 1.
        funds["daily_ret"] = 0.
        fetcher._save_fund_list_cache(funds)
        with patch.object(fetcher, "fetch_fund_list", return_value=funds), \
                patch.object(fetcher, "_backfill_codes", return_value=0) as backfill:
            result = fetcher.run_pipeline()
        self.assertEqual(backfill.call_args.args[0], ["000005"])
        self.assertEqual(result["appended"], 1)
        self.assertEqual(result["recomputed"], 1)
        with contextlib.closing(fetcher._conn()) as c:
            self.assertEqual(c.execute("SELECT MAX(date) FROM fund_nav_daily WHERE code='000002'")
                             .fetchone()[0], end.strftime("%Y-%m-%d"))
        self.assertEqual(fetcher.scale_universe(), ["000001", "000005"])

    def test_comparison_pool_excludes_fof_by_type_or_name(self):
        import pct_rank
        funds = [dict(code="000001", name="嘉实领航资产配置混合C", type="FOF-稳健型"),
                 dict(code="000002", name="测试混合(FOF)C", type="混合型"),
                 dict(code="000003", name="普通混合C", type="混合型")]
        with contextlib.closing(fetcher._conn()) as c:
            c.execute("INSERT INTO fund_list VALUES (1,?,?)", (json.dumps(funds), time.time()))
            self.assertEqual(pct_rank._pool_universe(c)[2], {"000001", "000002"})

    def test_batch_skips_annualization_and_preserves_all_eight_period_metrics(self):
        period_cols = set(fetcher.RETURN_DAYS) | set(fetcher.DRAWDOWN_DAYS)
        expected = fetcher._metrics_from_nav(self.nav, cols=period_cols)
        with patch.object(fetcher, "_annualized", side_effect=AssertionError("batch annualized")) as annualized:
            self.assertEqual(fetcher.recompute_all(), 2)
            annualized.assert_not_called()
        stored = fetcher.load_all_precomputed()
        self.assertEqual(set(stored["000001"]), period_cols | {"data_points"})
        for col, value in expected.items():
            self.assertAlmostEqual(stored["000001"][col], value, places=12)
        self.assertEqual(stored["000002"]["mdd_1y"], 0)

    def test_existing_storage_is_migrated_without_losing_period_values(self):
        if sqlite3.sqlite_version_info < (3, 35):
            self.skipTest("SQLite before DROP COLUMN support")
        fetcher.recompute_all()
        before = fetcher.load_all_precomputed()
        with contextlib.closing(fetcher._conn()) as c:
            c.execute("ALTER TABLE fund_sharpe ADD COLUMN ann_return REAL")
            c.execute("ALTER TABLE fund_sharpe ADD COLUMN volatility REAL")
            c.execute("UPDATE fund_sharpe SET ann_return=1.2,volatility=.3")
            c.commit()
        # The reader also works before migration, against yesterday's assets.
        self.assertEqual(fetcher.load_all_precomputed(), before)
        fetcher.init_db()
        with contextlib.closing(fetcher._conn()) as c:
            cols = {r["name"] for r in c.execute("PRAGMA rank.table_info(fund_sharpe)")}
        self.assertFalse({"ann_return", "volatility"} & cols)
        self.assertEqual(fetcher.load_all_precomputed(), before)

    def test_detail_returns_annualized_values_without_persisting_them(self):
        import web_tasks
        before = fetcher.load_all_precomputed()
        with patch.object(fetcher, "_load_nav_df", return_value=self.nav), \
                patch.object(fetcher, "_annualized", wraps=fetcher._annualized) as annualized:
            result = web_tasks.detail(fetcher, pd.DataFrame({"code": ["000001"]}),
                                      fetcher._conn, "000001")["metrics"]
        annualized.assert_called_once()
        self.assertIsNotNone(result["ann_return"])
        self.assertIsNotNone(result["volatility"])
        self.assertEqual(fetcher.load_all_precomputed(), before)

    def test_holdings_loads_requested_year_and_preserves_etf_passthrough(self):
        year = fetcher.datetime.now().year
        with patch.object(fetcher, "resolve_target_etf", side_effect=[("510000", "测试ETF"), None]), \
                patch.object(fetcher, "_fetch_holdings_year", return_value=pd.DataFrame()) as fetch_year:
            result = fetcher.fetch_holdings("000001", start_year=year)
        fetch_year.assert_called_once_with("510000", str(year))
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
