"""Batch storage omits annualization; detail calculations keep it on demand."""
import contextlib
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
                 for name, filename, _, _ in fetcher.DB_LAYOUT}
        for name, value in (("DB_PATH", paths), ("_schema_ready", False), ("LAZY_NAV", False)):
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
        with patch.object(fetcher, "fetch_nav", return_value=self.nav), \
                patch.object(fetcher, "_annualized", wraps=fetcher._annualized) as annualized:
            result = fetcher.compute_metrics_for_fund("000001")
        annualized.assert_called_once()
        self.assertIsNotNone(result["ann_return"])
        self.assertIsNotNone(result["volatility"])
        stored = fetcher.load_all_precomputed()["000001"]
        self.assertNotIn("ann_return", stored)
        self.assertNotIn("volatility", stored)


if __name__ == "__main__":
    unittest.main()
