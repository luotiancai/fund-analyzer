"""Batch storage omits annualization; detail calculations keep it on demand."""
import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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
        with contextlib.closing(fetcher._conn()) as c:
            c.executemany("INSERT INTO fund_inception VALUES (?,?,?,?)",
                          [(code, "2020-01-01", "fixture", time.time()) for code in ("000001", "000005")])
            c.commit()
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

    def test_scale_maintenance_keeps_only_c_shares_even_if_old_nav_exists(self):
        funds = pd.DataFrame([
            dict(code="000001", name="普通混合C", type="混合型"),
            dict(code="000002", name="普通混合A", type="混合型"),
            dict(code="000003", name="新增混合C", type="混合型"),
            dict(code="000004", name="资产配置C", type="FOF-混合型"),
            dict(code="000005", name="测试债券C", type="债券型"),
        ])
        fetcher._save_fund_list_cache(funds)
        self.assertEqual(fetcher.scale_universe(), ["000001", "000003"])
        fetcher._save_fund_list_cache(funds[funds["code"] == "000002"])
        self.assertEqual(fetcher.scale_universe(), [])

    def test_new_standard_moves_previous_standard_to_history_without_changing_trades(self):
        old_trades = [{"买入日": "2024-01-01", "卖出日": "2024-02-01", "费后收益": 10.}]
        new_trades = [{"买入日": "2024-01-01", "卖出日": "2024-02-02", "费后收益": 5.}]
        old = fetcher.save_strategy_run(old_trades, {"aum_basis": "merged"}, is_standard=True)
        comparison = fetcher.save_strategy_run(new_trades, {"aum_basis": "single"})
        self.assertEqual(fetcher.load_backtest_trades()[3], old)
        current = fetcher.save_strategy_run(new_trades, {"aum_basis": "single"}, is_standard=True)
        self.assertEqual(fetcher.load_backtest_trades()[3], current)
        with contextlib.closing(fetcher._conn()) as c:
            rows = {r["id"]: dict(r) for r in c.execute("SELECT * FROM strategy_runs")}
        self.assertEqual(rows[old]["is_standard"], 0)
        self.assertEqual(rows[comparison]["is_standard"], 0)
        self.assertEqual(rows[current]["is_standard"], 1)
        self.assertEqual(json.loads(rows[old]["trades"]), old_trades)
        self.assertEqual(json.loads(rows[current]["params"])["review_notes"], {})

    def test_failed_standard_insert_keeps_previous_standard_active(self):
        trades = [{"买入日": "2024-01-01", "卖出日": "2024-02-01", "费后收益": 10.}]
        old = fetcher.save_strategy_run(trades, {"aum_basis": "merged"}, is_standard=True)
        with contextlib.closing(fetcher._conn()) as c:
            c.execute("CREATE TRIGGER strategy.reject_new_run BEFORE INSERT ON strategy_runs "
                      "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
            c.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            fetcher.save_strategy_run(trades, {"aum_basis": "single"}, is_standard=True)
        self.assertEqual(fetcher.load_backtest_trades()[3], old)

    def test_own_share_aum_excludes_a_and_unpublished_quarters(self):
        funds = pd.DataFrame([
            dict(code="000001", name="普通混合C", type="混合型"),
            dict(code="000003", name="普通混合A", type="混合型"),
            dict(code="000004", name="缺失混合C", type="混合型"),
            dict(code="000005", name="缺失混合A", type="混合型"),
        ])
        fetcher._save_fund_list_cache(funds)
        with contextlib.closing(fetcher._conn()) as c:
            c.executemany("INSERT INTO fund_scale_hist VALUES (?,?,?,?,?)", [
                ("000001", "2025-09-30", 1.9, "2025-10-25", time.time()),
                ("000001", "2025-12-31", 3.0, "2026-01-25", time.time()),
                ("000003", "2025-09-30", 10., "2025-10-25", time.time()),
                ("000005", "2025-09-30", 20., "2025-10-25", time.time()),
            ])
            c.commit()
        asof = "2026-01-10"
        with patch.object(fetcher, "_SIBLING_MAP", None):
            self.assertEqual(fetcher.funds_aum_asof(["000001", "000004"], asof),
                             {"000001": 1.9, "000004": None})
            self.assertEqual(fetcher.fund_aum_asof("000001", asof, fetch_if_missing=False), 1.9)
            self.assertIsNone(fetcher.fund_aum_asof("000004", asof, fetch_if_missing=False))
            merged = fetcher.funds_aum_asof(["000001", "000004"], asof, merge_classes=True)
            self.assertAlmostEqual(merged["000001"], 11.9)
            self.assertEqual(merged["000004"], 20.)
            self.assertAlmostEqual(fetcher.fund_aum_asof("000001", asof,
                                                       fetch_if_missing=False, merge_classes=True), 11.9)
        many = fetcher.funds_aum_asof([f"{i:06d}" for i in range(1200)], asof)
        self.assertEqual(len(many), 1200)
        self.assertEqual(many["000001"], 1.9)

    def test_backtest_default_uses_c_share_aum_and_includes_exact_threshold(self):
        import backtest_qvix as B
        asof = (self.nav["date"].iloc[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        fetcher._save_fund_list_cache(pd.DataFrame([
            dict(code="000001", name="普通混合C", type="混合型"),
            dict(code="000003", name="普通混合A", type="混合型"),
        ]))
        with contextlib.closing(fetcher._conn()) as c:
            c.executemany("INSERT INTO fund_scale_hist VALUES (?,?,?,?,?)", [
                ("000001", "2023-03-31", 1.9, "2023-04-25", time.time()),
                ("000003", "2023-03-31", 10., "2023-04-25", time.time()),
            ])
            c.commit()
            with patch.object(B, "_metrics_asof_cached", return_value={"000001": {"ret_3m": -10.}}), \
                    patch.object(fetcher, "_SIBLING_MAP", None):
                args = dict(pick="bottom", min_aum=2.0)
                self.assertIsNone(B.find_champion_on_date(c, asof, **args)[0])
                self.assertEqual(B.find_champion_on_date(c, asof, aum_basis="merged", **args)[0], "000001")
                c.execute("UPDATE fund_scale_hist SET aum=2.0 WHERE code='000001'")
                c.commit()
                self.assertEqual(B.find_champion_on_date(c, asof, **args)[0], "000001")

    def test_inception_uses_share_class_estabdate_and_rejects_bad_responses(self):
        response = Mock()
        response.json.return_value = dict(ErrCode=0, Datas=dict(
            FCODE="029368", ESTABDATE="2026-10-19", ISSBDATE="2010-08-30"))
        with patch.object(fetcher.requests, "get", return_value=response):
            self.assertEqual(fetcher._fetch_fund_inception_date("029368"), "2026-10-19")
        for data in (dict(FCODE="000001", ESTABDATE="2020-01-01"),
                     dict(FCODE="029368", ESTABDATE="--"),
                     dict(FCODE="029368", ESTABDATE="2026-02-30"),
                     dict(FCODE="029368", ESTABDATE="2999-01-01")):
            with self.subTest(data=data), patch.object(fetcher.time, "sleep"), \
                    patch.object(fetcher.requests, "get", return_value=response), \
                    self.assertLogs(fetcher.logger, level="WARNING"):
                response.json.return_value = dict(ErrCode=0, Datas=data)
                self.assertIsNone(fetcher._fetch_fund_inception_date("029368"))

    def test_inception_cache_only_fetches_missing_eligible_c_shares(self):
        funds = pd.DataFrame([
            dict(code="000001", name="已缓存混合C", type="混合型"),
            dict(code="000002", name="新增混合C", type="混合型"),
            dict(code="000003", name="资产配置C", type="FOF-稳健型"),
            dict(code="000004", name="测试3个月定开混合C", type="混合型"),
            dict(code="000005", name="普通混合A", type="混合型"),
            dict(code="000006", name="待重试混合C", type="混合型"),
            dict(code="000007", name="测试债券C", type="债券型"),
            dict(code="000008", name="测试海外C", type="QDII-普通股票"),
        ])
        with contextlib.closing(fetcher._conn()) as c:
            c.execute("INSERT INTO fund_inception VALUES (?,?,?,?)",
                      ("000001", "2010-01-01", "fixture", time.time()))
            c.commit()
        with patch.object(fetcher, "_fetch_fund_inception_date",
                          side_effect=lambda code: "2026-10-19" if code == "000002" else None) as lookup:
            result = fetcher.refresh_inception_dates(funds, workers=1)
            self.assertEqual({call.args[0] for call in lookup.call_args_list}, {"000002", "000006"})
        self.assertEqual(result, dict(total=3, cached=1, fetched=1, failed=["000006"]))
        with contextlib.closing(fetcher._conn()) as c:
            self.assertEqual(c.execute("SELECT inception_date FROM fund_inception WHERE code='000002'")
                             .fetchone()[0], "2026-10-19")
            self.assertIsNone(c.execute("SELECT * FROM fund_inception WHERE code='000006'").fetchone())
        with patch.object(fetcher, "_fetch_fund_inception_date", return_value="2026-01-01") as lookup:
            retry = fetcher.refresh_inception_dates(funds, workers=1)
            lookup.assert_called_once_with("000006")
            self.assertEqual(retry, dict(total=3, cached=2, fetched=1, failed=[]))

    def test_short_nav_is_stored_but_calculation_thresholds_are_preserved(self):
        short = self.nav.tail(5).reset_index(drop=True)
        with patch.object(fetcher, "_fetch_nav_full", return_value=short) as download:
            # Existing callers still require 20 points; storage keeps all 5.
            self.assertIsNone(fetcher.fetch_nav("000003"))
            self.assertEqual(len(fetcher._load_nav_df("000003")), 5)
            self.assertEqual(len(fetcher.fetch_nav("000003", min_points=1)), 5)
            download.assert_called_once_with("000003")
        self.assertIsNone(fetcher._metrics_from_nav(short))
        asof = (short["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertNotIn("000003", fetcher.compute_metrics_asof(asof, cols={"ret_3m"}))
        self.assertNotIn("000003", fetcher.compute_metrics_asof(
            asof, cols={"ret_3m", "mdd_3m"}, low_memory=True))

    def test_backfill_counts_short_history_as_saved(self):
        short = self.nav.tail(3).reset_index(drop=True)
        with patch.object(fetcher, "_fetch_nav_full", return_value=short):
            self.assertEqual(fetcher._backfill_codes(["000003"], workers=1), 1)
        self.assertEqual(len(fetcher._load_nav_df("000003")), 3)

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
