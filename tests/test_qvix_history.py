from contextlib import closing
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import fetcher
import qvix_calc


class QvixHistoryTests(unittest.TestCase):
    def test_sina_close_parser_selects_requested_dates_and_rejects_zero_prices(self):
        begin, end = dt.date(2026, 11, 19), dt.date(2026, 11, 20)
        with patch.object(qvix_calc.requests, "get") as get:
            get.return_value.text = ('/* Sina */\n([{"d":"2026-11-18","c":"0.01"},'
                                     '{"d":"2026-11-19","c":"0"},'
                                     '{"d":"2026-11-20","c":"0.0724"}]);')
            closes = qvix_calc._fetch_sina_closes("10011425", begin, end)
        self.assertEqual(closes, {begin: None, end: 0.0724})
        self.assertEqual(get.call_args.kwargs["params"], {"symbol": "CON_OP_10011425"})

    def test_sina_refresh_keeps_previous_values_and_uses_separate_history(self):
        prior, target = dt.date(2026, 11, 19), dt.date(2026, 11, 20)
        calendar = {prior, target}
        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: str(Path(tmp) / file) for name, file, _ in fetcher.DB_LAYOUT}
            with patch.multiple(fetcher, DB_PATH=paths, _schema_ready=False), \
                    patch.object(fetcher, "datetime", wraps=dt.datetime) as clock, \
                    patch.object(fetcher, "_trading_days", return_value=calendar), \
                    patch.object(qvix_calc, "compute_qvix_history") as compute:
                clock.now.return_value = dt.datetime(2026, 11, 21, 7)
                fetcher.save_qvix_self_history([{"date": str(target), "qvix": 15.66}],
                                               trading_days=calendar)
                compute.return_value = {prior: (14.9, None), target: (15.7, None)}
                self.assertEqual(fetcher.update_qvix_sina_daily(), (15.7, None))
                compute.assert_called_once_with([prior, target], trading_days=calendar, price_source="sina")
                compute.return_value = {target: (None, "接口超时")}
                self.assertEqual(fetcher.update_qvix_sina_daily(), (None, "接口超时"))
                compute.assert_called_with([target], trading_days=calendar, price_source="sina")
                self.assertEqual(fetcher.load_qvix_sina_history(), {str(prior): 14.9, str(target): 15.7})
                self.assertEqual(fetcher.load_qvix_self_history()["qvix"].tolist(), [15.66])

    def test_listing_uses_the_contract_even_when_implied_volatility_is_zero(self):
        day = dt.date(2026, 11, 20)
        row = {"TRADE_DATE": str(day), "CONTRACT_ID": "510050C2612M02800",
               "SECURITY_ID": "10011425", "IMPLC_VOLATLTY": "0.000"}
        with patch.object(qvix_calc, "_get_json", return_value={"result": [row]}):
            self.assertEqual(qvix_calc._fetch_contracts(day), [("510050C2612M02800", "10011425")])
            with self.assertRaisesRegex(ValueError, "日期不符"):
                qvix_calc._fetch_contracts(day - dt.timedelta(days=1))

    def test_failed_close_request_is_distinct_from_a_forward_below_the_strikes(self):
        day = dt.date(2020, 2, 3)
        contracts = {("C", 2.5): "1", ("P", 2.5): "2",
                     ("C", 2.6): "3", ("P", 2.6): "4"}
        quotes = {symbol: {day: price} for symbol, price in
                  {"1": 0.01, "2": 0.10, "3": 0.01, "4": 0.20}.items()}
        pairs = [((23, contracts), (51, contracts))]
        curve = [(30, 0.02)]
        value, note = qvix_calc._calculate(day, pairs, quotes, curve)
        self.assertIsNone(value)
        self.assertTrue(qvix_calc.is_chain_broken(note))
        del quotes["2"]
        value, note = qvix_calc._calculate(day, pairs, quotes, curve)
        self.assertIsNone(value)
        self.assertFalse(qvix_calc.is_chain_broken(note))

    def test_missing_data_does_not_erase_confirmed_history_or_change_threshold(self):
        broken = f"{qvix_calc.CHAIN_BROKEN}: 远期价低于最低挂牌行权价"
        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: str(Path(tmp) / file) for name, file, _ in fetcher.DB_LAYOUT}
            calendar = ("2024-11-20", "2026-11-18", "2026-11-19", "2026-11-20")
            with patch.multiple(fetcher, DB_PATH=paths, _schema_ready=False):
                fetcher.init_db()
                fetcher.save_qvix_self_history([
                    {"date": "2024-11-20", "qvix": 12.0},
                    {"date": "2026-11-18", "qvix": 14.0},
                    {"date": "2026-11-19", "qvix": None, "note": broken},
                    {"date": "2026-11-20", "qvix": 16.0},
                ], trading_days=calendar)
                fetcher.save_qvix_self_history([
                    {"date": day, "qvix": None, "note": "接口超时"}
                    for day in ("2026-11-19", "2026-11-20")
                ], trading_days=calendar)
                fetcher.save_qvix_self_history([
                    {"date": day, "qvix": 25.0} for day in calendar
                ], trading_days=calendar, series="300")
                with closing(fetcher._conn()) as conn:
                    rows = conn.execute("SELECT qvix,note,threshold FROM qvix.qvix_self_history ORDER BY date")
                    self.assertEqual([tuple(row) for row in rows],
                                     [(12.0, None, None), (14.0, None, None),
                                      (None, broken, None), (16.0, None, 15.6)])
                hist = fetcher.load_qvix_self_history(series="300")
                self.assertEqual(hist["qvix"].tolist(), [25.0] * len(calendar))
                self.assertEqual(hist["threshold"].iloc[-1], 25.0)

    def test_calendar_year_threshold_uses_prior_close_and_calendar_coverage(self):
        days = pd.bdate_range("2022-02-28", "2024-03-01")
        history = pd.DataFrame({"date": days, "qvix": 10.0})
        history.loc[history.index[0], "qvix"] = 100.0
        history.loc[history.index[-1], "qvix"] = 99.0
        self.assertEqual(fetcher.qvix_rolling_thresholds(history, days, before=True, pct=1).iloc[-1], 100)
        self.assertEqual(fetcher.qvix_rolling_thresholds(history, days, pct=1).iloc[-1], 99)
        self.assertTrue(pd.isna(fetcher.qvix_rolling_thresholds(history, days).iloc[0]))
        incomplete = history.drop(history.index[300:340])
        self.assertTrue(pd.isna(fetcher.qvix_rolling_thresholds(incomplete, days).iloc[-1]))
