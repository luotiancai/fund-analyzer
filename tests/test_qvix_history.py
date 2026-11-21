from contextlib import closing
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetcher
import qvix_calc


class QvixHistoryTests(unittest.TestCase):
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
            with patch.multiple(fetcher, DB_PATH=paths, _schema_ready=False,
                                QVIX_THR_WINDOW=3, QVIX_THR_MINP=2):
                fetcher.init_db()
                fetcher.save_qvix_self_history([
                    {"date": "2026-11-18", "qvix": 14.0},
                    {"date": "2026-11-19", "qvix": None, "note": broken},
                    {"date": "2026-11-20", "qvix": 16.0},
                ])
                fetcher.save_qvix_self_history([
                    {"date": day, "qvix": None, "note": "接口超时"}
                    for day in ("2026-11-19", "2026-11-20")
                ])
                with closing(fetcher._conn()) as conn:
                    rows = conn.execute("SELECT qvix,note,threshold FROM qvix.qvix_self_history ORDER BY date")
                    self.assertEqual([tuple(row) for row in rows],
                                     [(14.0, None, None), (None, broken, None), (16.0, None, 15.8)])
