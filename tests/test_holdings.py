import json
import sqlite3
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import requests

import fetcher


def holdings_html(kind, name="测试标的"):
    code_col, name_col, code = (
        ("股票代码", "股票名称", "000001") if kind == "股票"
        else ("债券代码", "债券名称", "010107")
    )
    shares = "<th>持股数 （万股）</th>" if kind == "股票" else ""
    count = "<td>12.50</td>" if kind == "股票" else ""
    return (
        f'<h4 class="t"><label>测试基金&nbsp;&nbsp;2025年4季度{kind}投资明细'
        '</label><label>&nbsp;&nbsp;来源：天天基金</label></h4>'
        f'<table><thead><tr><th>{code_col}</th><th>{name_col}</th>'
        f'<th>占净值 比例</th>{shares}<th>持仓市值 （万元）</th></tr></thead>'
        f'<tbody><tr><td>{code}</td><td>{name}</td><td>4.57%</td>'
        f'{count}<td>1,234.56</td></tr></tbody></table>'
    )


def response(content, status=200):
    result = requests.Response()
    result.status_code = status
    result.encoding = "utf-8"
    result._content = (
        f'var apidata={{ content:{json.dumps(content)},'
        'arryear:[2025],curyear:2025};\n'
    ).encode("utf-8")
    return result


class HoldingsTests(unittest.TestCase):
    def setUp(self):
        # Reproduce deployment environments without the old AKShare module.
        blocker = patch.dict(sys.modules, {"akshare": None, "akshare.utils": None})
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_stock_and_bond_holdings_without_akshare(self):
        with patch.object(fetcher.requests, "get", side_effect=[
            response(holdings_html("股票")), response(holdings_html("债券")),
        ]):
            result = fetcher._fetch_holdings_year("000001", "2025")

        self.assertEqual(list(result.columns), fetcher._HOLDINGS_COLS)
        self.assertEqual(result["quarter"].tolist(), ["2025Q4", "2025Q4"])
        self.assertEqual(result["kind"].tolist(), ["股票", "债券"])
        self.assertEqual(result["代码"].tolist(), ["000001", "010107"])
        self.assertEqual(result["占净值比例"].tolist(), [4.57, 4.57])
        self.assertEqual(result["持仓市值"].tolist(), [1234.56, 1234.56])
        self.assertEqual(result.iloc[0]["持股数"], 12.5)
        self.assertTrue(pd.isna(result.iloc[1]["持股数"]))

    def test_escaped_content(self):
        name = '测试"标的\\路径'
        with patch.object(fetcher.requests, "get", return_value=response(
            holdings_html("股票", name))):
            result = fetcher._em_f10_holdings_raw("000001", "2025", "jjcc")
        self.assertEqual(result.iloc[0]["股票名称"], name)

    def test_no_disclosures_is_empty(self):
        with patch.object(fetcher.requests, "get", return_value=response("")):
            result = fetcher._fetch_holdings_year("000198", "2025")
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), fetcher._HOLDINGS_COLS)

    def test_invalid_responses_are_fetch_failures(self):
        for body in [
            '<html>Not Found</html>',
            'var apidata={arryear:[2025]};',
            'var apidata={content:"unterminated};',
            'var apidata={content:null};',
        ]:
            with self.subTest(body=body):
                result = response("")
                result._content = body.encode("utf-8")
                with patch.object(fetcher.requests, "get", return_value=result):
                    self.assertIsNone(fetcher._em_f10_holdings_raw(
                        "000001", "2025", "jjcc"))

    def test_http_errors_and_timeouts_are_fetch_failures(self):
        with patch.object(fetcher.requests, "get", return_value=response("", 503)):
            self.assertIsNone(fetcher._em_f10_holdings_raw("000001", "2025", "jjcc"))
        with patch.object(fetcher.requests, "get", side_effect=requests.Timeout):
            self.assertIsNone(fetcher._em_f10_holdings_raw("000001", "2025", "jjcc"))

    def test_failed_refresh_keeps_stale_cache(self):
        year = datetime.now().year
        cached = pd.DataFrame([{
            "quarter": f"{year}Q1", "kind": "股票", "代码": "000001",
            "名称": "缓存标的", "占净值比例": 4.57, "持股数": 12.5,
            "持仓市值": 1234.56,
        }], columns=fetcher._HOLDINGS_COLS)
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE fund_holdings "
                     "(code TEXT, year TEXT, data TEXT, saved_at REAL)")
        conn.execute("INSERT INTO fund_holdings VALUES (?, ?, ?, ?)",
                     ("000001", str(year), cached.to_json(orient="records"), 0))
        with patch.object(fetcher, "_conn", return_value=conn), \
                patch.object(fetcher, "resolve_target_etf", return_value=None), \
                patch.object(fetcher, "HOLDINGS_START_YEAR", year), \
                patch.object(fetcher.requests, "get", return_value=response("", 503)):
            result = fetcher.fetch_holdings("000001")
        pd.testing.assert_frame_equal(result, cached)


if __name__ == "__main__":
    unittest.main()
