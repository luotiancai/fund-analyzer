import unittest

from fund_rules import is_fof


class FundRulesTests(unittest.TestCase):
    def test_fof_identification_preserves_ordinary_and_feeder_funds(self):
        for fund_type, name, expected in (
            ("FOF-稳健型", "嘉实领航资产配置混合C", True),
            ("混合型", "兴全积极配置混合(FOF-LOF)C", True),
            (None, "基金中基金C", True),
            ("fof", None, True),
            ("指数型-股票", "普通ETF联接C", False),
            ("混合型", "普通混合(LOF)C", False),
            (None, None, False),
        ):
            with self.subTest(fund_type=fund_type, name=name):
                self.assertEqual(is_fof(fund_type, name), expected)
