import unittest

from fund_rules import is_fof, is_locked_fund


class FundRulesTests(unittest.TestCase):
    def test_periodic_and_holding_funds_are_excluded_without_matching_duration_indices(self):
        for name, expected in (
            ("红土创新科技创新3个月定开混合C", True),
            ("华安睿明两年定开混合C", True),
            ("测试定期开放混合C", True),
            ("测试一年持有混合C", True),
            ("中欧景气前瞻一年混合C", True),
            ("测试滚动持有混合C", True),
            ("测试封闭运作混合C", True),
            ("普通混合C", False),
            ("嘉实彭博国开债1-5年指数C", False),
            ("国寿安保中债1-3年指数C", False),
            (None, False),
        ):
            with self.subTest(name=name):
                self.assertEqual(is_locked_fund(name), expected)

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
