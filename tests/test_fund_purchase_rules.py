import datetime as dt
import unittest

from fund_purchase_rules import purchase_rules, restricted_codes_on


class PurchaseRulesTests(unittest.TestCase):
    def test_north_funds_are_excluded_on_the_two_affected_buy_dates(self):
        for date in ("2024-09-26", "2024-10-11"):
            with self.subTest(date=date):
                self.assertTrue({"017513", "017516"} <= restricted_codes_on(date))

    def test_new_limits_only_affect_dates_after_they_take_effect(self):
        self.assertNotIn("017518", restricted_codes_on("2024-09-26"))
        self.assertIn("017518", restricted_codes_on("2024-10-11"))

    def test_recovered_funds_are_not_permanently_blacklisted(self):
        self.assertIn("017520", restricted_codes_on("2025-04-07"))
        self.assertNotIn("017520", restricted_codes_on("2026-03-23"))
        self.assertNotIn("016387", restricted_codes_on("2024-10-11"))

    def test_restricted_snapshots_have_effective_announcement_evidence(self):
        rules = purchase_rules()
        for date, funds in rules["by_date"].items():
            dt.date.fromisoformat(date)
            for entry in funds.values():
                for source in entry["scopes"].values():
                    evidence = rules["sources"][source]
                    self.assertLessEqual(evidence["effective_date"], date)
                    self.assertTrue(evidence["url"].endswith(source + "_1.pdf"))

    def test_unreviewed_dates_do_not_silently_produce_an_unrestricted_pool(self):
        with self.assertRaisesRegex(ValueError, "历史限购尚未核实"):
            restricted_codes_on("2099-01-01")
