import unittest

from src.engine.online_report import HASHES, complete_scores, matches


class PairingTests(unittest.TestCase):
    def test_requires_all_four_hashes(self):
        row = {k: k for k in HASHES}
        self.assertTrue(matches(row, row))
        for key in HASHES:
            self.assertFalse(matches(row, {**row, key: "other"}))
        self.assertFalse(matches({}, {}))

    def test_only_complete_coverage_yields_mean(self):
        rows = {("noise", 1): {"top1": 8}, ("blur", 1): {"top1": 6}}
        self.assertIsNone(complete_scores(rows, ["noise", "blur"], [1, 2]))
        self.assertEqual(complete_scores(rows, ["noise", "blur"], [1]), [7])
