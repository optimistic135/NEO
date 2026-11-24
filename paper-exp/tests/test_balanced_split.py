import unittest
import numpy as np

from src.protocols.balanced_corruption import build_split, digest


class SplitTests(unittest.TestCase):
    def test_fixed_budget_and_balance(self):
        labels = np.repeat(np.arange(100), 100)
        for seed in (1234, 2020, 9999):
            a, v = build_split(labels, 100, seed, 512, 500)
            self.assertEqual(len(np.unique(np.r_[a, v])), 1012)
            np.testing.assert_array_equal(np.bincount(labels[v]), np.full(100, 5))
            self.assertEqual(np.count_nonzero(np.bincount(labels[a]) == 6), 12)
            self.assertEqual(digest(a), digest(build_split(labels, 100, seed, 512, 500)[0]))
