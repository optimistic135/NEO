import unittest

import numpy as np

from src.utils import settings
from src.data.streams import balanced
from src.data.dart import load_manifest
from src.engine.runner import batches


class ProtocolTests(unittest.TestCase):
    def test_balanced_reproducible_full_permutation(self):
        a = balanced(10000, 137)
        np.testing.assert_equal(a, balanced(10000, 137))
        np.testing.assert_equal(np.sort(a), np.arange(10000))
        self.assertFalse(np.array_equal(a, balanced(10000, 2029)))

    def test_online_keeps_subset_boundary(self):
        self.assertEqual(list(map(len, batches(np.arange(200).reshape(2, 100), 64))), [64, 36, 64, 36])

    def test_unverified_online_protocol_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            load_manifest(None, "imagenet-c", 137, 500000, [0], 100, 1000)
