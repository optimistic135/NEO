import unittest

import numpy as np

from protocol import split, split_hash


class SplitTest(unittest.TestCase):
    def test_all_dataset_shapes_and_seeds(self):
        for classes, per_class in ((10, 1000), (100, 100), (1000, 50)):
            targets = np.repeat(np.arange(classes), per_class)
            for seed in (1234, 2020, 9999):
                with self.subTest(classes=classes, seed=seed):
                    adapt, validation = split(targets, seed)
                    self.assertEqual(len(adapt), 512)
                    self.assertEqual(len(np.unique(adapt)), 512)
                    np.testing.assert_array_equal(np.sort(np.r_[adapt, validation]), np.arange(len(targets)))
                    counts = np.bincount(targets[adapt], minlength=classes)
                    self.assertLessEqual(counts.max() - counts.min(), 1)
                    self.assertEqual(split_hash(adapt), split_hash(split(targets, seed)[0]))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            split(np.repeat([0, 1], 200), 2020)


if __name__ == "__main__":
    unittest.main()
