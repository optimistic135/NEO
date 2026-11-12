import unittest

import numpy as np

from cap_experiments.protocol import controlled_stream, manifest, natural_stream


class CIFARProtocolTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(10), 1000)
        self.ids = list(range(len(self.labels)))

    def test_natural_partition_is_complete_and_repeatable(self):
        stream = natural_stream(10_000, 512, 2020)
        self.assertEqual(stream, natural_stream(10_000, 512, 2020))
        self.assertNotEqual(stream, natural_stream(10_000, 512, 9999))
        splits = manifest(stream, self.labels, self.ids)
        self.assertEqual(len(splits["adaptation"]["indices"]), 512)
        self.assertEqual(len(splits["validation"]["indices"]), 9488)

    def test_controlled_streams_hold_validation_fixed_and_disjoint_from_adaptation(self):
        balanced = None
        for ratio in (10, 30, 50, 70, 90, 100):
            stream = controlled_stream(self.labels, 512, 480, 1234, ratio)
            self.assertEqual(stream, controlled_stream(self.labels, 512, 480, 1234, ratio))
            splits = manifest(stream, self.labels, self.ids)
            self.assertEqual(len(splits["adaptation"]["indices"]), 512)
            self.assertAlmostEqual(
                splits["adaptation"]["label_counts"][stream.dominant_class] / 512,
                ratio / 100,
                delta=1 / 512,
            )
            self.assertEqual(splits["balanced_validation"]["label_counts"], [48] * 10)
            self.assertEqual(len(splits["matched_validation"]["indices"]), 480)
            self.assertEqual(
                splits["matched_validation"]["label_counts"][stream.dominant_class], 480 * ratio // 100
            )
            if balanced is None:
                balanced = stream.validation["balanced_validation"]
                self.assertEqual(balanced, stream.validation["matched_validation"])
            self.assertEqual(stream.validation["balanced_validation"], balanced)


if __name__ == "__main__":
    unittest.main()
