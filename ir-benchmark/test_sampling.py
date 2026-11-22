import unittest

import numpy as np
from sampling import CapacityError, DESIGN, build_stream


class Streams(unittest.TestCase):
    def test_cifar_coverage_and_reproducibility(self):
        for dataset in ("cifar10-c", "cifar100-c"):
            k, size, ratios = DESIGN[dataset]
            labels = np.repeat(np.arange(k), size)
            for ir in ratios:
                stream, capped = build_stream(labels, dataset, ir, 2020)
                self.assertFalse(capped)
                np.testing.assert_array_equal(np.sort(stream.indices.ravel()), np.arange(k * size))
                self.assertEqual(stream.sha256, build_stream(labels, dataset, ir, 2020)[0].sha256)
                self.assertAlmostEqual(stream.p_max / stream.p_min, ir)
                self.assertEqual(sum(map(len, stream.batches(64))), size * k)

    def test_imagenet_no_repeat_inside_subset(self):
        labels = np.repeat(np.arange(1000), 50)
        for ir in (1, 1000, 5000, 500000):
            stream, capped = build_stream(labels, "imagenet-c", ir, 1234, "cap")
            self.assertEqual(capped, ir >= 5000)
            self.assertEqual(stream.indices.shape, (1000, 100))
            self.assertTrue(np.all(np.diff(np.sort(stream.indices), axis=1) > 0))
            if ir == 1:
                np.testing.assert_array_equal(stream.counts.sum(0), np.full(1000, 100))
                covered = stream.counts[np.arange(1000), stream.dominant_classes].sum()
                self.assertGreater(covered, 0)
            for images, counts in zip(stream.indices, stream.counts):
                np.testing.assert_array_equal(np.bincount(labels[images], minlength=1000), counts)
            self.assertEqual(list(map(len, stream.batches(64)))[:4], [64, 36, 64, 36])

    def test_strict_infeasibility(self):
        labels = np.repeat(np.arange(1000), 50)
        for ir in (5000, 500000):
            with self.assertRaises(CapacityError):
                build_stream(labels, "imagenet-c", ir, 1234)
