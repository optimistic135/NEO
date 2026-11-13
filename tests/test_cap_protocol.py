import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from calibration_library.metrics import ECELoss
from cap_experiments.evaluation import Metrics
from cap_experiments.protocol import (
    controlled_imagenet_stream, controlled_stream, imagenet_directory, manifest, natural_stream,
)


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


class ImageNetProtocolTest(unittest.TestCase):
    def test_grouped_directories_include_existing_bur_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for entry in ("noise/gaussian_noise/5", "bur/motion_blur/5", "snow/5"):
                path = root / entry
                path.mkdir(parents=True)
                self.assertEqual(imagenet_directory(root, path.parent.name, 5), path)
            with self.assertRaises(ValueError):
                imagenet_directory(root, "zoom_blur", 5)

    def test_fixed_holdout_and_single_class_resampling(self):
        labels = np.repeat(np.arange(1000), 50)
        ids = list(range(len(labels)))
        uniform = controlled_imagenet_stream(labels, 512, 1000, 2020, 0)
        self.assertEqual(len(set(uniform.adaptation)), 512)
        self.assertEqual(len(set(labels[uniform.adaptation])), 512)
        for ratio in (10, 30, 50, 70, 90, 100):
            stream = controlled_imagenet_stream(labels, 512, 1000, 2020, ratio)
            splits = manifest(stream, labels, ids, 1000)
            self.assertEqual(len(stream.adaptation), 512)
            self.assertEqual(stream.validation, uniform.validation)
            self.assertEqual(splits["balanced_validation"]["label_counts"], [1] * 1000)
            self.assertEqual(splits["adaptation"]["label_counts"][20], (512 * ratio + 50) // 100)
            self.assertAlmostEqual(splits["matched_validation"]["target_prior"][20], ratio / 100)
        self.assertEqual(len(set(stream.adaptation)), 49)

    def test_weighted_metrics_and_unweighted_ece(self):
        logits = torch.tensor([[3., 0., 0., 0., 0.], [3., 0., 0., 0., 0.]])
        targets = torch.tensor([0, 1])
        unweighted = Metrics()
        unweighted.add(logits, targets)
        self.assertAlmostEqual(
            unweighted.result()["ece"], ECELoss().loss(logits.numpy(), targets.numpy())
        )
        weighted = Metrics(np.array([0.9, 0.1]))
        for batch in range(2):
            weighted.add(logits[batch:batch + 1], targets[batch:batch + 1])
        self.assertAlmostEqual(weighted.result()["top1"], 90)
        self.assertAlmostEqual(weighted.result()["ece"], abs(logits.softmax(1)[0, 0].item() - 0.9))


if __name__ == "__main__":
    unittest.main()
