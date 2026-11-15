import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from cap_experiments.ir_evaluation import ArrivalFeatures, Timings, evaluate_online
from cap_experiments.online_ir import SPECS, make_stream
from tta_library.plead_lite import FeatureLogitWindow, NEOOnline, PLEADLite, describe_step
from tta_library.plead_v2 import PLEADV2, V2Config


class SamplingTest(unittest.TestCase):
    def test_integer_quotas_all_datasets_and_irs(self):
        for classes, size, batch_size, irs in SPECS.values():
            rng = np.random.default_rng(2020)
            labels = rng.permutation(np.repeat(np.arange(classes), size))
            for ir in irs:
                with self.subTest(classes=classes, ir=ir):
                    stream = make_stream(labels, classes, size, ir, 2020)
                    self.assertEqual(stream.sha256, make_stream(labels, classes, size, ir, 2020).sha256)
                    np.testing.assert_array_equal(np.sort(stream.indices.flatten()), np.arange(len(labels)))
                    np.testing.assert_array_equal(stream.counts.sum(axis=0), np.full(classes, size))
                    self.assertEqual(len(set(stream.dominant_classes)), classes)
                    for position, subset in enumerate(stream.indices):
                        counts = np.bincount(labels[subset], minlength=classes)
                        np.testing.assert_array_equal(counts, stream.counts[position])
                        self.assertLessEqual(abs(counts[stream.dominant_classes[position]] - size * stream.p_max), .5 + 1e-10)
                    self.assertEqual(sum(map(len, stream.batches(batch_size))), len(labels))
                    self.assertEqual(len(list(stream.batches(batch_size))), classes * ((size + batch_size - 1) // batch_size))

    def test_capacity_error(self):
        with self.assertRaises(ValueError):
            make_stream(np.array([0, 1, 0]), 2, 2, 20, 2020)


class CacheTest(unittest.TestCase):
    def test_fifo_neo_and_fit_gate_parity(self):
        torch.manual_seed(2020)
        classifier = nn.Linear(16, 10)
        cfg = V2Config(max_bank=50)
        window = FeatureLogitWindow(classifier, 50)
        neo, lite = NEOOnline(), PLEADLite(window, cfg)
        v2 = PLEADV2(classifier, "plead_v2_fit_gate", cfg)
        seen = torch.empty(0, 16)
        for n in (16, 13, 37, 5, 50, 31):
            images = torch.randn(n, 16, requires_grad=True)
            seen = torch.cat((seen, images.detach()))[-50:]
            window.append(images)
            torch.testing.assert_close(window.features, seen)
            self.assertFalse(window.features.requires_grad)
            torch.testing.assert_close(neo.update(window, n), classifier(images - seen.mean(0)))
            logits, diag = lite.update(window, n)
            torch.testing.assert_close(logits, v2.update(images), atol=3e-4, rtol=5e-5)
            step = describe_step(diag.tolist())
            self.assertEqual(step["fallback_reasons"], v2.history[-1]["fallback_reasons"])
            self.assertEqual(step["bank_size"], len(seen))
            for name in ("kappa", "gamma", "lambda"):
                self.assertAlmostEqual(step[name], v2.history[-1][name], places=4)
            self.assertFalse(logits.requires_grad)

    def test_zero_energy_and_singleton(self):
        classifier = nn.Linear(3, 4, bias=False)
        classifier.weight.data.copy_(torch.tensor([[1., 0, 0], [0., 0, 0], [-1., 0, 0], [-2., 0, 0]]))
        window = FeatureLogitWindow(classifier, 64)
        lite = PLEADLite(window)
        repeated = torch.ones(1, 3).repeat(64, 1)
        window.append(repeated[:1])
        logits, diag = lite.update(window, 1)
        self.assertIn("singleton_bank", describe_step(diag.tolist())["fallback_reasons"])
        self.assertTrue(torch.isfinite(logits).all())
        window.append(repeated)
        logits, diag = lite.update(window, 64)
        step = describe_step(diag.tolist())
        self.assertIn("degenerate_sxx", step["fallback_reasons"])
        self.assertEqual(step["kappa"], 0)
        self.assertEqual(step["gamma"], 0)
        # Centering classifier rows leaves a common logit offset relative to NEO.
        torch.testing.assert_close(logits.softmax(1), NEOOnline().update(window, 64).softmax(1))

    def test_arrival_cache_never_reads_a_future_id(self):
        calls = []

        def extract(batch):
            calls.extend(batch.flatten().tolist())
            return batch

        images = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(images, images.long().reshape(-1))
        device = torch.device("cpu")
        cache = ArrivalFeatures(dataset, extract, nn.Linear(1, 5), device)
        clock = Timings(device)
        features, n = cache.arrive(np.array([3, 1]), clock)
        self.assertEqual(calls, [3, 1])
        self.assertEqual(n, 2)
        features, n = cache.arrive(np.array([1, 8]), clock)
        self.assertEqual(calls, [3, 1, 8])
        self.assertEqual(features.flatten().tolist(), [1, 8])
        self.assertEqual(n, 1)

    def test_online_window_survives_subset_boundary_and_uses_one_shared_forward(self):
        classes, size = 5, 6
        labels = np.repeat(np.arange(classes), size)
        stream = make_stream(labels, classes, size, 20, 2020)
        classifier = nn.Linear(3, classes)
        dataset = TensorDataset(torch.rand(len(labels), 3), torch.from_numpy(labels))
        calls = []

        def extract(batch):
            calls.append(len(batch))
            return batch

        device = torch.device("cpu")
        cache = ArrivalFeatures(dataset, extract, classifier, device)
        args = SimpleNamespace(device=device, smoke=True, smoke_subsets=2, batch_size=4)
        rows, subsets, steps = evaluate_online(stream, labels, cache, classifier, args, V2Config(), "test")
        self.assertEqual(calls, [4, 2, 4, 2])
        self.assertEqual([s["bank_size"] for s in steps], [4, 6, 6, 6])
        self.assertEqual(len(subsets), 2)
        for row in rows:
            self.assertEqual(row["samples"], 12)
            self.assertEqual(row["backbone_forward_count"], 4)
