import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from cap_experiments.evaluation import evaluate_stream
from cap_experiments.longtail import longtail_prior, longtail_stream
from cap_experiments.protocol import Stream, manifest
from tta_library.cap_neo import CAPConfig, CAPNEO
from tta_library.neo import NEO
from tta_library.plead_v2 import PLEADV2, V2Config, weighted_ridge


class PLEADV2Test(unittest.TestCase):
    def test_ridge_without_class_ids_and_zero_energy(self):
        # One distinct, reliable prototype signal per example is sufficient.
        t = torch.eye(32)
        kappa, reason, stats = weighted_ridge(t, 2 * t + 5, torch.ones(32), V2Config())
        self.assertIsNone(reason)
        self.assertAlmostEqual(kappa, 2 / 1.01, places=6)
        self.assertEqual(stats["neff"], 32)
        kappa, reason, _ = weighted_ridge(torch.ones_like(t), t, torch.ones(32), V2Config())
        self.assertEqual((kappa, reason), (0.0, "degenerate_sxx"))

    def test_gate_zero_removes_debias_and_uses_arrived_samples_only(self):
        classifier = nn.Linear(20, 20, bias=False)
        classifier.weight.data.copy_(torch.eye(20))
        gate = PLEADV2(classifier, "plead_v2_fit_gate")
        fit = PLEADV2(classifier, "plead_v2_fit")
        shrink = CAPNEO(classifier, "neo_shrink")
        x = (3 * torch.eye(20)).requires_grad_()
        actual = gate.update(x)
        fit.update(x)
        self.assertFalse(actual.requires_grad)
        self.assertGreater(gate.history[-1]["kappa"], 0)
        self.assertEqual(gate.history[-1]["gamma"], 0)
        self.assertEqual(fit.history[-1]["gamma"], 1)
        self.assertEqual(gate.history[-1]["bank_size"], 20)
        self.assertIn("insufficient_neff", gate.history[-1]["gate_fallback_reason"])
        torch.testing.assert_close(actual, shrink.update(x))
        q = gate.q.clone()
        gate.predict(x * 100)
        torch.testing.assert_close(q, gate.q)
        self.assertEqual(gate.history[-1]["bank_size"], 20)
        gate.update(x.repeat(26, 1))
        self.assertEqual(gate.history[-1]["bank_size"], 512)

    def test_cross_view_forward_counts(self):
        model = nn.Module()
        model.head = nn.Linear(6, 5)
        neo = NEO(model, 5)
        neo.imagenet_mask = None
        neo.feature_extractor = nn.Identity()
        args = SimpleNamespace(device=torch.device("cpu"), workers=0, batch_size=10,
                               no_progress=True, check_equivalence=False)
        dataset = TensorDataset(torch.rand(50, 2, 3), torch.arange(50) % 5)
        stream = Stream("test", None, None, list(range(40)), {"balanced_validation": list(range(40, 50))})
        calls = []

        def extractor(images):
            calls.append(images.clone())
            return images.flatten(1)

        for methods, expected in ((("plead_v2_fit", "plead_v2_fit_gate"), 5),
                                  (("plead_v2_fit_cv", "plead_v2"), 9)):
            calls.clear()
            rows, _, _ = evaluate_stream(dataset, stream, extractor, neo, methods, CAPConfig(),
                                         args, 2020, "test", v2_config=V2Config())
            self.assertEqual(len(calls), expected)
            if expected == 9:
                torch.testing.assert_close(calls[0].flip(-1), calls[1])
            for row in rows:
                self.assertEqual(row["cross_view"], expected == 9)
                self.assertEqual(row["gamma_gate"], row["method"] in ("plead_v2_fit_gate", "plead_v2"))
                self.assertGreaterEqual(row["method_seconds"], row["cross_view_seconds"])


class LongtailTest(unittest.TestCase):
    def test_probabilities_fixed_order_and_disjoint_views(self):
        classes = 1000
        labels = np.repeat(np.arange(classes), 50)
        previous_order, previous_balanced = None, None
        for rho in (1, 10, 100):
            stream, audit = longtail_stream(labels, classes, 2020, rho, 512, 1)
            order = audit["class_order"]
            prior = longtail_prior(order, rho)
            np.testing.assert_allclose(prior.sum(), 1)
            self.assertAlmostEqual(prior[order[0]] / prior[order[-1]], rho)
            self.assertEqual(stream, longtail_stream(labels, classes, 2020, rho, 512, 1)[0])
            splits = manifest(stream, labels, list(range(len(labels))), classes)
            self.assertEqual(len(set(stream.adaptation)), 512)
            self.assertGreater(splits["adaptation"]["label_counts"].count(0), 0)
            balanced = stream.validation["balanced_validation"]
            self.assertFalse(set(balanced).intersection(stream.validation["matched_validation"]))
            if previous_order is not None:
                self.assertEqual(order, previous_order)
                self.assertEqual(balanced, previous_balanced)
            previous_order, previous_balanced = order, balanced
