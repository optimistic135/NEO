import unittest

import torch

from cap_experiments.simplified_protocol import VARIANTS
from tta_library.plead_lite import FeatureLogitWindow
from tta_library.plead_simplified import FIELDS, PLEAD, prior_evidence


class SimplifiedTest(unittest.TestCase):
    def setUp(self):
        self.head = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(3))
        self.features = torch.tensor([[5., 1, 0], [2., 5, 1], [4., 2, 0], [1., 4, 1]])

    def test_evidence_equation_and_exact_uniform_limit(self):
        stable = torch.tensor([[.98, .01, .01]]).repeat(4, 1)
        uniform = torch.full((4, 2), .5)
        noisy = torch.tensor([[1., 0], [0., 1]]).repeat(2, 1)
        self.assertEqual(prior_evidence(stable)[-1].item(), 1)
        self.assertEqual(prior_evidence(uniform)[-1].item(), 1)
        self.assertEqual(prior_evidence(noisy)[-1].item(), 0)

    def test_fit_and_all_variants(self):
        for name, config in VARIANTS.items():
            with self.subTest(name=name):
                window = FeatureLogitWindow(self.head, config.window)
                raw = window.append(self.features)
                model = PLEAD(window, config)
                prediction, values, prior, _ = model.update(window, raw)
                d = dict(zip(FIELDS, values))
                self.assertEqual(d["fallback_code"], 0)
                expected = ((d["suu"] / d["sxx"]).sqrt() if config.norm_only else
                            (d["sxy"] / d["sxx"]).clamp_min(0))
                torch.testing.assert_close(d["kappa"], expected)
                gamma = d["gamma"]
                if config.fixed_gamma is not None:
                    self.assertEqual(gamma.item(), config.fixed_gamma)
                b = d["kappa"] * ((prior @ window.centered) @ window.centered.T)
                torch.testing.assert_close(prediction, raw - (window.projected.mean(0) - gamma*b))
                self.assertFalse(prediction.requires_grad)

    def test_fallback_retains_state_and_batch_variant_drops_history(self):
        for name in ("plead", "current_batch_only"):
            config = VARIANTS[name]
            window = FeatureLogitWindow(self.head, 4 if name == "plead" else 128)
            model = PLEAD(window, config)
            model.update(window, window.append(self.features))
            q = model.q.clone()
            raw = window.append(self.features[:1].repeat(4, 1))
            pred, d, _, _ = model.update(window, raw)
            self.assertEqual(d[0], 4)
            self.assertEqual(d[-1], 3)
            torch.testing.assert_close(pred, raw - q)
            torch.testing.assert_close(model.q, q)

    def test_no_future_features_and_independent_reset(self):
        window = FeatureLogitWindow(self.head, 128)
        model = PLEAD(window)
        raw = window.append(self.features[:2])
        result, diag, _, _ = model.update(window, raw)
        self.assertEqual(diag[-1], 1)
        torch.testing.assert_close(result, raw)
        model.update(window, window.append(self.features[2:]))
        self.assertGreater(model.q.norm().item(), 0)
        self.assertEqual(PLEAD(window).q.norm().item(), 0)
