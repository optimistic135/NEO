import unittest
from unittest.mock import patch

import torch
from torch import nn

from tta_library.cap_neo import CAPConfig, CAPNEO
from tta_library.neo import NEO


class CAPNEOTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.head = nn.Linear(8, 5)
        self.features = torch.randn(11, 8)

    def test_equivalence_with_original_neo_on_each_batch_and_validation(self):
        # The original NEO selects its center device at construction time.
        model = nn.Module()
        model.classifier = self.head
        model.vit = lambda x: (x[:, None, :],)
        with patch("tta_library.neo.get_device", return_value=torch.device("cpu")):
            neo = NEO(model, 5)
        neo.imagenet_mask = None
        cap = CAPNEO(self.head, "cap_pa", CAPConfig(min_fit_samples=20))
        for batch in self.features.split(4):
            expected = neo(batch).softmax(dim=1)
            torch.testing.assert_close(cap.update(batch).softmax(dim=1), expected)
            self.assertEqual(cap.history[-1]["kappa"], 0)
            self.assertEqual(cap.history[-1]["lambda"], 1)
        torch.testing.assert_close(
            cap.predict(self.features).softmax(dim=1),
            neo(self.features, adapt=False).softmax(dim=1),
        )

    def test_shrinkage_matches_formula_and_uses_full_bank(self):
        cap = CAPNEO(self.head, "neo_shrink")
        cap.update(self.features[:1])
        self.assertEqual(cap.history[-1]["lambda"], 0)
        cap.update(self.features[1:])
        a = self.features @ cap.prototypes.T
        q_hat = a.mean(dim=0)
        n = len(a)
        variance = (a - q_hat).square().sum() / (n * (n - 1))
        shrinkage = (1 - variance / (q_hat.square().sum() + cap.config.eps)).clamp(0, 1)
        torch.testing.assert_close(cap.q, shrinkage * q_hat)
        self.assertEqual(cap.history[-1]["bank_size"], n)

    def test_validation_is_frozen_and_reset_discards_bank(self):
        cap = CAPNEO(self.head)
        cap.update(self.features[:4])
        state = cap.q.clone()
        history = list(cap.history)
        cap.predict(self.features[4:])
        torch.testing.assert_close(cap.q, state)
        self.assertEqual(cap.history, history)
        self.assertEqual(len(cap.features), 4)
        cap.reset()
        self.assertEqual(len(cap.features), 0)
        self.assertEqual(len(cap.raw_logits), 0)
        self.assertEqual(cap.history, [])
        torch.testing.assert_close(cap.predict(self.features), self.head(self.features))

    def test_previous_correction_is_used_for_bank_predictions(self):
        cap = CAPNEO(self.head, "neo_shrink")
        cap.update(self.features[:4] + 2)
        previous = cap.q.clone()
        full_bank = torch.cat((self.features[:4] + 2, self.features[4:]))
        expected_prior = (self.head(full_bank) - previous).softmax(dim=1).mean(dim=0)
        cap.update(self.features[4:])
        torch.testing.assert_close(
            torch.tensor(cap.history[-1]["predicted_prior"]), expected_prior
        )

    def test_oracle_recovers_known_scale_and_detects_single_class(self):
        cap = CAPNEO(self.head, "cap_oracle_labels", CAPConfig(min_fit_samples=4))
        labels = torch.arange(5).repeat_interleave(4)
        offset = torch.ones(8)
        cap.update(3 * cap.prototypes[labels] + offset, oracle_labels=labels)
        self.assertAlmostEqual(cap.history[-1]["kappa"], 3.0, places=5)
        torch.testing.assert_close(cap.q, cap.prototypes @ offset)
        cap.reset()
        labels = torch.zeros(4, dtype=torch.long)
        cap.update(self.features[:4], oracle_labels=labels)
        self.assertEqual(cap.history[-1]["kappa"], 0)
        self.assertIn("insufficient_class_coverage", cap.history[-1]["fallback_reasons"])

    def test_shapes_nonfinite_and_labels(self):
        cap = CAPNEO(self.head)
        with self.assertRaises(ValueError):
            cap.update(self.features[:, :2])
        with self.assertRaises(ValueError):
            cap.update(self.features * float("nan"))
        with self.assertRaises(ValueError):
            cap.update(self.features, oracle_labels=torch.zeros(len(self.features), dtype=torch.long))
        self.assertEqual(len(cap.features), 0)

    def test_low_confidence_uses_neo_residual_and_no_gradient(self):
        weight = self.head.weight.detach().clone()
        cap = CAPNEO(self.head, "cap_pa", CAPConfig(confidence=1, min_fit_samples=2))
        logits = cap.update(self.features.requires_grad_())
        self.assertFalse(logits.requires_grad)
        self.assertFalse(cap.features.requires_grad)
        self.assertIn("low_confidence", cap.history[-1]["fallback_reasons"])
        torch.testing.assert_close(cap.q, cap.prototypes @ self.features.detach().mean(dim=0))
        torch.testing.assert_close(self.head.weight, weight)


if __name__ == "__main__":
    unittest.main()
