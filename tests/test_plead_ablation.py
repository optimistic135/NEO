import unittest

import torch
from torch import nn

from cap_experiments.ablation_protocol import COMPONENTS, FULL, VARIANTS, experiments
from tta_library.plead_lite import FeatureLogitWindow, PLEADLite, describe_step


class AblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2020)
        self.head = nn.Linear(10, 7)
        self.features = torch.randn(64, 10)

    def run_variant(self, variant):
        window = FeatureLogitWindow(self.head, variant.window)
        model = PLEADLite(window, variant.fit(), variant.components)
        raw = window.append(self.features, allow_oversize=True)
        logits, diag, audit = model.update(window, len(raw), batch_logits=raw, audit=True)
        return logits, describe_step(diag.tolist()), audit, window

    def test_full_matches_existing_update_path(self):
        window = FeatureLogitWindow(self.head, 128)
        window.append(self.features)
        old_logits, old_diag = PLEADLite(window).update(window, len(self.features))
        logits, stats, _, _ = self.run_variant(FULL)
        torch.testing.assert_close(logits, old_logits, rtol=0, atol=0)
        self.assertEqual(stats, describe_step(old_diag.tolist()))

    def test_ablation_interventions_and_finite_predictions(self):
        for name, variant in COMPONENTS.items():
            output, stats, audit, window = self.run_variant(variant)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(audit).all())
            if name == "plead_no_debias":
                self.assertEqual(stats["gamma"], 0)
                self.assertEqual(audit[1].item(), 0.0)
                u = window.projected
                q = u.mean(0)
                variance = (u - q).square().sum() / (len(u) * (len(u)-1))
                expected = (1 - variance / (q.square().sum() + variant.fit().eps)).clamp(0, 1)
                self.assertAlmostEqual(stats["lambda"], expected.item(), places=6)
            if name == "plead_no_shrink":
                self.assertEqual(stats["lambda"], 1)
            if name == "plead_no_weight":
                self.assertAlmostEqual(stats["neff"], len(self.features), places=5)
            if name == "plead_no_gate" and not stats["fit_code"]:
                self.assertEqual(stats["gamma"], 1)
            if name == "plead_no_ridge":
                self.assertEqual(variant.fit().ridge_fraction, 0)

    def test_no_gate_respects_failed_kappa(self):
        self.features[:] = 1
        for name in ("plead_no_gate", "plead_no_ridge"):
            output, step, audit, _ = self.run_variant(VARIANTS[name])
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(step["kappa"], 0)
            self.assertEqual(step["gamma"], 0)
            self.assertEqual(audit[1].item(), 0.0)

    def test_window_smaller_than_batch_keeps_score_for_every_arrival(self):
        logits, stats, _, window = self.run_variant(VARIANTS["window_32"])
        self.assertEqual(logits.shape, (64, 7))
        self.assertEqual(stats["bank_size"], 32)
        torch.testing.assert_close(window.features, self.features[-32:])

    def test_sensitivity_shares_the_full_configuration(self):
        groups = experiments("imagenet-c", 5000)
        self.assertEqual(set(groups), {"online_ir", "estimation", "sensitivity", "mechanism"})
        self.assertTrue(all("plead" in names for names in groups.values()))
        self.assertNotIn("sensitivity", experiments("cifar10-c", 50))
