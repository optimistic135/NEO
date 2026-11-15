import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from cap_experiments.protocol import natural_stream
from run_neo_paper import create_adapters, evaluation_phases
from tta_library.neo import NEO


class NEOPaperProtocolTest(unittest.TestCase):
    def test_scoring_population(self):
        stream = natural_stream(700, 512, 2020)
        for protocol in ("table2", "continual"):
            self.assertEqual(evaluation_phases(protocol, stream), [("adaptation", stream.adaptation, True)])
        _, (name, validation, adapt) = evaluation_phases("heldout-512", stream)
        self.assertEqual(name, "heldout")
        self.assertFalse(adapt)
        self.assertFalse(set(validation).intersection(stream.adaptation))
        self.assertEqual(len(validation), 188)

    @torch.no_grad()
    def test_original_neo_counts_all_512_and_resets(self):
        args = SimpleNamespace(protocol="table2")
        head = nn.Linear(4, 7)
        model = nn.Module()
        model.head = head
        device = torch.device("cpu")
        with patch("tta_library.neo.get_device", return_value=device):
            neo = NEO(model, 7)
            neo.imagenet_mask = None
            neo.feature_extractor = nn.Identity()
            original, bank, _ = create_adapters(neo, args)
            samples = torch.cat([torch.full((64, 4), float(i)) for i in range(8)])
            for batch in samples.split(64):
                original(batch)
                bank.append(batch)
            self.assertEqual(original.adapt_sample_count, 512)
            self.assertEqual(len(bank.features), 512)
            torch.testing.assert_close(original.corrupt_class_center, samples.mean(0))
            original, bank, _ = create_adapters(neo, args)
            self.assertEqual(original.adapt_sample_count, 0)
            self.assertEqual(len(bank.features), 0)

    @torch.no_grad()
    def test_continual_uses_author_ema_across_boundaries(self):
        args = SimpleNamespace(protocol="continual", alpha=0.1)
        model = nn.Module()
        model.head = nn.Linear(4, 7)
        device = torch.device("cpu")
        with patch("tta_library.neo.get_device", return_value=device), patch(
            "tta_library.neo_cont.get_device", return_value=device,
        ):
            neo = NEO(model, 7)
            original, _, _ = create_adapters(neo, args)
            original(torch.ones(64, 4))
            original(torch.full((64, 4), 2.0))
            torch.testing.assert_close(original.corrupt_class_center, torch.full((4,), 0.29))
