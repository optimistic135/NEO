import unittest

import torch
from torch import nn

from src.utils import settings
from src.methods.neo import NEO
from src.methods.plead import PLEAD
from tta_library.plead_simplified import Variant


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 10)
        self.requires_grad_(False)

    def features(self, x):
        return x

    def forward(self, x):
        return self.head(x)


class MethodTests(unittest.TestCase):
    def test_plead_200_arrivals_and_128_bank(self):
        model = Model()
        adapter = PLEAD(model, Variant())
        images = torch.randn(200, 8)
        first = adapter.update(images)
        self.assertEqual(first.shape, (200, 10))
        self.assertEqual(len(adapter.window.features), 128)
        self.assertEqual(adapter.forward_passes, 1)
        adapter.reset()
        torch.testing.assert_close(first, adapter.update(images))

    def test_neo_cumulative_center(self):
        adapter = NEO(Model())
        adapter.reset()
        a, b = torch.randn(4, 8), torch.randn(8, 8)
        adapter.update(a)
        got = adapter.update(b)
        torch.testing.assert_close(got, adapter.model.head(b-torch.cat([a,b]).mean(0)))
