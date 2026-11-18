import unittest

import torch
from torchvision.models import resnet50

from src.utils import settings
from src.models.loading import Classifier


class ModelTests(unittest.TestCase):
    def test_resnet_features_reconstruct_native_forward(self):
        model = resnet50(weights=None).eval()
        wrapped = Classifier(model, "resnet50", "fc")
        with torch.no_grad():
            images = torch.randn(1, 3, 64, 64)
            torch.testing.assert_close(wrapped(images), model(images))
