from copy import deepcopy

import torch
from torch import nn


class Adapter:
    timing = "update-before-predict"

    def __init__(self, model):
        self.model = deepcopy(model)
        self.forward_passes = self.backward_passes = 0

    def predict(self, images):
        self.forward_passes += 1
        return self.model(images)

    def features(self, images):
        self.forward_passes += 1
        return self.model.features(images)

    def reset(self):
        self.forward_passes = self.backward_passes = 0

    def update(self, images):
        return self.predict(images)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def configure_norm(model, gradient=False, skip=None):
    parameters = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = module.running_var = None
        if gradient and isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) and not (skip and skip(name)):
            module.requires_grad_(True)
            parameters.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
    return parameters


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class Optimizing(Adapter):
    timing = "predict-before-update"

    def reset(self):
        super().reset()
        for param, value in zip(self.params, self.initial):
            param.data.copy_(value)
        self.optimizer.state.clear()

    def save_initial(self):
        self.initial = [param.detach().clone() for param in self.params]

    def backward(self, loss):
        loss.backward()
        self.backward_passes += 1
