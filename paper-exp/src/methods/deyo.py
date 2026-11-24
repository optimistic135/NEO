import math
from types import SimpleNamespace

import torch

from src.methods.base import Optimizing, configure_norm
from src.utils.official import module


def top_norm(name):
    return any(f"encoder.layer.{i}." in name for i in (9, 10, 11)) or name == "network.vit.layernorm"


class DeYO(Optimizing):
    timing = "predict-before-update"

    def __init__(self, model, cfg):
        super().__init__(model)
        self.params = configure_norm(self.model, gradient=True, skip=top_norm)
        self.optimizer = torch.optim.SGD(self.params, lr=cfg["lr"], momentum=cfg["momentum"])
        self.save_initial()
        implementation = module(cfg, "methods/deyo.py")
        args = SimpleNamespace(**cfg["selection"], wandb_log=False)
        scale = math.log(model.head.out_features)
        self.adapt = implementation.DeYO(self.model, args, self.optimizer,
                                         deyo_margin=cfg["filter_margin"] * scale,
                                         margin_e0=cfg["weight_margin"] * scale)
        self.events = {}

    def update(self, images):
        logits, first, final = self.adapt(images, self.forward_passes, targets=None)
        self.forward_passes += 1 + bool(first)
        self.backward_passes += bool(final)
        self.events = {"entropy_count": first, "plpd_count": final}
        return logits.detach()
