import math

import torch

from src.methods.base import Optimizing, configure_norm, entropy


class SAR(Optimizing):
    timing = "first-pass-prediction-before-sam-update"

    def __init__(self, model, cfg, batch):
        super().__init__(model)
        self.cfg = cfg
        self.params = configure_norm(self.model, True, lambda n: any(
            part in n for part in ("layer4", "blocks.9", "blocks.10", "blocks.11",
                                   "encoder.layer.9", "encoder.layer.10", "encoder.layer.11"))
            or n.endswith(("network.norm", "network.vit.layernorm")))
        lr = cfg["resnet_lr"] if model.architecture == "resnet50" else cfg["vit_lr_per_64"] * batch / 64
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=cfg["momentum"])
        self.margin = cfg["entropy_margin_factor"] * math.log(model.head.out_features)
        self.save_initial()
        self.ema = None
        self.events = {}

    def reset(self):
        super().reset()
        self.ema = None
        self.events = {}

    @torch.enable_grad()
    def update(self, images):
        logits = self.predict(images)
        first = entropy(logits)
        reliable = first < self.margin
        self.optimizer.zero_grad(set_to_none=True)
        if not reliable.any():
            self.events = {"fallback": "empty_first_filter"}
            return logits.detach()
        self.backward(first[reliable].mean())
        with torch.no_grad():
            active = [p for p in self.params if p.grad is not None]
            norm = torch.stack([p.grad.norm() for p in active]).norm()
            old = [p.clone() for p in active]
            for p in active:
                p.add_(p.grad * (self.cfg["rho"] / (norm + 1e-12)))
        self.optimizer.zero_grad(set_to_none=True)
        second = entropy(self.predict(images))[reliable]
        second = second[second < self.margin]
        if len(second):
            loss = second.mean()
            self.backward(loss)
        with torch.no_grad():
            for p, initial in zip(active, old):
                p.copy_(initial)
        if not len(second):
            self.events = {"fallback": "empty_second_filter"}
            return logits.detach()
        self.optimizer.step()
        value = float(loss.detach())
        decay = self.cfg["recovery_momentum"]
        self.ema = value if self.ema is None else decay*self.ema + (1-decay)*value
        self.events = {}
        if self.ema < self.cfg["reset_entropy"]:
            for param, value in zip(self.params, self.initial):
                param.data.copy_(value)
            self.optimizer.state.clear()
            self.events = {"fallback": "entropy_recovery"}
        return logits.detach()
