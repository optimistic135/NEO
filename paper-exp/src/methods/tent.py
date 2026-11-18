import torch

from src.methods.base import Optimizing, configure_norm, entropy


class TENT(Optimizing):
    def __init__(self, model, cfg, imagenet):
        super().__init__(model)
        self.params = configure_norm(self.model, True)
        self.optimizer = (torch.optim.SGD(self.params, lr=cfg["imagenet_lr"], momentum=cfg["momentum"])
                          if imagenet else torch.optim.Adam(self.params, lr=cfg["cifar_lr"]))
        self.save_initial()

    @torch.enable_grad()
    def update(self, images):
        logits = self.predict(images)
        self.optimizer.zero_grad(set_to_none=True)
        self.backward(entropy(logits).mean())
        self.optimizer.step()
        return logits.detach()
