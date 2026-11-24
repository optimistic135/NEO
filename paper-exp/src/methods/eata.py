import math
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from src.methods.base import Adapter, configure_norm
from src.utils.official import module


class SourceRequired(ValueError):
    pass


def source_dataset(context):
    dataset = context["dataset"]
    root, transform = context["clean_root"], context["transform"]
    if dataset == "imagenet-c":
        if root is None:
            raise SourceRequired("Full EATA requires independent clean ImageNet training images for Fisher")
        return ImageFolder(root / "train", transform=transform)
    cls = CIFAR10 if dataset == "cifar10-c" else CIFAR100
    return cls(root, train=True, transform=transform, download=False)


@torch.enable_grad()
def estimate_fisher(model, context, settings):
    dataset = source_dataset(context)
    size = settings["fisher_size"]
    if len(dataset) < size:
        raise SourceRequired(f"Need {size} independently sourced images for EATA Fisher estimation")
    gen = torch.Generator().manual_seed(context["seed"])
    indices = torch.randperm(len(dataset), generator=gen)[:size]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=context["batch_size"])
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fishers = {n: torch.zeros_like(p) for n, p in params.items()}
    for images, _ in loader:
        outputs = model(images.to(model.head.weight.device))
        loss = cross_entropy(outputs, outputs.detach().argmax(1))
        grads = torch.autograd.grad(loss, tuple(params.values()))
        for name, grad in zip(params, grads):
            fishers[name].add_(grad.detach().square())
    return {n: (f.div_(len(loader)), params[n].detach().clone()) for n, f in fishers.items()}


class EATA(Adapter):
    timing = "predict-before-update"

    def __init__(self, model, cfg, context):
        super().__init__(model)
        params = configure_norm(self.model, gradient=True)
        opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"])
        start = perf_counter()
        bank = context["fisher_cache"]
        seed = context["seed"]
        if seed not in bank:
            bank[seed] = estimate_fisher(self.model, context, cfg)
        self.source_seconds = perf_counter() - start
        official = module(cfg, "eata.py")
        self.core = official.EATA(self.model, opt, bank[seed], cfg["fisher_alpha"],
                                  e_margin=cfg["entropy_factor"] * math.log(model.head.out_features),
                                  d_margin=cfg["diversity_margin"])
        self.events = {}

    def reset(self):
        super().reset()
        self.core.reset()
        self.core.reset_model_probs(None)
        self.core.num_samples_update_1 = self.core.num_samples_update_2 = 0
        self.events = {}

    def update(self, images):
        before = self.core.num_samples_update_2
        outputs = self.core(images)
        self.forward_passes += 1
        kept = self.core.num_samples_update_2 - before
        self.backward_passes += bool(kept)
        self.events = {"selected": kept}
        return outputs.detach()
