import torch
from torch.nn import functional as F

from src.methods.base import Optimizing, configure_norm, entropy


class ROID(Optimizing):
    timing = "pre-update-logits-with-current-batch-prior"

    def __init__(self, model, cfg, imagenet, batch, augment):
        super().__init__(model)
        self.cfg, self.batch, self.augment = cfg, batch, augment
        self.eps = float(cfg["eps"])
        if not 0 < self.eps < 1:
            raise ValueError("ROID eps must be between 0 and 1")
        self.params = configure_norm(self.model, True)
        self.optimizer = (torch.optim.SGD(self.params, lr=cfg["imagenet_lr"], momentum=cfg["momentum"])
                          if imagenet else torch.optim.Adam(self.params, lr=cfg["cifar_lr"]))
        self.save_initial()
        self.reset()

    def reset(self):
        super().reset()
        classes = self.model.head.out_features
        self.prior = self.model.head.weight.new_full((classes,), 1/classes)
        self.events = {}

    @torch.enable_grad()
    def update(self, images):
        logits = self.predict(images)
        prob = logits.softmax(1)
        eps = self.eps
        with torch.no_grad():
            diversity = 1 - F.cosine_similarity(self.prior[None], prob, dim=1)
            diversity = (diversity-diversity.min()) / (diversity.max()-diversity.min()).clamp_min(eps)
            keep = diversity >= diversity.mean()
            certainty = -entropy(logits)
            certainty = (certainty-certainty.min()) / (certainty.max()-certainty.min()).clamp_min(eps)
            weights = (diversity * certainty / self.cfg["temperature"]).exp()[keep]
            batch_prior = prob.mean(0)
            self.prior.lerp_(batch_prior, 1-self.cfg["prior_momentum"])
        clipped = prob.clamp(max=self.cfg["clip"])
        slr = -(clipped * (clipped / (1-clipped) + eps).log()).sum(1)
        if keep.any():
            aug = self.predict(self.augment(images[keep]))
            original = logits[keep]
            sce = -.5 * ((original.softmax(1)*aug.log_softmax(1)).sum(1)
                          + (aug.softmax(1)*original.log_softmax(1)).sum(1))
            self.optimizer.zero_grad(set_to_none=True)
            self.backward(((slr[keep]+sce)*weights).sum()/self.batch)
            self.optimizer.step()
        with torch.no_grad():
            for param, source in zip(self.params, self.initial):
                param.lerp_(source, 1-self.cfg["source_momentum"])
            smooth = max(1/len(images), 1/prob.shape[1]) / batch_prior.max()
            return logits.detach() * (batch_prior+smooth)/(1+smooth*prob.shape[1])


def build_roid(model, cfg, context):
    from src.utils.official import module

    upstream = module(cfg, "classification/augmentations/transforms_cotta.py")
    transform = upstream.get_tta_transforms(224, padding_mode="reflect", cotta_augs=False)
    mean, std = (model.head.weight.new_tensor(x)[None, :, None, None] for x in model.normalization)

    def augment(images):
        return (transform(images * std + mean) - mean) / std

    return ROID(model, cfg, context["dataset"] == "imagenet-c", context["batch_size"], augment)
