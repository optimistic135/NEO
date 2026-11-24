import torch

from src.methods.base import Adapter
from src.utils.official import module


class CoTTA(Adapter):
    timing = "ema-teacher-prediction-before-student-update"

    def __init__(self, model, cfg, imagenet=False):
        super().__init__(model)
        dialect = "imagenet" if imagenet else "cifar"
        transforms = module(cfg, f"{dialect}/my_transforms.py")
        changes = [
            ("import my_transforms as my_transforms", ""),
            ("resample=PIL.Image.BILINEAR,", "interpolation=transforms.InterpolationMode.BILINEAR,"),
            ("fillcolor=None", "fill=0"),
            (".float().cuda()", ".to(p)"),
        ]
        if not imagenet:
            changes.append(("img_shape = (32, 32, 3)", "img_shape = (224, 224, 3)"))
        upstream = module(cfg, f"{dialect}/cotta.py", changes, my_transforms=transforms)
        weights = []
        for layer in self.model.modules():
            for name, param in layer.named_parameters(recurse=False):
                if name in ("weight", "bias"):
                    param.requires_grad_(True)
                    weights.append(param)
        self.model.train()
        if imagenet:
            opt = torch.optim.SGD(weights, lr=cfg["imagenet_lr"], momentum=.9)
            self.core = upstream.CoTTA(self.model, opt)
        else:
            opt = torch.optim.Adam(weights, lr=cfg["lr"])
            self.core = upstream.CoTTA(self.model, opt, mt_alpha=cfg["mt"], rst_m=cfg["rst"], ap=cfg["ap"])
        mean, std = (self.model.head.weight.new_tensor(x).reshape(1, 3, 1, 1) for x in model.normalization)
        augment = self.core.transform
        self.core.transform = lambda images: (augment(images * std + mean) - mean) / std
        self.core.model_anchor.requires_grad_(False)
        self.events = {}

    def reset(self):
        super().reset()
        self.core.reset()
        self.core.model_anchor.requires_grad_(False)

    def update(self, images):
        def count(*_):
            self.forward_passes += 1
        hooks = [x.register_forward_pre_hook(count) for x in (self.model, self.core.model_ema, self.core.model_anchor)]
        try:
            logits = self.core(images)
        finally:
            for hook in hooks:
                hook.remove()
        self.backward_passes += 1
        return logits.detach()

    @torch.no_grad()
    def predict(self, images):
        self.forward_passes += 1
        return self.core.model_ema.eval()(images)
