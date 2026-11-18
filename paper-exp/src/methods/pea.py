from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.methods.base import Adapter
from src.utils.upstream import load_upstream


class HFBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, tokens):
        return self.block(tokens)[0]


class HFViTView(torch.nn.Module):
    """PEA-compatible block boundaries with the checkpoint's original weights."""

    def __init__(self, network):
        super().__init__()
        vit = network.vit
        self.patch_embed = vit.embeddings.patch_embeddings
        self.cls_token = vit.embeddings.cls_token
        self.pos_embed = vit.embeddings.position_embeddings
        self.pos_drop = vit.embeddings.dropout
        self.blocks = torch.nn.ModuleList(HFBlock(block) for block in vit.encoder.layer)
        self.norm = vit.layernorm
        self.head = network.classifier

    def forward(self, images):
        patches = self.patch_embed(images)
        tokens = torch.cat((self.cls_token.expand(len(images), -1, -1), patches), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])


class PEA(Adapter):
    timing = "two-pass-current-batch"
    offline = {}

    def __init__(self, model, dataset, spec, transform, seed, upstream, cfg, smoke):
        super().__init__(model)
        if not spec.get("pea_source"):
            raise FileNotFoundError(f"PEA requires clean source statistics or an existing source dataset for {dataset}")
        resnet = model.architecture == "resnet50"
        root = spec["pea_source"]
        if dataset == "imagenet-c":
            train = datasets.ImageFolder(Path(root) / "train", transform=transform)
        else:
            cls = datasets.CIFAR10 if dataset == "cifar10-c" else datasets.CIFAR100
            train = cls(root, train=True, transform=transform, download=False)
        kind = "resnet" if resnet else "vit"
        self.upstream = load_upstream(upstream, "pea", f"pea_{kind}.py")
        self.device = model.head.weight.device
        self.microbatch = cfg["augment_microbatch"]
        native = self.model.network
        self.network = HFViTView(native).eval() if self.model.head_name == "classifier" else native
        network = self.network
        if network is not native:
            with torch.no_grad():
                example = torch.zeros(1, 3, 224, 224, device=self.device)
                torch.testing.assert_close(network(example), native(example).logits)
        suffix = "ResNet" if resnet else "ViT"
        self.stats = getattr(self.upstream, f"PEAStats{suffix}")(
            momentum=.05 if dataset == "cifar10-c" else .02, alpha=cfg["alpha"],
            gamma=2.5 if resnet or dataset == "cifar10-c" else 4.,
            entropy_threshold=({"cifar10-c": .2, "cifar100-c": .5, "imagenet-c": 1.}[dataset]
                               if resnet else .5 if dataset == "cifar10-c" else 1.),
        )
        n = min(cfg["smoke_source_samples" if smoke else "source_samples"], len(train))
        first, _ = random_split(train, [n, len(train)-n], generator=torch.Generator().manual_seed(seed))
        def source_images(items):
            return torch.stack([image for image, _ in items]), None
        loader = DataLoader(first, batch_size=cfg["source_batch"], shuffle=True, num_workers=0, collate_fn=source_images)
        key = (dataset, model.architecture, str(self.device), seed, root, n)
        if key not in self.offline:
            args = {"max_samples": n} if resnet else {"use_cls_only": False}
            self.offline[key] = tuple({k: v.cpu() for k,v in result.items()} for result in
                getattr(self.upstream, f"compute_source_stats_{kind}")(network, loader, self.device, **args))
        means, variances, cov = self.offline[key]
        group = (64 if dataset == "cifar10-c" else 32) if resnet else (192 if dataset == "cifar10-c" else 768)
        self.aligner = getattr(self.upstream, f"PEA{suffix}")(
            network, means, variances, cov, self.stats, group_size=group).to(self.device).eval()
        self.infer = getattr(self.upstream, f"pea_{kind}_infer")
        self.options = {} if resnet else {"weight_use_cls_only": dataset != "cifar10-c"}

    def reset(self):
        super().reset()
        self.stats.reset()
        self.aligner.reset_state()

    @torch.no_grad()
    def update(self, images):
        count = [0]
        def count_pass(*_):
            count[0] += 1
        hooks = [module.register_forward_pre_hook(count_pass) for module in (self.network, self.aligner)]
        try:
            logits, _ = self.infer(self.network, self.aligner, images.cpu(), self.device,
                                   microbatch_variants=self.microbatch, **self.options)
        finally:
            for hook in hooks:
                hook.remove()
        self.forward_passes += count[0]
        ent = self.upstream.entropy_from_logits(logits)
        if self.stats.entropy_spike(ent):
            self.stats.reset()
            self.aligner.reset_state()
            self.events = {"fallback": "entropy_reset"}
        else:
            self.events = {}
        self.stats.update_entropy(ent)
        return logits
