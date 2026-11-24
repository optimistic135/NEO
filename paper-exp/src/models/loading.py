import os
from pathlib import Path

import timm
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor
from safetensors.torch import load_file
from huggingface_hub import try_to_load_from_cache

from src.utils.settings import config, sha256


class Classifier(nn.Module):
    def __init__(self, network, architecture, head):
        super().__init__()
        self.network = network
        self.architecture = architecture
        self.head_name = head

    @property
    def head(self):
        return getattr(self.network, self.head_name)

    def features(self, images):
        net = self.network
        if self.head_name == "classifier":
            return net.vit(images)[0][:, 0]
        if self.architecture == "vit_base" or hasattr(net, "forward_head"):
            return net.forward_head(net.forward_features(images), pre_logits=True)
        images = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            images = layer(images)
        return net.avgpool(images).flatten(1)

    def forward(self, images):
        return self.head(self.features(images))


def cached(repo, filename, revision):
    file = try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(file, str) or not Path(file).is_file():
        raise FileNotFoundError(f"Missing cached {repo}@{revision}/{filename}")
    return Path(file)


def load(dataset, architecture, classes):
    extra = {}
    if architecture == "resnet50":
        cfg = config("models/resnet50.yaml")
        path = Path(cfg["root"]) / cfg["checkpoints"][dataset]
        head = "fc"
        if dataset == "imagenet-c":
            network = resnet50(weights=None)
            network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
            kind = "torchvision-state-dict"
        else:
            # These are the user-provided, inspected PEA model archives.
            network = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(network, timm.models.resnet.ResNet):
                raise ValueError("PEA CIFAR checkpoint must contain a timm ResNet")
            if tuple(len(getattr(network, f"layer{i}")) for i in range(1, 5)) != (3, 4, 6, 3):
                raise ValueError("PEA checkpoint does not have ResNet-50 stage depths")
            transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                   T.Normalize((.485, .456, .406), (.229, .224, .225))])
            kind = "PEA-full-model"
    else:
        spec = config("models/vit_base.yaml")[dataset]
        path = cached(spec["repo"], spec["weights"], spec["revision"])
        extra.update(spec)
        if dataset == "imagenet-c":
            from dataset.selectedRotateImageFolder import te_transforms_imageC

            network = timm.create_model("vit_base_patch16_224", pretrained=False)
            network.load_state_dict(load_file(str(path)), strict=True)
            transform, head, kind = te_transforms_imageC, "head", "timm-safetensors"
        else:
            network, info = ViTForImageClassification.from_pretrained(
                str(path.parent), local_files_only=True, output_loading_info=True,
                use_safetensors=path.suffix == ".safetensors",
            )
            bad = {k: v for k, v in info.items() if v}
            if bad:
                raise ValueError(f"ViT loading diagnostics: {bad}")
            processor_path = (cached("google/vit-base-patch16-224-in21k", "preprocessor_config.json",
                                     "b4569560a39a0f1af58e3ddaf17facf20ab919b0")
                              if dataset == "cifar10-c" else path)

            processor = ViTImageProcessor.from_pretrained(str(processor_path.parent), local_files_only=True)
            transform = lambda image: processor(images=image, return_tensors="pt").pixel_values[0]
            extra["processor"] = processor.to_dict()
            head, kind = "classifier", "huggingface"
    model = Classifier(network, architecture, head).eval().requires_grad_(False)
    if not isinstance(model.head, nn.Linear) or model.head.out_features != classes:
        raise ValueError(f"Expected a linear {classes}-class head")
    if "processor" in extra:
        model.normalization = (processor.image_mean, processor.image_std)
    else:
        normalizer = transform.transforms[-1] if isinstance(transform, T.Compose) else transform
        model.normalization = (normalizer.mean, normalizer.std)
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224)
        expected = network(x)
        if hasattr(expected, "logits"):
            expected = expected.logits
        torch.testing.assert_close(model(x), expected)
    return model, transform, dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size,
           architecture=type(network).__module__ + "." + type(network).__name__, classes=classes,
           features=model.head.in_features, format=kind, preprocessing=extra.get("processor", repr(transform)),
           missing_keys=[], unexpected_keys=[], log="Strict load or archived model with native-forward parity", **extra)
