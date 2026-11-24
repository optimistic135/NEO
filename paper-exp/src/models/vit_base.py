import torch
from PIL import Image

from src.models.loading import cached, load
from src.utils.settings import sha256


def load_vit_base(spec, profile, classes):
    if profile != "cifar100-c" or classes != 100:
        raise ValueError("This experiment supports ViT-B/16 with 100-way CIFAR classification")
    checkpoint = spec[profile]
    path = cached(checkpoint["repo"], checkpoint["weights"], checkpoint["revision"])
    if sha256(path) != spec["cifar100_sha256"]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    for filename, key in (("config.json", "cifar100_config_sha256"),
                          ("preprocessor_config.json", "cifar100_processor_sha256")):
        if sha256(path.parent / filename) != spec[key]:
            raise ValueError(f"Checkpoint metadata SHA256 mismatch: {filename}")
    model, transform, audit = load(profile, "vit_base", classes)
    cfg = model.network.config
    actual = (cfg.image_size, cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads)
    if actual != (224, 16, 768, 12, 12) or audit["sha256"] != spec["cifar100_sha256"]:
        raise ValueError(f"Expected original ViT-B/16 checkpoint, got {actual}")
    normalizer = model.normalization
    if any(len(x) != 3 for x in normalizer) or min(normalizer[1]) <= 0:
        raise ValueError("Invalid checkpoint normalization")
    sample = transform(Image.new("RGB", (32, 32)))
    if sample.shape != (3, 224, 224) or not torch.isfinite(sample).all():
        raise ValueError("Invalid input transform")
    audit.update(image_size=224, patch_size=16, num_layers=12, attention_heads=12,
                 model_config_sha256=spec["cifar100_config_sha256"],
                 processor_sha256=spec["cifar100_processor_sha256"])
    return model, transform, audit
