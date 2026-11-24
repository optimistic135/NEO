from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from src.utils.settings import HOME, sha256 as file_sha256


@torch.no_grad()
def source_stats(prompt, context):
    root = Path(context["clean_root"])
    key = dict(checkpoint=context["audit"]["sha256"], preprocessing=context["audit"]["preprocessing"],
               config=context["audit"]["model_config_sha256"], source_sha256=context["source_sha256"],
               implementation_sha256=file_sha256(HOME / "src/models/prompt_interface.py"),
               stats="unprompted-all-block-cls-next-ln-fp64-unbiased")
    digest = sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = context["output"] / "source_stats" / f"{digest}.pt"
    device = prompt.prompts.device
    if path.exists():
        return torch.load(path, weights_only=True, map_location=device), 0.
    path.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    dataset = CIFAR100(root, train=True, transform=context["transform"], download=False)
    total = square = None
    seen = 0
    for images, _ in tqdm(DataLoader(dataset, batch_size=64), desc="Clean CIFAR-100 source statistics"):
        matrix = torch.cat(prompt.layers(images.to(device)), dim=1).double()
        s, q = matrix.sum(0), matrix.square().sum(0)
        total = s if total is None else total + s
        square = q if square is None else square + q
        seen += len(images)
    if seen != 50000:
        raise ValueError("Expected the complete clean CIFAR-100 training split")
    std = ((square - total.square() / seen) / (seen - 1)).clamp_min(0).sqrt().float()
    value = (std, (total / seen).float())
    if not all(torch.isfinite(t).all() for t in value):
        raise ValueError("Non-finite clean source statistics")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    tmp = path.with_suffix(".tmp")
    torch.save(tuple(t.cpu() for t in value), tmp)
    tmp.replace(path)
    return value, seconds


def clean_digest(root):
    return file_sha256(Path(root) / "cifar-100-python/train")
