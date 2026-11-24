from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder

from src.datasets.cifar_c import load_cifar_c


def load_data(spec, corruption, severity, transform):
    root = Path(spec["root"])
    if spec["classes"] == 1000:
        from cap_experiments.protocol import imagenet_directory

        data = ImageFolder(imagenet_directory(root, corruption, severity), transform=transform)
        ids = [str(Path(path).relative_to(root)) for path, _ in data.samples]
    else:
        data = load_cifar_c(spec, corruption, severity, transform)
        return data, list(range(len(data)))
    counts = np.bincount(data.targets, minlength=spec["classes"])
    if len(counts) != spec["classes"] or not np.all(counts == spec["per_class"]):
        raise ValueError(f"Unexpected class counts for {corruption}: {counts.tolist()}")
    return data, ids


def balanced(n, seed):
    return np.random.default_rng(seed).permutation(n)
