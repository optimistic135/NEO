from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from cap_experiments.protocol import imagenet_directory


class CIFAR(Dataset):
    def __init__(self, root, corruption, severity, transform):
        self.transform = transform
        self.pixels = np.load(root / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
        if self.pixels.shape != (50000, 32, 32, 3) or self.pixels.dtype != np.uint8:
            raise ValueError(f"Malformed CIFAR-C array: {self.pixels.shape}, {self.pixels.dtype}")
        self.offset = (severity - 1) * 10000
        labels = np.load(root / "labels.npy", allow_pickle=False)
        self.targets = labels if len(labels) == 10000 else labels[self.offset:self.offset + 10000]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.pixels[self.offset + index])), int(self.targets[index])


def load_data(spec, corruption, severity, transform):
    root = Path(spec["root"])
    if spec["classes"] == 1000:
        data = ImageFolder(imagenet_directory(root, corruption, severity), transform=transform)
        ids = [str(Path(path).relative_to(root)) for path, _ in data.samples]
    else:
        data = CIFAR(root, corruption, severity, transform)
        ids = list(range(len(data)))
    counts = np.bincount(data.targets, minlength=spec["classes"])
    if len(counts) != spec["classes"] or not np.all(counts == spec["per_class"]):
        raise ValueError(f"Unexpected class counts for {corruption}: {counts.tolist()}")
    return data, ids


def balanced(n, seed):
    return np.random.default_rng(seed).permutation(n)
