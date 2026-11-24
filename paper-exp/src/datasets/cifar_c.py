from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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


def load_cifar_c(configuration, corruption, severity, transform):
    if configuration.get("type", "cifar_c") != "cifar_c" or not 1 <= severity <= 5:
        raise ValueError("Expected a CIFAR-C dataset and severity in [1,5]")
    data = CIFAR(Path(configuration["root"]), corruption, severity, transform)
    counts = np.bincount(data.targets, minlength=configuration["classes"])
    if len(counts) != configuration["classes"] or not np.all(counts == configuration["per_class"]):
        raise ValueError(f"Unexpected class counts for {corruption}: {counts.tolist()}")
    return data
