"""CIFAR-10-C input and reproducible stream definitions."""

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dataset.selectedRotateImageFolder import SelectedRotateImageFolder


NUM_CLASSES = 10
PER_LEVEL = 10_000
PROCESSOR_ID = "google/vit-base-patch16-224-in21k"
PROCESSOR_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"


class CIFARArrays(Dataset):
    """Read a single severity from the original arrays without duplicating images."""

    def __init__(self, root, corruption, level, processor):
        start = (level - 1) * PER_LEVEL
        stop = start + PER_LEVEL
        images = np.load(root / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
        if images.shape != (5 * PER_LEVEL, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Unexpected image array for {corruption}: {images.shape}, {images.dtype}")
        labels = np.load(root / "labels.npy", allow_pickle=False)
        if labels.shape == (5 * PER_LEVEL,):
            labels = labels[start:stop]
        if labels.shape != (PER_LEVEL,) or not np.isin(labels, np.arange(NUM_CLASSES)).all():
            raise ValueError("labels.npy must contain one or five aligned sets of 10,000 CIFAR labels")

        self.images = images[start:stop]
        self.processor = processor
        # Match ImageFolder's class-major order for the repository's PNG conversion.
        self.samples = sorted(enumerate(map(int, labels)), key=lambda sample: sample[1])
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_index, label = self.samples[index]
        image = Image.fromarray(self.images[image_index])
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, label


def load_dataset(root, corruption, level, processor):
    if (root / "labels.npy").is_file():
        dataset = CIFARArrays(root, corruption, level, processor)
        sample_ids = [index for index, _ in dataset.samples]
    else:
        directory = root / corruption / str(level)
        dataset = SelectedRotateImageFolder(
            str(directory), None, original=True, rotation=False, processor=processor
        )
        if dataset.class_to_idx != {str(i): i for i in range(NUM_CLASSES)}:
            raise ValueError(f"Expected class folders 0 through 9 in {directory}")
        sample_ids = [str(Path(path).relative_to(root)) for path, _ in dataset.samples]
    # ImageFolder.targets is not reordered by SelectedRotateImageFolder's shuffle.
    labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    if len(dataset) != PER_LEVEL or not np.all(np.bincount(labels, minlength=NUM_CLASSES) == 1000):
        raise ValueError(f"Expected 10,000 CIFAR-10-C images with 1,000 per class for {corruption}")
    return dataset, labels, sample_ids


@dataclass(frozen=True)
class Stream:
    name: str
    ratio: int | None
    dominant_class: int | None
    adaptation: list[int]
    validation: dict[str, list[int]]


def natural_stream(size, adapt_size, seed):
    # Same seeded random partition as prepare_test_data. Give ordering its own
    # seed so it is independent of model initialization, W&B, and job order.
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    generator = torch.Generator().manual_seed(seed + 1)
    adapt = [permutation[i] for i in torch.randperm(adapt_size, generator=generator).tolist()]
    remaining = permutation[adapt_size:]
    validation = [remaining[i] for i in torch.randperm(len(remaining), generator=generator).tolist()]
    return Stream("natural", None, None, adapt, {"validation": validation})


def _counts(size, dominant, ratio):
    major = (size * ratio + 50) // 100
    if ratio == 10:
        major = (size + NUM_CLASSES - 1) // NUM_CLASSES
    base, remainder = divmod(size - major, NUM_CLASSES - 1)
    counts = [base] * NUM_CLASSES
    counts[dominant] = major
    for offset in range(1, remainder + 1):
        counts[(dominant + offset) % NUM_CLASSES] += 1
    return counts


def controlled_stream(labels, adapt_size, validation_size, seed, ratio):
    if validation_size < NUM_CLASSES or validation_size % NUM_CLASSES:
        raise ValueError("Controlled validation size must be a positive multiple of ten")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == label)) for label in range(NUM_CLASSES)]
    if any(len(pool) < adapt_size + validation_size for pool in pools):
        raise ValueError("Each class needs enough samples for the 100% stream and held-out validation")
    dominant = seed % NUM_CLASSES
    adapt_counts = _counts(adapt_size, dominant, ratio)
    matched_counts = _counts(validation_size, dominant, ratio)

    # Validation always draws from a fixed held-out tail of each class pool.
    # The balanced subset is identical for every ratio at a given seed.
    adapt = np.concatenate([pool[:count] for pool, count in zip(pools, adapt_counts)])
    balanced = np.concatenate([pool[-validation_size:][:validation_size // NUM_CLASSES] for pool in pools])
    matched = np.concatenate([pool[-validation_size:][:count] for pool, count in zip(pools, matched_counts)])
    adapt = np.random.default_rng(seed + 1).permutation(adapt).tolist()
    balanced = np.random.default_rng(seed + 2).permutation(balanced).tolist()
    matched = balanced.copy() if ratio == 10 else np.random.default_rng(seed + 3).permutation(matched).tolist()
    return Stream(
        f"ratio_{ratio:03d}", ratio, dominant, adapt,
        {"balanced_validation": balanced, "matched_validation": matched},
    )


def manifest(stream, labels, sample_ids):
    splits = {"adaptation": stream.adaptation, **stream.validation}
    adapt = set(stream.adaptation)
    for name, indices in splits.items():
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"Empty split or duplicate samples in {name}")
        if name != "adaptation" and adapt.intersection(indices):
            raise ValueError(f"Adaptation overlaps {name}")

    return {
        name: {
            "indices": indices,
            "sample_ids": [sample_ids[index] for index in indices],
            "label_counts": np.bincount(labels[indices], minlength=NUM_CLASSES).tolist(),
        }
        for name, indices in splits.items()
    }
