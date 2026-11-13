"""Corruption datasets and reproducible stream definitions."""

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dataset.selectedRotateImageFolder import SelectedRotateImageFolder, te_transforms_imageC


NUM_CLASSES = 10
PER_LEVEL = 10_000
PROCESSOR_ID = "google/vit-base-patch16-224-in21k"
PROCESSOR_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
IMAGENET_CLASSES = 1000
IMAGENET_GROUPS = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}


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


def imagenet_directory(root, corruption, level):
    group = next(group for group, names in IMAGENET_GROUPS.items() if corruption in names)
    parents = [root, root / group]
    if group == "blur":
        parents.append(root / "bur")
    paths = [parent / corruption / str(level) for parent in parents]
    found = {path.resolve(): path for path in paths if path.is_dir()}
    if len(found) != 1:
        raise ValueError(f"Expected one ImageNet-C directory for {corruption}: {paths}")
    return next(iter(found.values()))


def load_imagenet_dataset(root, corruption, level):
    directory = imagenet_directory(root, corruption, level)
    dataset = SelectedRotateImageFolder(
        str(directory), te_transforms_imageC, original=True, rotation=False
    )
    labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    if (len(dataset.classes) != IMAGENET_CLASSES
            or not np.all(np.bincount(labels, minlength=IMAGENET_CLASSES) == 50)):
        raise ValueError(f"Expected 1,000 ImageNet classes with 50 images each in {directory}")
    sample_ids = [str(Path(path).relative_to(root)) for path, _ in dataset.samples]
    return dataset, labels, sample_ids


@dataclass(frozen=True)
class Stream:
    name: str
    ratio: int | None
    dominant_class: int | None
    adaptation: list[int]
    validation: dict[str, list[int]]
    validation_priors: dict[str, list[float]] | None = None


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
    matched = np.random.default_rng(seed + 3).permutation(matched).tolist()
    balanced_members, matched_members = set(balanced), set(matched)
    # Store the order actually seen by both views of the shared validation pass.
    matched = [index for index in balanced if index in matched_members] + [
        index for index in matched if index not in balanced_members
    ]
    return Stream(
        f"ratio_{ratio:03d}", ratio, dominant, adapt,
        {"balanced_validation": balanced, "matched_validation": matched},
    )


def controlled_imagenet_stream(labels, adapt_size, validation_size, seed, ratio):
    """Hold out the same images per class at every ratio, including the uniform control."""
    per_class = validation_size // IMAGENET_CLASSES
    if not 1 <= per_class < 50 or validation_size % IMAGENET_CLASSES:
        raise ValueError("ImageNet controlled validation needs 1–49 images per class")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(indices) for indices in np.split(
        np.argsort(labels, kind="stable"), IMAGENET_CLASSES
    )]
    dominant = seed % IMAGENET_CLASSES if ratio else None
    prior = np.full(IMAGENET_CLASSES, 1 / IMAGENET_CLASSES)
    order = rng.permutation(IMAGENET_CLASSES)
    counts = np.zeros(IMAGENET_CLASSES, dtype=int)
    remaining = adapt_size
    if dominant is not None:
        share = ratio / 100
        prior.fill((1 - share) / (IMAGENET_CLASSES - 1))
        prior[dominant] = share
        counts[dominant] = (adapt_size * ratio + 50) // 100
        remaining -= counts[dominant]
        order = order[order != dominant]
    base, extra = divmod(remaining, len(order))
    counts[order] = base
    counts[order[:extra]] += 1

    samples = []
    for pool, count in zip(pools, counts):
        available = pool[:-per_class]
        samples.extend(available[:count])
        if count > len(available):
            samples.extend(rng.choice(available, count - len(available), replace=True))
    adapt = np.random.default_rng(seed + 1).permutation(samples).tolist()
    balanced = np.random.default_rng(seed + 2).permutation(
        np.concatenate([pool[-per_class:] for pool in pools])
    ).tolist()
    return Stream(
        f"ratio_{ratio:03d}", ratio, dominant, adapt,
        {"balanced_validation": balanced, "matched_validation": balanced},
        {"matched_validation": prior.tolist()},
    )


def manifest(stream, labels, sample_ids, num_classes=NUM_CLASSES):
    splits = {"adaptation": stream.adaptation, **stream.validation}
    adapt = set(stream.adaptation)
    for name, indices in splits.items():
        repeats_allowed = name == "adaptation" and stream.validation_priors is not None
        if not indices or (not repeats_allowed and len(set(indices)) != len(indices)):
            raise ValueError(f"Empty split or duplicate samples in {name}")
        if name != "adaptation" and adapt.intersection(indices):
            raise ValueError(f"Adaptation overlaps {name}")

    result = {
        name: {
            "indices": indices,
            "sample_ids": [sample_ids[index] for index in indices],
            "label_counts": np.bincount(labels[indices], minlength=num_classes).tolist(),
        }
        for name, indices in splits.items()
    }
    for name, prior in (stream.validation_priors or {}).items():
        result[name]["target_prior"] = prior
    return result
