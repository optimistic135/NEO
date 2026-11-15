"""Balanced integer allocation for continuous Online IR streams."""

from dataclasses import dataclass
import hashlib

import numpy as np


SPECS = {
    "cifar10-c": (10, 1000, 64, (1, 5, 20, 50, 5000)),
    "cifar100-c": (100, 100, 64, (1, 200, 500, 50000)),
    "imagenet-c": (1000, 50, 50, (1, 1000, 5000, 500000)),
}


@dataclass(frozen=True)
class OnlineIR:
    indices: np.ndarray
    dominant_classes: np.ndarray
    class_order: np.ndarray
    counts: np.ndarray
    p_max: float
    p_min: float

    @property
    def index_sha256(self):
        return hashlib.sha256(self.indices.astype("<i8", copy=False).tobytes()).hexdigest()

    @property
    def sha256(self):
        digest = hashlib.sha256()
        for array in (self.indices, self.dominant_classes, self.class_order):
            digest.update(array.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def batches(self, batch_size):
        for subset in self.indices:
            for start in range(0, len(subset), batch_size):
                yield subset[start:start + batch_size].tolist()


def make_stream(labels, classes, subset_size, ir, seed):
    labels = np.asarray(labels)
    if (classes < 2 or subset_size < 1 or ir < 1 or labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= classes))):
        raise ValueError("Invalid Online IR class labels or configuration")
    available = np.bincount(labels, minlength=classes)
    if np.any(available != subset_size):
        raise ValueError(f"Expected exactly {subset_size} distinct images per class, got {available.tolist()}")

    # Half-up rounding of the diagonal quota using integer arithmetic.
    denom = ir + classes - 1
    diagonal = (2 * subset_size * ir + denom) // (2 * denom)
    baseline, extra = divmod(subset_size - diagonal, classes - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(classes)
    subset_order = rng.permutation(classes)
    counts = np.full((classes, classes), baseline, dtype=np.int64)
    ranks = np.arange(classes)
    counts[ranks, order] = diagonal
    for offset in range(1, extra + 1):
        counts[ranks, order[(ranks + offset) % classes]] += 1
    pools = [rng.permutation(np.flatnonzero(labels == label)) for label in range(classes)]
    consumed = np.zeros(classes, dtype=np.int64)
    indices = np.empty((classes, subset_size), dtype=np.int64)
    for position, rank in enumerate(subset_order):
        quota = counts[rank]
        members = np.flatnonzero(quota)
        images = [pools[label][consumed[label]:consumed[label] + quota[label]] for label in members]
        consumed += quota
        indices[position] = rng.permutation(np.concatenate(images))
    if not np.array_equal(available, consumed) or len(np.unique(indices)) != indices.size:
        raise RuntimeError("Cyclic quota allocation lost or repeated images")
    return OnlineIR(indices, order[subset_order], order, counts[subset_order], ir / denom, 1 / denom)
