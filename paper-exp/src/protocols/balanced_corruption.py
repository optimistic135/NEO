from hashlib import sha256

import numpy as np


def build_split(labels, classes, seed, adaptation, validation):
    """Reserve the equal validation quota, then apportion the adaptation remainder."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or set(labels) != set(range(classes)):
        raise ValueError("Unexpected label space")
    if min(adaptation, validation) <= 0 or validation % classes:
        raise ValueError("Validation requires an equal positive quota per class")
    rng = np.random.default_rng(seed)
    base, remainder = divmod(adaptation, classes)
    adapt, valid = [], []
    per_class = validation // classes
    for order, label in enumerate(rng.permutation(classes)):
        pool = rng.permutation(np.flatnonzero(labels == label))
        samples = base + (order < remainder)
        if len(pool) < per_class + samples:
            raise ValueError(f"Insufficient samples for class {label}")
        valid.extend(pool[:per_class])
        adapt.extend(pool[per_class:per_class + samples])
    return rng.permutation(adapt), rng.permutation(valid)


def digest(indices):
    return sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
