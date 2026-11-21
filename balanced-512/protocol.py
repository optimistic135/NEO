from hashlib import sha256

import numpy as np


def split(targets, seed, budget=512):
    """Allocate floor(budget/C) per class, with the remainder seeded across classes."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(targets))
    base, remainder = divmod(budget, len(classes))
    pools = [rng.permutation(np.flatnonzero(targets == label)) for label in classes]
    quotas = [base + (i < remainder) for i in range(len(classes))]
    if any(len(pool) < quota for pool, quota in zip(pools, quotas)) or len(targets) <= budget:
        raise ValueError("Insufficient images for disjoint adaptation and validation")
    adapt = rng.permutation(np.concatenate([p[:q] for p, q in zip(pools, quotas)]))
    validation = rng.permutation(np.concatenate([p[q:] for p, q in zip(pools, quotas)]))
    return adapt, validation


def split_hash(indices):
    return sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
