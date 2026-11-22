"""Cyclic integer quotas, with genuinely distinct images inside each subset."""

import numpy as np

from cap_experiments.online_ir import OnlineIR, make_stream

DESIGN = {
    "cifar10-c": (10, 1000, (1, 20, 50, 5000)),
    "cifar100-c": (100, 100, (1, 200, 500, 50000)),
    "imagenet-c": (1000, 100, (1, 1000, 5000, 500000)),
}


class CapacityError(ValueError):
    pass


def build_stream(labels, dataset, ir, seed, policy="strict"):
    classes, size, ratios = DESIGN[dataset]
    if ir not in ratios or policy not in ("strict", "cap"):
        raise ValueError(f"Unsupported IR or policy: {dataset}/{ir}/{policy}")
    if dataset != "imagenet-c":
        return make_stream(labels, classes, size, ir, seed), False

    labels = np.asarray(labels)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= classes))):
        raise ValueError("Invalid ImageNet labels")
    pools = [np.flatnonzero(labels == c) for c in range(classes)]
    capacity = min(map(len, pools))
    denom = ir + classes - 1
    requested = (2 * size * ir + denom) // (2 * denom)
    if capacity == 0 or sum(map(len, pools)) < size:
        raise CapacityError("Insufficient class coverage or total images")
    if requested > capacity and policy == "strict":
        raise CapacityError(f"IR={ir} needs {requested} distinct dominant-class images per subset; capacity={capacity}")
    diagonal = min(requested, capacity)
    minority, extra = divmod(size - diagonal, classes - 1)
    if minority + bool(extra) > capacity:
        raise CapacityError("Minority quota exceeds class capacity")
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(classes)
    subset_order = rng.permutation(classes)
    counts = np.full((classes, classes), minority, dtype=np.int64)
    indices = np.empty((classes, size), dtype=np.int64)
    for pos, rank in enumerate(subset_order):
        quota = counts[pos]
        quota[class_order[rank]] = diagonal
        quota[class_order[(rank + np.arange(1, extra + 1)) % classes]] += 1
        chosen = [rng.choice(pools[c], int(quota[c]), replace=False) for c in np.flatnonzero(quota)]
        indices[pos] = rng.permutation(np.concatenate(chosen))
    return OnlineIR(indices, class_order[subset_order], class_order, counts, ir / denom, 1 / denom), requested > diagonal
