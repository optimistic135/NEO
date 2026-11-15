"""Fixed-budget long-tail and held-out balanced sampling."""

import numpy as np

from cap_experiments.protocol import Stream, natural_stream


def longtail_prior(order, rho):
    if rho < 1 or len(order) < 2:
        raise ValueError("Need at least two classes and rho >= 1")
    weights = rho ** (-np.linspace(0., 1., len(order)))
    prior = np.empty(len(order))
    prior[order] = weights / weights.sum()
    return prior


def _draw(pools, size, prior, rng):
    """Systematic rounding of target counts, then sample distinct images from each pool."""
    capacity = np.asarray([len(pool) for pool in pools])
    if capacity.sum() < size:
        raise ValueError(f"Requested {size} distinct images, only {capacity.sum()} available")
    positions = (np.arange(size) + rng.random()) / size
    # Randomize the phase order of classes so ties and rounding do not privilege label IDs.
    order = rng.permutation(len(pools))
    cumulative = prior[order].cumsum()
    cumulative[-1] = 1
    assigned = order[np.searchsorted(cumulative, positions, side="right")]
    requested = np.bincount(assigned, minlength=len(pools))
    counts = np.minimum(requested, capacity)
    overflow = int(size - counts.sum())
    for _ in range(overflow):
        weights = prior * (counts < capacity)
        counts[rng.choice(len(pools), p=weights / weights.sum())] += 1
    indices = np.concatenate([pool[:count] for pool, count in zip(pools, counts)])
    rest = [pool[count:] for pool, count in zip(pools, counts)]
    return rng.permutation(indices).tolist(), rest, {
        "requested_counts": requested.tolist(),
        "redistributed_count": overflow,
        "capacity_limited_classes": np.flatnonzero(requested > capacity).tolist(),
    }


def ablation_stream(labels, classes, seed, budget, validation_per_class):
    natural = natural_stream(len(labels), budget, seed)
    remaining = np.asarray(natural.validation["validation"])
    balanced = []
    for label in range(classes):
        available = remaining[labels[remaining] == label]
        if len(available) < validation_per_class:
            raise ValueError(f"Class {label}: not enough distinct held-out images for balanced validation")
        balanced.extend(available[:validation_per_class])
    validation = np.random.default_rng(seed + 2).permutation(balanced).tolist()
    return Stream("balanced", None, None, natural.adaptation, {"balanced_validation": validation})


def longtail_stream(labels, classes, seed, rho, budget, validation_per_class):
    # The same class ranking and fixed held-out balanced subset are used across rhos.
    rng = np.random.default_rng(seed)
    order = rng.permutation(classes)
    prior = longtail_prior(order, rho)
    pools = [rng.permutation(np.flatnonzero(labels == label)) for label in range(classes)]
    if min(map(len, pools)) < validation_per_class:
        raise ValueError("Insufficient images for distinct balanced holdout")
    balanced = np.concatenate([pool[:validation_per_class] for pool in pools])
    pools = [pool[validation_per_class:] for pool in pools]
    adaptation, remainder, adapt_audit = _draw(pools, budget, prior, np.random.default_rng(seed + 1))
    matched, _, matched_audit = _draw(remainder, len(balanced), prior, np.random.default_rng(seed + 2))
    balanced = np.random.default_rng(seed + 3).permutation(balanced).tolist()
    stream = Stream(f"rho_{rho}", rho, None, adaptation, {
        "balanced_validation": balanced, "matched_validation": matched,
    })
    return stream, {
        "rho": rho, "class_order": order.tolist(), "target_prior": prior.tolist(),
        "adaptation_sampling": adapt_audit, "matched_sampling": matched_audit,
    }
