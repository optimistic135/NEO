"""Consume externally authenticated official DART indices without inventing draws."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.utils.settings import sha256


def data_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_manifest(root, dataset, seed, ir, ids, subset, classes):
    if root is None:
        raise FileNotFoundError("Verified DART author sampling manifests are required for Online IR")
    base = Path(root) / dataset / f"{seed}_{ir}"
    meta = json.loads(base.with_suffix(".json").read_text())
    path = base.with_suffix(".npy")
    if (not meta.get("author_source") or meta["indices_sha256"] != sha256(path)
            or meta["sample_ids_sha256"] != data_digest(ids)):
        raise ValueError(f"Official index provenance or data ordering mismatch: {base}")
    indices = np.load(path, allow_pickle=False)
    if (indices.shape != (classes, subset) or indices.dtype.kind not in "iu"
            or indices.min() < 0 or indices.max() >= len(ids)):
        raise ValueError(f"Invalid DART indices: {base}")
    order = meta["dominant_class_order"]
    if sorted(order) != list(range(classes)):
        raise ValueError(f"Invalid DART subset order: {base}")
    return indices, meta
