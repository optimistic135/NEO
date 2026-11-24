"""Use exactly the same seeded split as the existing balanced-512 benchmark."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[3]
_spec = spec_from_file_location("balanced_512_protocol", _ROOT / "balanced-512/protocol.py")
_shared = module_from_spec(_spec)
_spec.loader.exec_module(_shared)
digest = _shared.split_hash


def build_split(labels, classes, seed, adaptation, validation):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or set(labels) != set(range(classes)):
        raise ValueError("Unexpected label space")
    if adaptation <= 0 or validation != "remainder":
        raise ValueError("balanced-512 reserves every remaining image for validation")
    return _shared.split(labels, seed, adaptation)
