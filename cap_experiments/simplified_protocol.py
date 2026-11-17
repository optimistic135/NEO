"""The Simplified PLEAD experiment matrix and exact reuse of equivalent configurations."""

from dataclasses import replace

from cap_experiments.ablation_protocol import IR_LEVELS, VALIDATION_PER_CLASS
from tta_library.plead_simplified import Variant


FULL = Variant()
VARIANTS = {
    "plead": FULL,
    "no_composition": replace(FULL, fixed_gamma=0),
    "hard_prior": replace(FULL, hard=True),
    "norm_only_kappa": replace(FULL, norm_only=True),
    "no_evidence_gate": replace(FULL, fixed_gamma=1),
    "current_batch_only": replace(FULL, batch_only=True),
    "gamma_half": replace(FULL, fixed_gamma=.5),
    "window_64": replace(FULL, window=64),
    "window_256": replace(FULL, window=256),
    "no_gate_window_256": replace(FULL, window=256, fixed_gamma=1),
}
COMPONENTS = tuple(VARIANTS)[:6]
GATE = ("no_composition", "gamma_half", "no_evidence_gate", "plead")
WINDOW = ("window_64", "plead", "window_256")
MECHANISM = ("plead", "no_composition", "no_evidence_gate")
COMBINATION = ("plead", "no_evidence_gate", "no_composition", "hard_prior", "no_gate_window_256")
SUITES = ("standard", "online_ir", "gate", "window", "mechanism", "combination")
WINDOW_LEVELS = {"cifar10-c": (50, 5000), "cifar100-c": (50000,), "imagenet-c": (5000, 500000)}


def experiments(dataset, ir):
    if ir is None:
        return {"standard": COMPONENTS}
    groups = {"online_ir": COMPONENTS}
    if dataset != "cifar100-c":
        groups["gate"] = GATE
    if ir in WINDOW_LEVELS[dataset]:
        groups["window"] = WINDOW
        groups["combination"] = COMBINATION
        if dataset != "cifar100-c":
            groups["mechanism"] = MECHANISM
    return groups
