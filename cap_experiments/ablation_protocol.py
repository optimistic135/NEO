"""PLEAD component and estimation profiles, paired on existing streams."""

from dataclasses import dataclass, replace

from tta_library.plead_lite import PLEADComponents
from tta_library.plead_v2 import V2Config


IR_LEVELS = {
    "cifar10-c": (1, 50, 5000),
    "cifar100-c": (1, 500, 50000),
    "imagenet-c": (1, 5000, 500000),
}
VALIDATION_PER_CLASS = {"cifar10-c": 50, "cifar100-c": 5, "imagenet-c": 1}
TARGETS = {("cifar10-c", 50), ("imagenet-c", 5000)}


@dataclass(frozen=True)
class Variant:
    window: int = 128
    ridge_ratio: float = 0.01
    components: PLEADComponents = PLEADComponents()

    def fit(self):
        return V2Config(max_bank=self.window, ridge_fraction=self.ridge_ratio)


FULL = Variant()
COMPONENTS = {
    "plead": FULL,
    "plead_no_debias": replace(FULL, components=PLEADComponents(debias=False)),
    "plead_no_weight": replace(FULL, components=PLEADComponents(weight=False)),
    "plead_no_gate": replace(FULL, components=PLEADComponents(gate=False)),
    "plead_no_shrink": replace(FULL, components=PLEADComponents(shrink=False)),
    "plead_no_ridge": replace(FULL, ridge_ratio=0),
}
VARIANTS = {
    **COMPONENTS,
    "hard_prior": replace(FULL, components=PLEADComponents(hard_prior=True)),
    **{f"window_{n}": replace(FULL, window=n) for n in (32, 64, 256)},
    **{f"ridge_{ratio:g}": replace(FULL, ridge_ratio=ratio) for ratio in (0.001, 0.1)},
}
ESTIMATION = ("plead", "hard_prior", "plead_no_weight", "plead_no_ridge")
WINDOW_SWEEP = ("window_32", "window_64", "plead", "window_256")
RIDGE_SWEEP = ("plead_no_ridge", "ridge_0.001", "plead", "ridge_0.1")


def experiments(dataset, ir):
    if ir is None:
        return {"standard": tuple(COMPONENTS)}
    tasks = {"online_ir": tuple(COMPONENTS)}
    if (dataset, ir) in TARGETS:
        tasks["estimation"] = ESTIMATION
        tasks["mechanism"] = ("plead",)
    if (dataset, ir) == ("imagenet-c", 5000):
        tasks["sensitivity"] = tuple(dict.fromkeys((*WINDOW_SWEEP, *RIDGE_SWEEP)))
    return tasks
