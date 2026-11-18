import torch

from src.methods.base import Adapter
from tta_library.plead_lite import FeatureLogitWindow
from tta_library.plead_simplified import PLEAD as Corrector, FIELDS, REASONS


class PLEAD(Adapter):
    def __init__(self, model, variant):
        super().__init__(model)
        self.variant = variant
        self.reset()

    def reset(self):
        super().reset()
        self.window = FeatureLogitWindow(self.model.head, self.variant.window)
        self.corrector = Corrector(self.window, self.variant)
        self.events = {}

    def predict(self, images):
        return super().predict(images) - self.corrector.q

    @torch.no_grad()
    def update(self, images):
        raw = self.window.append(self.features(images), allow_oversize=True)
        logits, step, _, _ = self.corrector.update(self.window, raw)
        fields = dict(zip(FIELDS, step.tolist()))
        reason = REASONS[int(fields["fallback_code"])]
        self.events = dict(kappa=fields["kappa"], gamma=fields["gamma"], valid_fit=int(not reason), fallback=reason)
        return logits
