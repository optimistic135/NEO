"""Single-view, unweighted classifier-geometry correction with prior evidence."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


MIN_BANK = 4
REASONS = ("", "insufficient_bank", "nonfinite_fit", "degenerate_sxx", "nonfinite_update")
FIELDS = ("bank_size", "kappa", "d", "v", "gamma", "a_norm", "b_norm", "q_norm", "sxx", "sxy", "suu", "fallback_code")


@dataclass(frozen=True)
class Variant:
    window: int = 128
    hard: bool = False
    norm_only: bool = False
    fixed_gamma: float | None = None
    batch_only: bool = False


def prior_evidence(probabilities):
    prior = probabilities.mean(dim=0)
    d = (prior - 1 / probabilities.shape[1]).square().sum()
    v = (probabilities[::2].mean(dim=0) - probabilities[1::2].mean(dim=0)).square().sum() / 4
    gamma = (1 - v / (d + torch.finfo(probabilities.dtype).eps)).clamp(0, 1)
    return prior, d, v, gamma


class PLEAD:
    """Consume a window of frozen features and logits. Labels never enter the adapter."""

    def __init__(self, window, variant=Variant()):
        self.q = window.weight.new_zeros(window.weight.shape[0])
        self.variant = variant

    @torch.no_grad()
    def update(self, window, raw):
        n = len(raw) if self.variant.batch_only else len(window.logits)
        u, bank_logits = window.projected[-n:], window.logits[-n:]
        soft = (bank_logits - self.q).softmax(dim=1)
        probabilities = (F.one_hot(soft.argmax(dim=1), soft.shape[1]).to(soft)
                         if self.variant.hard else soft)
        if n < MIN_BANK:
            diag = self.q.new_zeros(len(FIELDS))
            diag[0], diag[7], diag[-1] = n, self.q.norm(), 1
            return raw - self.q, diag, probabilities.mean(dim=0), soft.mean(dim=0)
        prior, d, v, gamma = prior_evidence(probabilities)
        t = (probabilities @ window.centered) @ window.centered.T
        a = u.mean(dim=0)
        t_mean = t.mean(dim=0)
        x, y = t - t_mean, u - a
        sxx, sxy, suu = x.square().sum(), (x * y).sum(), y.square().sum()
        eps = torch.finfo(u.dtype).eps
        # Centering noise scales with the dimensionless relative precision and prototype energy.
        floor = (eps**2 * t.square().sum()).clamp_min(torch.finfo(u.dtype).tiny)
        safe_sxx = sxx.clamp_min(floor)
        if self.variant.norm_only:
            kappa = (suu / (safe_sxx + floor)).sqrt()
        else:
            kappa = (sxy / safe_sxx).clamp_min(0)
        if self.variant.fixed_gamma is not None:
            gamma = gamma.new_full((), self.variant.fixed_gamma)
        b = kappa * t_mean
        candidate = a - gamma * b
        fit = torch.stack((sxx, sxy, suu, kappa, d, v))
        finite_update = torch.isfinite(candidate).all() & torch.isfinite(raw - candidate).all()
        code = torch.where(
            ~torch.isfinite(fit).all(), 2,
            torch.where(sxx <= floor, 3, torch.where(~finite_update, 4, 0)),
        )
        self.q = torch.where(code == 0, candidate, self.q)
        zero = kappa.new_zeros(())
        valid = code == 0
        diag = torch.stack((
            zero + n, torch.where(valid, kappa, zero), d, v, torch.where(valid, gamma, zero),
            a.norm(), torch.where(valid, b.norm(), zero), self.q.norm(), sxx, sxy, suu, code,
        ))
        return raw - self.q, diag, prior, soft.mean(dim=0)
