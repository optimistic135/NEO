"""Single-view PLEAD and windowed NEO over cached frozen model outputs."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tta_library.plead_v2 import V2Config


FIT_REASONS = (
    "", "nonfinite_fit_input", "insufficient_neff", "nonfinite_fit_moments",
    "degenerate_sxx", "nonpositive_sxy", "nonfinite_correlation", "weak_correlation",
    "nonfinite_kappa",
)
GATE_REASONS = ("", "nonfinite_gate")
CORRECTION_REASONS = (
    "", "singleton_bank", "nonfinite_shrinkage", "nonfinite_correction",
    "nonfinite_previous_correction",
)
DIAGNOSTIC_FIELDS = (
    "bank_size", "kappa", "lambda", "gamma", "neff", "sxx", "correlation",
    "fit_code", "half0_code", "half1_code", "gate_code", "shrink_code", "correction_code",
)


def describe_step(values):
    step = dict(zip(DIAGNOSTIC_FIELDS, values))
    reasons = [FIT_REASONS[int(step["fit_code"])]]
    for index in range(2):
        reason = FIT_REASONS[int(step[f"half{index}_code"])]
        if reason:
            reasons.append(f"gate_half_{index}_{reason}")
    reasons.append(GATE_REASONS[int(step["gate_code"])])
    reasons.extend(CORRECTION_REASONS[int(step[name])] for name in ("shrink_code", "correction_code"))
    step["fallback_reasons"] = [reason for reason in reasons if reason]
    return step


def _fit(t, u, weights, cfg):
    """Three independent weighted fits in one device operation, with V2 check precedence."""
    mass = weights.sum(dim=1)
    neff = mass.square() / (weights.square().sum(dim=1) + cfg.eps)
    safe_mass = torch.where(mass > 0, mass, 1)
    x = t[None] - ((weights @ t) / safe_mass[:, None])[:, None]
    y = u[None] - ((weights @ u) / safe_mass[:, None])[:, None]
    weighted_x = weights[:, :, None] * x
    xx = (weighted_x * x).sum(dim=(1, 2))
    xy = (weighted_x * y).sum(dim=(1, 2))
    yy = (weights[:, :, None] * y.square()).sum(dim=(1, 2))
    corr = torch.where(yy > 0, xy / (xx.sqrt() * yy.sqrt()), 0)
    scale = xy / (xx + cfg.ridge_fraction * xx)
    valid_input = torch.isfinite(t).all() & torch.isfinite(u).all() & torch.isfinite(weights).all(dim=1)
    checks = (
        ~valid_input,
        neff < cfg.min_neff,
        ~(torch.isfinite(xx) & torch.isfinite(xy) & torch.isfinite(yy)),
        xx <= cfg.energy_floor * neff,
        xy <= 0,
        ~torch.isfinite(corr),
        corr < cfg.min_correlation,
        ~torch.isfinite(scale),
    )
    code = torch.zeros_like(neff)
    for index, failed in reversed(tuple(enumerate(checks, 1))):
        code = torch.where(failed, index, code)
    return torch.where(code == 0, scale, 0), code, neff, xx, corr


class FeatureLogitWindow:
    def __init__(self, classifier, capacity):
        if capacity < 1 or not isinstance(classifier, torch.nn.Linear) or classifier.out_features < 2:
            raise ValueError("Expected a positive capacity and a linear classifier with at least two classes")
        self.capacity = capacity
        self.weight = classifier.weight.detach().float()
        self.bias = None if classifier.bias is None else classifier.bias.detach().float()
        self.centered = self.weight - self.weight.mean(dim=0, keepdim=True)
        self.features = self.weight.new_empty((0, self.weight.shape[1]))
        self.logits = self.weight.new_empty((0, self.weight.shape[0]))
        self.projected = self.logits

    @torch.no_grad()
    def append(self, features, *, allow_oversize=False):
        if (features.ndim != 2 or features.shape[1] != self.weight.shape[1]
                or not len(features) or (len(features) > self.capacity and not allow_oversize)
                or features.device != self.weight.device):
            raise ValueError("Invalid feature shape, device or batch larger than FIFO capacity")
        features = features.detach().float()
        logits = F.linear(features, self.weight, self.bias)
        retained = features[-self.capacity:]
        projected = retained @ self.centered.T
        tail = self.capacity - len(retained)
        start = max(0, len(self.features) - tail)
        self.features = torch.cat((self.features[start:], retained))
        self.logits = torch.cat((self.logits[start:], logits[-self.capacity:]))
        self.projected = torch.cat((self.projected[start:], projected))
        return logits

    def newest_logits(self, count):
        if not 0 < count <= len(self.logits):
            raise ValueError("Invalid size for most recent batch")
        return self.logits[-count:]


class NEOOnline:
    @torch.no_grad()
    def update(self, window, count):
        # Distribute the original head(h - mean_h) through the frozen affine layer.
        return window.newest_logits(count) - F.linear(window.features.mean(dim=0), window.weight)


@dataclass(frozen=True)
class PLEADComponents:
    debias: bool = True
    weight: bool = True
    gate: bool = True
    shrink: bool = True
    hard_prior: bool = False


class PLEADLite:
    def __init__(self, window, config=V2Config(), components=PLEADComponents()):
        self.q = window.weight.new_zeros(window.weight.shape[0])
        self.cfg = config
        self.components = components

    @torch.no_grad()
    def update(self, window, count, *, batch_logits=None, audit=False):
        raw = window.newest_logits(count) if batch_logits is None else batch_logits
        u = window.projected
        n = len(u)
        probabilities = (window.logits - self.q).softmax(dim=1)
        top = probabilities.topk(2, dim=1).values
        margin = top[:, 0] - top[:, 1]
        reliability = (margin / (margin + self.cfg.margin_offset)).clamp(0, 1)
        if not self.components.weight:
            reliability = torch.ones_like(reliability)
        if self.components.hard_prior:
            probabilities = F.one_hot(probabilities.argmax(dim=1), probabilities.shape[1]).to(probabilities)
        t = (probabilities @ window.centered) @ window.centered.T
        parity = torch.arange(n, device=u.device) % 2
        weights = torch.stack((reliability, reliability * (parity == 0), reliability * (parity == 1)))
        scales, codes, neff, sxx, corr = _fit(t, u, weights, self.cfg)
        kappa = scales[0]
        b_hat = kappa * t.mean(dim=0)
        b1 = scales[1] * t[::2].mean(dim=0)
        b2 = scales[2] * t[1::2].mean(dim=0)
        uncertainty = (b1 - b2).square().sum() / 4
        signal = b_hat.square().sum()
        full_ok = codes[0] == 0
        first_half_ok = full_ok & (codes[1] == 0)
        halves_ok = first_half_ok & (codes[2] == 0)
        gate_ok = torch.isfinite(uncertainty) & torch.isfinite(signal)
        gamma = torch.where(
            halves_ok & gate_ok, (1 - uncertainty / (signal + self.cfg.eps)).clamp(0, 1), 0,
        )
        if not self.components.gate:
            gamma = full_ok.to(gamma)
        if not self.components.debias:
            gamma = torch.zeros_like(gamma)
        applied = gamma * kappa
        residual = torch.where(applied > 0, u - applied * t, u)
        q_raw = residual.mean(dim=0)
        variance = (residual - q_raw).square().sum() / max(1, n * (n - 1))
        energy = q_raw.square().sum()
        shrink_ok = torch.isfinite(variance) & torch.isfinite(energy)
        shrinkage = torch.where(
            shrink_ok & (n > 1), (1 - variance / (energy + self.cfg.eps)).clamp(0, 1), 0,
        )
        if not self.components.shrink:
            shrinkage = torch.ones_like(shrinkage)
        candidate = shrinkage * q_raw
        usable = torch.isfinite(candidate).all() & torch.isfinite(raw - candidate).all()
        previous_usable = torch.isfinite(raw - self.q).all()
        self.q = torch.where(usable, candidate, torch.where(previous_usable, self.q, 0))
        zero = kappa.new_zeros(())
        shrink_code = (zero + 1) if n == 1 else torch.where(shrink_ok, zero, 2)
        correction_code = torch.where(usable, zero, torch.where(previous_usable, 3, 4))
        # Undefined fit statistics remain explicitly absent when decoded by the evaluator.
        missing = kappa.new_full((), float("nan"))
        stats_valid = (codes[0] != 1) & (codes[0] != 2)
        correlation_valid = stats_valid & ((codes[0] == 0) | (codes[0] >= 6))
        diagnostics = torch.stack((
            zero + n, kappa, shrinkage, gamma, torch.where(codes[0] != 1, neff[0], missing),
            torch.where(stats_valid, sxx[0], missing), torch.where(correlation_valid, corr[0], missing),
            codes[0], torch.where(full_ok & self.components.gate, codes[1], zero),
            torch.where(first_half_ok & self.components.gate, codes[2], zero),
            (halves_ok & ~gate_ok & self.components.gate).float(),
            shrink_code if self.components.shrink else zero, correction_code,
        ))
        if audit:
            audit_values = torch.cat((
                self.q.norm().reshape(1),
                torch.where(applied > 0, (gamma * b_hat).norm(), zero).reshape(1),
                probabilities.mean(dim=0),
            ))
            return raw - self.q, diagnostics, audit_values
        return raw - self.q, diagnostics
