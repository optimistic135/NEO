"""Sparse-aware PLEAD with optional cross-view weights and split-bank gating."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from tta_library.cap_neo import class_distribution


ABLATION_METHODS = (
    "plead_original", "plead_v2_fit", "plead_v2_fit_cv", "plead_v2_fit_gate", "plead_v2",
)
V2_OPTIONS = {
    "plead_v2_fit": (False, False),
    "plead_v2_fit_cv": (True, False),
    "plead_v2_fit_gate": (False, True),
    "plead_v2": (True, True),
}


@dataclass(frozen=True)
class V2Config:
    max_bank: int = 512
    ridge_fraction: float = 0.01
    margin_offset: float = 0.05
    min_neff: float = 16
    energy_floor: float = 1e-8
    min_correlation: float = 0.05
    eps: float = 1e-8
    augmentation: str = "horizontal_flip_after_preprocessing"


def weighted_ridge(t, u, weights, cfg):
    stats = dict(neff=None, sxx=None, sxy=None, correlation=None)
    if not (torch.isfinite(t).all() and torch.isfinite(u).all() and torch.isfinite(weights).all()):
        return 0.0, "nonfinite_fit_input", stats

    mass = weights.sum()
    neff = float((mass.square() / (weights.square().sum() + cfg.eps)).item())
    stats["neff"] = neff
    if neff < cfg.min_neff:
        return 0.0, "insufficient_neff", stats

    w = weights[:, None]
    x = t - (w * t).sum(dim=0) / mass
    y = u - (w * u).sum(dim=0) / mass
    xx, xy, yy = ((w * value).sum().item() for value in (x.square(), x * y, y.square()))
    stats.update(sxx=_finite(xx), sxy=_finite(xy))
    if not all(math.isfinite(value) for value in (xx, xy, yy)):
        return 0.0, "nonfinite_fit_moments", stats
    if xx <= cfg.energy_floor * neff:
        return 0.0, "degenerate_sxx", stats
    if xy <= 0:
        return 0.0, "nonpositive_sxy", stats
    correlation = xy / (math.sqrt(xx) * math.sqrt(yy)) if yy > 0 else 0.0
    stats["correlation"] = _finite(correlation)
    if not math.isfinite(correlation):
        return 0.0, "nonfinite_correlation", stats
    if correlation < cfg.min_correlation:
        return 0.0, "weak_correlation", stats
    kappa = xy / (xx + cfg.ridge_fraction * xx)
    if not math.isfinite(kappa):
        return 0.0, "nonfinite_kappa", stats
    return kappa, None, stats


def _finite(value):
    return value if math.isfinite(value) else None


class PLEADV2:
    def __init__(self, classifier, method, config=V2Config()):
        self.cross_view, self.use_gate = V2_OPTIONS[method]
        self.config = config
        self.weight = classifier.weight.detach().float()
        self.bias = None if classifier.bias is None else classifier.bias.detach().float()
        self.a = self.weight - self.weight.mean(dim=0, keepdim=True)
        self.q = self.weight.new_zeros(len(self.weight))
        self.a_logits = self.weight.new_empty((0, len(self.weight)))
        self.a_projected = self.a_logits
        self.b_logits = self.a_logits
        self.b_projected = self.a_logits
        self.history = []

    def _raw(self, a):
        if a.ndim != 2 or a.shape[1] != self.weight.shape[1] or not len(a):
            raise ValueError("Expected nonempty [batch, feature] input to PLEAD-V2")
        return F.linear(a.detach().float(), self.weight, self.bias)

    @torch.no_grad()
    def predict(self, a):
        return self._raw(a) - self.q

    @torch.no_grad()
    def update(self, a, b=None):
        if self.cross_view != (b is not None) or (b is not None and a.shape != b.shape):
            raise ValueError("View B must match A and is used only by cross-view variants")
        raw = self._raw(a)
        projected = a.detach().float() @ self.a.T
        raw_b = self._raw(b) if self.cross_view else raw
        projected_b = b.detach().float() @ self.a.T if self.cross_view else projected
        if not torch.isfinite(raw).all() or not torch.isfinite(projected).all():
            raise ValueError("Nonfinite classifier output or feature projection")

        limit = self.config.max_bank
        self.a_logits = torch.cat((self.a_logits, raw))[-limit:]
        self.a_projected = torch.cat((self.a_projected, projected))[-limit:]
        if self.cross_view:
            self.b_logits = torch.cat((self.b_logits, raw_b))[-limit:]
            self.b_projected = torch.cat((self.b_projected, projected_b))[-limit:]
        a_prob = (self.a_logits - self.q).softmax(dim=1)
        b_prob = (self.b_logits - self.q).softmax(dim=1) if self.cross_view else a_prob
        u = self.b_projected if self.cross_view else self.a_projected
        top = a_prob.topk(2, dim=1).values
        margin = top[:, 0] - top[:, 1]
        consistency = 1 - 0.5 * (a_prob - b_prob).abs().sum(dim=1)
        weights = (margin / (margin + self.config.margin_offset) * consistency).clamp(0, 1)
        t = (a_prob @ self.a) @ self.a.T
        kappa, reason, stats = weighted_ridge(t, u, weights, self.config)

        reasons = [reason] if reason else []
        gamma = 0.0 if self.use_gate and reason else 1.0
        gate_reason = None
        uncertainty = None
        if kappa and self.use_gate:
            estimates = []
            for index in range(2):
                half = slice(index, None, 2)
                scale, half_reason, _ = weighted_ridge(t[half], u[half], weights[half], self.config)
                if half_reason:
                    gate_reason = f"gate_half_{index}_{half_reason}"
                    break
                estimates.append(scale * t[half].mean(dim=0))
            if gate_reason:
                gamma = 0.0
                reasons.append(gate_reason)
            else:
                uncertainty = (estimates[0] - estimates[1]).square().sum().item() / 4
                signal = (kappa * t.mean(dim=0)).square().sum().item()
                if math.isfinite(uncertainty) and math.isfinite(signal):
                    gamma = max(0.0, min(1.0, 1 - uncertainty / (signal + self.config.eps)))
                else:
                    gamma = 0.0
                    gate_reason = "nonfinite_gate"
                    reasons.append(gate_reason)

        # Apply the same scalar to every per-sample prior before estimating shrinkage.
        residuals = self.a_projected
        if kappa and gamma:
            residuals = residuals - (gamma * kappa) * t
        q_raw = residuals.mean(dim=0)
        n = len(residuals)
        shrinkage = 0.0
        if n == 1:
            reasons.append("singleton_bank")
        else:
            variance = (residuals - q_raw).square().sum().item() / (n * (n - 1))
            signal = q_raw.square().sum().item()
            if math.isfinite(variance) and math.isfinite(signal):
                shrinkage = max(0.0, min(1.0, 1 - variance / (signal + self.config.eps)))
            else:
                reasons.append("nonfinite_shrinkage")
        candidate = shrinkage * q_raw
        if not torch.isfinite(candidate).all() or not torch.isfinite(raw - candidate).all():
            reasons.append("nonfinite_correction")
            candidate = self.q
        if not torch.isfinite(raw - candidate).all():
            reasons.append("nonfinite_previous_correction")
            candidate = torch.zeros_like(self.q)
        self.q = candidate
        distribution = (
            class_distribution(a_prob, a_prob.argmax(dim=1)) if torch.isfinite(a_prob).all()
            else {"predicted_prior": None, "predicted_argmax_prior": None}
        )
        self.history.append({
            "bank_size": n, "kappa": kappa, "gamma": gamma, "lambda": shrinkage,
            "correction_norm": _finite(self.q.norm().item()), "feature_shift_norm": None,
            "fallback_reasons": reasons, "fit_fallback_reason": reason, "gate_fallback_reason": gate_reason,
            "gate_uncertainty": _finite(uncertainty) if uncertainty is not None else None,
            **stats, **distribution,
        })
        return raw - self.q
