"""Classifier-aware centering and shrinkage on frozen ViT features."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


METHODS = ("neo", "neo_shrink", "cap_pa", "cap_neo", "cap_oracle_labels")


@dataclass(frozen=True)
class CAPConfig:
    confidence: float = 0.7
    min_fit_samples: int = 16
    min_per_class: int = 2
    min_coverage: int = 2
    min_alignment: float = 0.05
    eps: float = 1e-8

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not 0 <= self.min_alignment <= 1:
            raise ValueError("Confidence and alignment thresholds must be in [0, 1]")
        if self.min_fit_samples < 2 or self.min_per_class < 1 or self.min_coverage < 2:
            raise ValueError("Require at least two fit samples and two covered classes")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")


def class_distribution(probabilities, assignments):
    classes = probabilities.shape[1]
    return {
        "predicted_prior": probabilities.mean(dim=0).tolist(),
        "predicted_argmax_prior": (
            torch.bincount(assignments, minlength=classes) / len(assignments)
        ).tolist(),
    }


def _fit_scale(features, expected_prototypes, assignments, confident, cfg):
    classes, counts = assignments[confident].unique(return_counts=True)
    supported = classes[counts >= cfg.min_per_class]
    selected = confident & (assignments[:, None] == supported).any(dim=1)
    fit_count = int(selected.sum().item())
    coverage = len(supported)
    stats = {"fit_count": fit_count, "covered_classes": coverage, "alignment": None}
    if int(confident.sum().item()) < cfg.min_fit_samples:
        return 0.0, "low_confidence", stats
    if coverage < cfg.min_coverage:
        return 0.0, "insufficient_class_coverage", stats
    if fit_count < cfg.min_fit_samples:
        return 0.0, "insufficient_supported_samples", stats

    x = expected_prototypes[selected]
    y = features[selected]
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    xx = x.square().sum()
    yy = y.square().sum()
    xy = (x * y).sum()
    if not torch.isfinite(torch.stack((xx, yy, xy))).all():
        return 0.0, "nonfinite_fit", stats
    if min(xx.item(), yy.item()) <= cfg.eps * fit_count:
        return 0.0, "degenerate_fit", stats
    alignment = float((xy / xx.sqrt() / yy.sqrt()).item())
    stats["alignment"] = alignment
    if xy.item() <= 0:
        return 0.0, "nonpositive_scale", stats
    if alignment < cfg.min_alignment:
        return 0.0, "weak_alignment", stats
    scale = float((xy / xx).item())
    if not math.isfinite(scale):
        return 0.0, "nonfinite_fit", stats
    return scale, None, stats


class CAPNEO:
    """Cumulative bank per stream, with a frozen predict path for validation."""

    def __init__(self, classifier, method="cap_neo", config=None):
        if method not in METHODS[1:]:
            raise ValueError(f"Unknown CAP method: {method}")
        if not isinstance(classifier, torch.nn.Linear):
            raise TypeError("CAP-NEO requires a linear classification head")
        self.method = method
        self.config = config if config is not None else CAPConfig()
        self.weight = classifier.weight.detach().float()
        self.bias = None if classifier.bias is None else classifier.bias.detach().float()
        self.classes, self.width = self.weight.shape
        if self.classes < 2 or self.width < 1:
            raise ValueError("The head must have at least two classes and one feature")
        if not torch.isfinite(self.weight).all() or (
            self.bias is not None and not torch.isfinite(self.bias).all()
        ):
            raise ValueError("Nonfinite classification head")
        self.prototypes = self.weight - self.weight.mean(dim=0, keepdim=True)
        self.use_prior = method != "neo_shrink"
        self.use_shrinkage = method != "cap_pa"
        self.oracle = method == "cap_oracle_labels"
        self.reset()

    def reset(self):
        self.features = self.weight.new_empty((0, self.width))
        self.raw_logits = self.weight.new_empty((0, self.classes))
        self.label_bank = (
            torch.empty(0, dtype=torch.long, device=self.weight.device) if self.oracle else None
        )
        self.q = self.weight.new_zeros(self.classes)
        self.history = []

    def _raw(self, features):
        if features.ndim != 2 or features.shape[1] != self.width or not len(features):
            raise ValueError(f"Expected nonempty [N, {self.width}] features")
        if features.device != self.weight.device or not features.is_floating_point():
            raise ValueError("Features must be floating point on the classifier device")
        features = features.detach().float()
        logits = F.linear(features, self.weight, self.bias)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite features or uncorrected logits")
        return features, logits

    @torch.no_grad()
    def predict(self, features):
        _, logits = self._raw(features)
        return logits - self.q

    @torch.no_grad()
    def update(self, features, *, oracle_labels=None):
        features, logits = self._raw(features)
        if self.oracle:
            if (
                oracle_labels is None or oracle_labels.shape != (len(features),)
                or oracle_labels.dtype != torch.long or oracle_labels.device != features.device
                or not ((oracle_labels >= 0) & (oracle_labels < self.classes)).all()
            ):
                raise ValueError("Oracle requires valid adaptation labels on the feature device")
        elif oracle_labels is not None:
            raise ValueError("Unsupervised methods cannot receive labels")

        bank = torch.cat((self.features, features))
        raw_bank = torch.cat((self.raw_logits, logits))
        labels = torch.cat((self.label_bank, oracle_labels)) if self.oracle else None
        previous = self.q
        probabilities = (raw_bank - previous).softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        probabilities_valid = bool(torch.isfinite(probabilities).all())
        confident = confidence >= self.config.confidence

        scale, prior_reason = 0.0, None
        fit = {"fit_count": 0, "covered_classes": 0, "alignment": None}
        residuals = bank
        if self.use_prior:
            if probabilities_valid or self.oracle:
                if self.oracle:
                    priors = F.one_hot(labels, num_classes=self.classes).to(bank.dtype)
                    assignments = labels
                    eligible = torch.ones_like(confident)
                else:
                    priors, assignments, eligible = probabilities, predicted, confident
                expected = priors @ self.prototypes
                scale, prior_reason, fit = _fit_scale(
                    bank, expected, assignments, eligible, self.config
                )
                if scale:
                    residuals = bank - scale * expected
            else:
                prior_reason = "nonfinite_probabilities"

        projected = residuals @ self.prototypes.T
        q_hat = projected.mean(dim=0)
        n = len(bank)
        variance = None
        shrinkage = 1.0
        shrink_reason = None
        if self.use_shrinkage:
            if n == 1:
                shrinkage, shrink_reason = 0.0, "singleton_bank"
            else:
                variance = float(((projected - q_hat).square().sum() / (n * (n - 1))).item())
                signal = float(q_hat.square().sum().item())
                if math.isfinite(variance) and math.isfinite(signal):
                    shrinkage = max(0.0, min(1.0, 1 - variance / (signal + self.config.eps)))
                else:
                    shrinkage, shrink_reason = 0.0, "nonfinite_shrinkage"
        candidate = shrinkage * q_hat
        correction_reason = None
        if not torch.isfinite(candidate).all() or not torch.isfinite(logits - candidate).all():
            candidate = previous
            correction_reason = "nonfinite_correction"
        corrected = logits - candidate
        if not torch.isfinite(corrected).all():
            candidate = torch.zeros_like(previous)
            corrected = logits
            correction_reason = "nonfinite_previous_correction"

        self.features, self.raw_logits, self.label_bank, self.q = bank, raw_bank, labels, candidate
        distribution = (
            class_distribution(probabilities, predicted) if probabilities_valid
            else {"predicted_prior": None, "predicted_argmax_prior": None}
        )
        reasons = [reason for reason in (prior_reason, shrink_reason, correction_reason) if reason]
        self.history.append({
            "bank_size": n,
            "kappa": scale,
            "lambda": shrinkage,
            "correction_norm": self._finite_scalar(self.q.norm()),
            "unshrunk_correction_norm": self._finite_scalar(q_hat.norm()),
            "feature_shift_norm": self._finite_scalar(residuals.mean(dim=0).norm()),
            "variance": variance if variance is None or math.isfinite(variance) else None,
            "confidence_fraction": float(confident.float().mean().item()) if probabilities_valid else None,
            "fallback_reasons": reasons,
            **fit,
            **distribution,
        })
        return corrected

    @staticmethod
    def _finite_scalar(tensor):
        value = float(tensor.item())
        return value if math.isfinite(value) else None
