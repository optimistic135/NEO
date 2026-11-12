"""One frozen backbone forward per batch, shared across selected methods."""

from collections import Counter
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from calibration_library.metrics import ECELoss
from tta_library.cap_neo import CAPConfig, CAPNEO, class_distribution
from utils.cli_utils import accuracy
from utils.utils import calculate_entropy


MIB = 1024 ** 2


class Measurements:
    def __init__(self, device):
        self.device = device
        self.seconds = 0.0
        self.peak_gpu_mib = None
        self.peak_extra_gpu_mib = None

    def measure(self, operation):
        gpu = self.device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(self.device)
            start_memory = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        result = operation()
        if gpu:
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            self.peak_gpu_mib = max(self.peak_gpu_mib or 0, peak / MIB)
            self.peak_extra_gpu_mib = max(self.peak_extra_gpu_mib or 0, (peak - start_memory) / MIB)
        self.seconds += perf_counter() - start
        return result


class Metrics:
    def __init__(self):
        self.logits = []
        self.targets = []
        self.entropy = 0.0
        self.correct1 = 0.0
        self.correct5 = 0.0
        self.count = 0

    def add(self, logits, targets):
        if not len(targets):
            return
        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        self.correct1 += float(top1.item()) * len(targets) / 100
        self.correct5 += float(top5.item()) * len(targets) / 100
        self.entropy += float(calculate_entropy(logits).sum().item())
        self.count += len(targets)
        self.logits.append(logits)
        self.targets.append(targets)

    def result(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.targets)
        return {
            "samples": self.count,
            "top1": 100 * self.correct1 / self.count,
            "top5": 100 * self.correct5 / self.count,
            "entropy": self.entropy / self.count,
            "ece": float(ECELoss().loss(logits.numpy(), labels.numpy(), n_bins=15, logits=True)),
        }


class BaselineDiagnostics:
    def __init__(self, classifier):
        self.weight = classifier.weight.detach().float()
        self.centered_weight = self.weight - self.weight.mean(dim=0, keepdim=True)
        self.classifier = classifier
        self.raw_logits = []
        self.history = []

    def record(self, features, previous_center, new_center):
        self.raw_logits.append(self.classifier(features))
        raw = torch.cat(self.raw_logits)
        prior = (raw - self.centered_weight @ previous_center).softmax(dim=1)
        self.history.append({
            "bank_size": len(raw),
            "kappa": 0.0,
            "lambda": 1.0,
            "correction_norm": float((self.centered_weight @ new_center).norm().item()),
            "feature_shift_norm": float(new_center.norm().item()),
            "fallback_reasons": [],
            **class_distribution(prior, prior.argmax(dim=1)),
        })


def _loader(dataset, indices, args, seed):
    return DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def _maximum(*values):
    return max((value for value in values if value is not None), default=None)


@torch.no_grad()
def evaluate_stream(dataset, stream, extractor, neo, methods, cfg, args, seed, description):
    start = perf_counter()
    neo.reset()
    head = neo.classifier
    neo.corrupt_class_center = neo.corrupt_class_center.to(head.weight.device)
    caps = {method: CAPNEO(head, method, cfg) for method in methods if method != "neo"}
    baseline = BaselineDiagnostics(head) if "neo" in methods else None
    parity = (
        CAPNEO(head, "cap_pa", CAPConfig(min_fit_samples=len(stream.adaptation) + 1))
        if args.check_equivalence and "neo" in methods else None
    )
    timings = {method: Measurements(args.device) for method in methods}
    shared = Measurements(args.device)
    phases = ["adaptation", *stream.validation]
    metrics = {method: {phase: Metrics() for phase in phases} for method in methods}

    validation = list(dict.fromkeys(index for indices in stream.validation.values() for index in indices))
    phase_members = {
        phase: np.isin(validation, indices) for phase, indices in stream.validation.items()
    }
    for adapting, indices in ((True, stream.adaptation), (False, validation)):
        phase = "adaptation" if adapting else "validation"
        loader = _loader(dataset, indices, args, seed)
        progress = tqdm(loader, desc=f"{description} {phase}", unit="batch", leave=False,
                        disable=args.no_progress)
        offset = 0
        for images, labels in progress:
            features = shared.measure(
                lambda: extractor(images.to(args.device, non_blocking=True)).float()
            )
            if features.ndim != 2 or features.shape[1] != head.in_features:
                raise ValueError("Unexpected ViT feature shape")
            if not torch.isfinite(features).all():
                raise ValueError("ViT produced nonfinite features")
            for method in methods:
                if method == "neo":
                    previous = neo.corrupt_class_center
                    logits = timings[method].measure(lambda: neo(features, adapt=adapting))
                    if adapting:
                        baseline.record(features, previous, neo.corrupt_class_center)
                    if parity is not None:
                        projected = parity.update(features) if adapting else parity.predict(features)
                        torch.testing.assert_close(
                            projected.softmax(dim=1), logits.softmax(dim=1), atol=2e-6, rtol=2e-5
                        )
                elif not adapting:
                    logits = timings[method].measure(lambda: caps[method].predict(features))
                elif method == "cap_oracle_labels":
                    logits = timings[method].measure(
                        lambda: caps[method].update(features, oracle_labels=labels.to(args.device))
                    )
                else:
                    logits = timings[method].measure(lambda: caps[method].update(features))
                if logits.shape != (len(labels), head.out_features) or not torch.isfinite(logits).all():
                    raise ValueError(f"{method} returned invalid logits")
                logits = logits.cpu()
                if adapting:
                    metrics[method]["adaptation"].add(logits, labels)
                else:
                    for view, membership in phase_members.items():
                        mask = membership[offset:offset + len(labels)]
                        metrics[method][view].add(logits[mask], labels[mask])
            offset += len(labels)
        if offset != len(indices):
            raise RuntimeError(f"Unexpected number of samples in {phase}")

    rows, histories = [], {}
    for method in methods:
        history = baseline.history if method == "neo" else caps[method].history
        histories[method] = history
        fallbacks = Counter(reason for step in history for reason in step["fallback_reasons"])
        final = history[-1]
        timing = timings[method]
        row = {
            "method": method,
            "oracle": method == "cap_oracle_labels",
            "kappa": final["kappa"],
            "lambda": final["lambda"],
            "correction_norm": final["correction_norm"],
            "feature_shift_norm": final["feature_shift_norm"],
            "predicted_prior": final["predicted_prior"],
            "predicted_argmax_prior": final["predicted_argmax_prior"],
            "fallback_updates": sum(bool(step["fallback_reasons"]) for step in history),
            "fallback_counts": dict(fallbacks),
            "shared_backbone_seconds": shared.seconds,
            "method_seconds": timing.seconds,
            "inference_seconds": shared.seconds + timing.seconds,
            "peak_gpu_mib": _maximum(shared.peak_gpu_mib, timing.peak_gpu_mib),
            "method_peak_extra_gpu_mib": timing.peak_extra_gpu_mib,
        }
        for phase, meter in metrics[method].items():
            row.update({f"{phase}_{key}": value for key, value in meter.result().items()})
        rows.append(row)
    return rows, histories, perf_counter() - start
