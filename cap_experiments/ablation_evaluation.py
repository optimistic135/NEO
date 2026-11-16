"""Feature-shared PLEAD ablation scoring and label-isolated diagnostics."""

from collections import Counter
import math

import numpy as np
import torch
from tqdm import tqdm

from cap_experiments.evaluation import Metrics
from cap_experiments.ir_evaluation import Timings
from tta_library.plead_lite import DIAGNOSTIC_FIELDS, FeatureLogitWindow, PLEADLite, describe_step


def batches(subsets, size):
    for subset_id, indices in enumerate(subsets):
        for offset in range(0, len(indices), size):
            yield subset_id, np.asarray(indices[offset:offset + size])


def _extract_step(diag, audit, classes, labels, variant):
    step = describe_step(diag.tolist())
    fit_code = int(step["fit_code"])
    gamma_reason = None
    if step["gamma"] == 0:
        if not variant.components.debias:
            gamma_reason = "debias_disabled"
        elif fit_code:
            gamma_reason = "full_fit_failed"
        elif step["half0_code"] or step["half1_code"]:
            gamma_reason = "split_fit_failed"
        elif step["gate_code"]:
            gamma_reason = "nonfinite_gate"
        else:
            gamma_reason = "uncertainty_clipped"
    correction, composition = audit[:2].tolist()
    prior = audit[2:].numpy()
    truth = np.bincount(labels, minlength=classes) / len(labels)
    step.update(
        correction_norm=correction, class_composition_correction_norm=composition,
        class_prior_tv_error=float(np.abs(prior - truth).sum() * .5),
        gamma_zero_reason=gamma_reason,
        effective_debiasing=composition > 0 and step["lambda"] > 0 and not step["correction_code"],
        ridge_alpha=step["sxx"] * variant.ridge_ratio,
    )
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in step.items()}


@torch.no_grad()
def evaluate(subsets, validation, labels, cache, classifier, variants, args, description):
    """The adapter API consumes only arriving features and a window, with no label input."""
    names = tuple(variants)
    windows = {size: FeatureLogitWindow(classifier, size)
               for size in dict.fromkeys(v.window for v in variants.values())}
    models = {name: PLEADLite(windows[v.window], v.fit(), v.components) for name, v in variants.items()}
    timings = Timings(args.device)
    classes = classifier.out_features
    if timings.gpu:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    meters = {name: Metrics() for name in names}
    validation_meters = {name: Metrics() for name in names} if validation is not None else {}
    updates, subset_rows = [], []
    label_bank = np.empty(0, dtype=np.int64)
    max_window = max(windows)
    size = args.batch_size
    hist = Counter()
    eval_batches = sum((len(s) + size - 1) // size for s in subsets)
    arrivals = calls = val_arrivals = val_calls = 0
    it = tqdm(batches(subsets, size), total=eval_batches, desc=description,
              unit="batch", leave=False, mininterval=1)
    for update_idx, (subset_id, indices) in enumerate(it):
        features, misses = cache.arrive(indices, timings)
        calls += bool(misses)
        arrivals += misses
        raw = {}
        for window_size, window in windows.items():
            with timings.measure(f"window_{window_size}"):
                raw[window_size] = window.append(features, allow_oversize=True)
        output = []
        for name, variant in variants.items():
            with timings.measure((name, update_idx)):
                logits, diag, audit = models[name].update(
                    windows[variant.window], len(indices), batch_logits=raw[variant.window], audit=True,
                )
                output.append(torch.cat((logits.flatten(), diag, audit)))
        packed = torch.stack(output).cpu()
        n_logits = len(indices) * classes
        diag_end = n_logits + len(DIAGNOSTIC_FIELDS)
        actual = labels[indices]
        target = torch.from_numpy(actual)
        label_bank = np.concatenate((label_bank, actual))[-max_window:]
        end = update_idx + 1 == eval_batches or hist["samples"] + len(indices) == len(subsets[subset_id])
        hist["samples"] += len(indices)
        for name, packed_row in zip(names, packed):
            logits = packed_row[:n_logits].reshape(len(indices), classes)
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite logits in {description}/{name}/{update_idx}")
            meter = meters[name]
            before = meter.correct1
            meter.add(logits, target)
            correct = float(meter.correct1 - before)
            hist[name] += correct
            variant = variants[name]
            updates.append(dict(
                method=name, subset_id=subset_id, update_idx=update_idx, batch_size=len(indices),
                top1=100 * correct / len(indices),
                **_extract_step(packed_row[n_logits:diag_end], packed_row[diag_end:], classes,
                                label_bank[-variant.window:], variant),
            ))
        if end:
            for name in names:
                subset_rows.append(dict(method=name, subset_id=subset_id, samples=hist["samples"],
                                        top1=100 * hist[name] / hist["samples"]))
            hist.clear()

    if validation is not None:
        for _, indices in batches([validation], size):
            features, misses = cache.arrive(indices, timings)
            val_calls += bool(misses)
            val_arrivals += misses
            with timings.measure("validation_head"):
                window = next(iter(windows.values()))
                raw_logits = torch.nn.functional.linear(features, window.weight, window.bias)
            target = torch.from_numpy(labels[indices])
            output = []
            for name in names:
                with timings.measure(f"validation_{name}"):
                    output.append(raw_logits - models[name].q)
            for name, logits in zip(names, torch.stack(output).cpu()):
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Nonfinite validation logits in {description}/{name}")
                validation_meters[name].add(logits, target)
    times = timings.result()
    peak = torch.cuda.max_memory_allocated(args.device) / 1024**2 if timings.gpu else None
    for step in updates:
        step["elapsed_seconds"] = times[step["method"], step["update_idx"]]
    by_method = {name: [] for name in names}
    for step in updates:
        by_method[step["method"]].append(step)
    rows = []
    for name in names:
        variant = variants[name]
        records = by_method[name]
        primary = validation_meters[name].result() if validation is not None else meters[name].result()
        adaptation = meters[name].result()
        n = len(records)
        adapter = sum(s["elapsed_seconds"] for s in records) + times[f"window_{variant.window}"]
        shared = times["backbone"] + times["replay_read"] + times["window_build"] + times["validation_head"]
        elapsed = adapter + shared + times[f"validation_{name}"]
        total_images = adaptation["samples"] + (primary["samples"] if validation is not None else 0)
        kv = [s["kappa"] for s in records]
        rows.append(dict(
            method=name, window=variant.window, ridge_ratio=variant.ridge_ratio,
            n_samples=primary["samples"], top1=primary["top1"], top5=primary["top5"], ece=primary["ece"],
            adaptation_top1=adaptation["top1"], adaptation_samples=adaptation["samples"],
            fallback_rate=sum(bool(step["fallback_reasons"]) for step in records) / n,
            gamma_zero_rate=sum(step["gamma"] == 0 for step in records) / n,
            effective_debiasing_rate=sum(step["effective_debiasing"] for step in records) / n,
            class_prior_tv_error=float(np.mean([s["class_prior_tv_error"] for s in records])),
            kappa_mean=float(np.mean(kv)), kappa_update_std=float(np.std(kv, ddof=1)) if n > 1 else None,
            gamma_mean=float(np.mean([s["gamma"] for s in records])),
            lambda_mean=float(np.mean([s["lambda"] for s in records])),
            correction_norm_mean=float(np.mean([s["correction_norm"] for s in records])),
            class_composition_correction_norm_mean=float(np.mean([s["class_composition_correction_norm"] for s in records])),
            adapter_seconds=adapter, total_seconds=elapsed, throughput=total_images / max(elapsed, 1e-12),
            peak_gpu_mib=peak, backbone_forward_count=calls + val_calls, backbone_images=arrivals + val_arrivals,
            cache_read_seconds=times["replay_read"], backbone_seconds=times["backbone"],
            cache_scope="cold" if arrivals + val_arrivals == total_images else
                        "cross_configuration" if arrivals + val_arrivals == 0 else "mixed",
        ))
    return rows, updates, subset_rows
