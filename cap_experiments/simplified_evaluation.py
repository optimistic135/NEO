"""Paired batch-transductive scoring with label audits outside the adapters."""

from collections import Counter
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from cap_experiments.ablation_evaluation import batches
from cap_experiments.evaluation import Metrics
from cap_experiments.ir_evaluation import Timings
from tta_library.plead_lite import FeatureLogitWindow
from tta_library.plead_simplified import FIELDS, PLEAD, REASONS


def audit_step(values, prior, labels, c):
    row = {name: value if isfinite(value) else None for name, value in zip(FIELDS, values.tolist())}
    counts = np.bincount(labels, minlength=c)
    truth = counts / counts.sum()
    row.update(fallback_reason=REASONS[int(row.pop("fallback_code"))],
               prior_tv=float(np.abs(prior - truth).sum() / 2), true_max_fraction=float(truth.max()))
    return row, [[int(k), int(counts[k])] for k in np.flatnonzero(counts)]


@torch.no_grad()
def evaluate(subsets, validation, labels, cache, head, variants, args, title, mechanism):
    sizes = dict.fromkeys(v.window for v in variants.values())
    windows = {size: FeatureLogitWindow(head, size) for size in sizes}
    models = {name: PLEAD(windows[v.window], v) for name, v in variants.items()}
    meters = {m: Metrics() for m in variants}
    holdouts = {m: Metrics() for m in variants} if validation else None
    timers = Timings(args.device)
    if timers.gpu:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    names = tuple(models)
    c, width = head.out_features, len(FIELDS)
    bank_labels = np.empty(0, dtype=np.int64)
    updates, subset_rows, priors = [], [], []
    by_method = {m: [] for m in models}
    n_forward = n_misses = 0
    batch_total = sum((len(s) + args.batch_size - 1) // args.batch_size for s in subsets)
    subset_correct = Counter()
    seen, since_switch = 0, 0
    iterator = tqdm(batches(subsets, args.batch_size), total=batch_total, desc=title,
                    unit="batch", leave=False, mininterval=1)
    for idx, (sid, indices) in enumerate(iterator):
        features, misses = cache.arrive(indices, timers)
        n_misses += misses
        n_forward += bool(misses)
        raw = {}
        for window_size, window in windows.items():
            with timers.measure(("window", window_size)):
                raw[window_size] = window.append(features)
        parts = []
        for method, model in models.items():
            win = variants[method].window
            with timers.measure((method, idx)):
                logits, diag, prior, soft = model.update(windows[win], raw[win])
            parts.append(torch.cat((logits.flatten(), diag, prior, soft)))
        size = len(indices)
        target = torch.from_numpy(labels[indices])
        bank_labels = np.concatenate((bank_labels, labels[indices]))[-max(windows):]
        seen += size
        for method, result in zip(names, torch.stack(parts).cpu()):
            logits = result[:size*c].reshape(size, c)
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite logits in {title}/{method}/{idx}")
            meter = meters[method]
            before = meter.correct1
            meter.add(logits, target)
            correct = meter.correct1 - before
            end = size*c + width
            prior = result[end:end+c].numpy()
            if not np.isfinite(prior).all():
                raise ValueError(f"Nonfinite prior in {title}/{method}/{idx}")
            config = variants[method]
            n = size if config.batch_only else config.window
            step, histogram = audit_step(result[size*c:end], prior, bank_labels[-n:], c)
            step.update(method=method, update_idx=idx, subset_id=sid, steps_since_switch=since_switch,
                        batch_size=size, correct1=correct, top1=100*correct/size)
            if method in mechanism:
                step["true_histogram"] = histogram
                if method == "plead":
                    step["soft_prior_row"] = len(priors)
                    priors.append(result[-c:].numpy().copy())
            updates.append(step)
            by_method[method].append(step)
            subset_correct[method] += correct
        since_switch += 1
        if seen == len(subsets[sid]):
            subset_rows.extend(dict(method=m, subset_id=sid, samples=seen, top1=100*subset_correct[m]/seen)
                               for m in models)
            seen, since_switch = 0, 0
            subset_correct.clear()

    if holdouts:
        window = next(iter(windows.values()))
        for _, indices in batches([validation], args.batch_size):
            features, misses = cache.arrive(indices, timers)
            n_misses += misses
            n_forward += bool(misses)
            with timers.measure("holdout_head"):
                raw = F.linear(features, window.weight, window.bias)
            target = torch.from_numpy(labels[indices])
            for method, model in models.items():
                with timers.measure(("holdout", method)):
                    predictions = raw - model.q
                holdouts[method].add(predictions.cpu(), target)

    times = timers.result()
    rows = []
    peak = torch.cuda.max_memory_allocated(args.device) / 1024**2 if timers.gpu else None
    for method, meter in meters.items():
        records = by_method[method]
        for step in records:
            step["adapter_seconds"] = times[method, step["update_idx"]]
        result = meter.result()
        primary = holdouts[method].result() if holdouts else result
        adapter = sum(s["adapter_seconds"] for s in records) + times["window", variants[method].window]
        total = sum(times[k] for k in ("backbone", "replay_read", "window_build", "holdout_head"))
        total += adapter + times["holdout", method]
        row = dict(method=method, top1=primary["top1"], top5=primary["top5"], ece=primary["ece"],
                   samples=primary["samples"], adaptation_top1=result["top1"], adaptation_samples=result["samples"],
                   fallback_rate=sum(bool(x["fallback_reason"]) for x in records)/len(records),
                   adapter_seconds=adapter, total_seconds=total, backbone_forward_count=n_forward,
                   backbone_images=n_misses, peak_gpu_mib=peak,
                   timing_scope="single_cold" if args.cold else "shared_experiment")
        for key in ("prior_tv", "kappa", "d", "v", "gamma", "a_norm", "b_norm", "q_norm"):
            finite = [x[key] for x in records if x[key] is not None]
            row[key] = float(np.mean(finite)) if finite else None
            if key in ("kappa", "gamma"):
                row[key + "_update_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
        if args.cold and n_misses != result["samples"] + (primary["samples"] if holdouts else 0):
            raise ValueError("Cold measurement reused a previous image")
        rows.append(row)
    return rows, updates, subset_rows, np.asarray(priors, dtype=np.float32)
