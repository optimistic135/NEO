"""Arrival-only frozen inference, online metrics, and synchronized timings."""

from collections import Counter
from contextlib import contextmanager
import math
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from cap_experiments.evaluation import Metrics
from tta_library.plead_lite import FeatureLogitWindow, NEOOnline, PLEADLite, describe_step
from tta_library.plead_v2 import PLEADV2, V2Config


METHODS = ("neo_online", "plead_lite")


class Timings:
    def __init__(self, device):
        self.device = device
        self.gpu = device.type == "cuda"
        self.events = []
        self.cpu = Counter()

    @contextmanager
    def measure(self, field):
        if self.gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events.append((field, start, end))
        else:
            start = perf_counter()
            yield
            self.cpu[field] += perf_counter() - start

    def result(self):
        self.cpu.update({name: 0 for name in ("backbone", "replay_read", "window_build", *METHODS)})
        if self.gpu:
            torch.cuda.synchronize(self.device)
            for field, start, end in self.events:
                self.cpu[field] += start.elapsed_time(end) / 1000
            self.events.clear()
        return self.cpu


class ArrivalFeatures:
    """Reuse only features produced at an arrival in the current or a completed replay."""

    def __init__(self, dataset, extractor, classifier, device):
        self.dataset = dataset
        self.extractor = extractor
        self.device = device
        self.features = torch.empty((len(dataset), classifier.in_features), device=device)
        self.available = np.zeros(len(dataset), dtype=bool)

    @torch.no_grad()
    def arrive(self, indices, clock):
        missing = indices[~self.available[indices]]
        if len(missing):
            # No dataset read takes place until a specific ordered batch arrives.
            images = torch.stack([self.dataset[int(index)][0] for index in missing])
            with clock.measure("backbone"):
                features = self.extractor(images.to(self.device)).float()
            with clock.measure("window_build"):
                self.features[missing] = features
            self.available[missing] = True
        with clock.measure("replay_read"):
            features = self.features[indices]
        return features, len(missing)


def sparse_histogram(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    return [[int(c), int(counts[c])] for c in np.flatnonzero(counts)]


def _finite_stats(diag):
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in diag.items()}


@torch.no_grad()
def evaluate_online(stream, labels, cache, classifier, args, v2, description):
    window = FeatureLogitWindow(classifier, min(128, stream.indices.shape[1]))
    neo, lite = NEOOnline(), PLEADLite(window, v2)
    timers = Timings(args.device)
    gpu = timers.gpu
    if gpu:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    tracker = {method: Metrics() for method in METHODS}
    batch_rows, subset_rows = [], []
    arrived_labels = np.empty(0, dtype=np.int64)
    missed_images = calls = total_batches = 0
    reasons = Counter()
    loop_start = perf_counter()

    limit = args.smoke_subsets if args.smoke else len(stream.indices)
    it = tqdm(enumerate(stream.indices[:limit]), total=limit, desc=description, unit="subset", leave=False)
    for subset, indices in it:
        subset_correct = Counter()
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset:offset + args.batch_size]
            features, misses = cache.arrive(selected, timers)
            count = len(features)
            with timers.measure("window_build"):
                window.append(features)
            with timers.measure("neo_online"):
                neo_logits = neo.update(window, count)
            with timers.measure("plead_lite"):
                lite_logits, diag = lite.update(window, count)
            # One transfer per batch for scoring and fit diagnostics. Labels stay in the evaluator.
            packed = torch.cat((neo_logits.reshape(-1), lite_logits.reshape(-1), diag)).cpu()
            first, second = neo_logits.numel(), neo_logits.numel() + lite_logits.numel()
            logits_by_method = (packed[:first].reshape_as(neo_logits), packed[first:second].reshape_as(lite_logits))
            if not torch.isfinite(packed[:second]).all():
                raise ValueError("Nonfinite online predictions")
            actual = labels[selected]
            target = torch.from_numpy(actual)
            for method, logits in zip(METHODS, logits_by_method):
                tracker[method].add(logits, target)
                subset_correct[method] += int((logits.topk(5, dim=1).indices[:, 0] == target).sum())
            arrived_labels = np.concatenate((arrived_labels, actual))[-window.capacity:]
            step = _finite_stats(describe_step(packed[second:].tolist()))
            if int(step["bank_size"]) != len(arrived_labels):
                raise ValueError("Tracked FIFO size differs from arrived population")
            step.update(subset=subset, batch=total_batches, size=count,
                        cache_new_images=misses, window_class_counts=sparse_histogram(arrived_labels, len(stream.indices)))
            batch_rows.append(step)
            reasons.update(step["fallback_reasons"])
            calls += bool(misses)
            missed_images += misses
            total_batches += 1
        dominant = int(stream.dominant_classes[subset])
        subset_rows.append({
            "subset": subset, "dominant_class": dominant, "p_max": stream.p_max, "p_min": stream.p_min,
            "dominant_fraction": float(stream.counts[subset, dominant] / len(indices)),
            "nonzero_classes": int(np.count_nonzero(stream.counts[subset])),
            "class_counts": sparse_histogram(labels[indices], len(stream.indices)),
            "samples": len(indices), "repeated_images": 0,
            **{f"{m}_top1": 100 * subset_correct[m] / len(indices) for m in METHODS},
        })

    timings = timers.result()
    wall_seconds = perf_counter() - loop_start
    peak = torch.cuda.max_memory_allocated(args.device) / 1024**2 if gpu else None
    rows = []
    for method in METHODS:
        metrics = tracker[method].result()
        execution = timings["backbone"] + timings["replay_read"] + timings["window_build"] + timings[method]
        is_lite = method == "plead_lite"
        row = dict(
            method=method, batches=total_batches, window=window.capacity, **metrics,
            backbone_forward_count=calls, backbone_images=missed_images,
            replay_cache_images=metrics["samples"] - missed_images,
            backbone_seconds=timings["backbone"], replay_cache_read_seconds=timings["replay_read"],
            window_cache_build_seconds=timings["window_build"], method_seconds=timings[method],
            total_inference_seconds=execution, stream_wall_seconds=wall_seconds,
            samples_per_second=metrics["samples"] / max(execution, 1e-12), peak_gpu_mib=peak,
            fallback_rate=sum(bool(s["fallback_reasons"]) for s in batch_rows) / total_batches if is_lite else 0,
            gamma_zero_rate=sum(s["gamma"] == 0 for s in batch_rows) / total_batches if is_lite else None,
            fallback_counts=dict(reasons) if is_lite else {},
        )
        if is_lite:
            for key in ("kappa", "lambda", "gamma", "neff", "sxx", "correlation"):
                values = [step[key] for step in batch_rows if step[key] is not None]
                row[key] = float(np.mean(values)) if values else None
        rows.append(row)
    return rows, subset_rows, batch_rows


@torch.no_grad()
def benchmark(dataset, indices, extractor, classifier, args, repeats=3):
    """Time both cold, single-view pipelines on identical preprocessed batches."""
    batches = [
        torch.stack([dataset[index][0] for index in batch])
        for batch in indices
    ]
    config = V2Config(max_bank=min(128, args.subset_size))
    if len(batches) < 2:
        raise ValueError("Need multiple benchmark batches")
    outputs = []
    for repeat in range(repeats + 1):
        for name in ("v2_fit_gate", "plead_lite"):
            timers = Timings(args.device)
            use_lite = name == "plead_lite"
            if use_lite:
                window = FeatureLogitWindow(classifier, config.max_bank)
                lite = PLEADLite(window, config)
            else:
                legacy = PLEADV2(classifier, "plead_v2_fit_gate", config)
            if timers.gpu:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            start = perf_counter()
            predictions = []
            for images in batches:
                with timers.measure("backbone"):
                    features = extractor(images.to(args.device)).float()
                if use_lite:
                    with timers.measure("window_build"):
                        window.append(features)
                    with timers.measure("adapter"):
                        logits, _ = lite.update(window, len(images))
                else:
                    with timers.measure("adapter"):
                        logits = legacy.update(features)
                predictions.append(logits)
            elapsed = timers.result()
            wall = perf_counter() - start
            peak = torch.cuda.max_memory_allocated(args.device) / 1024**2 if timers.gpu else None
            predictions = [prediction.cpu() for prediction in predictions]
            if use_lite:
                differences = [float((x - y).abs().max()) for x, y in zip(reference, predictions)]
                for x, y in zip(reference, predictions):
                    torch.testing.assert_close(x, y, atol=3e-4, rtol=5e-5)
            else:
                reference = predictions
                differences = [0.0]
                del legacy
            if repeat:
                outputs.append(dict(
                    method=name, repeat=repeat, samples=sum(len(b) for b in batches), backbone_forward_count=len(batches),
                    backbone_seconds=elapsed["backbone"], external_window_build_seconds=elapsed["window_build"] if use_lite else None,
                    cache_read_and_update_seconds=elapsed["adapter"] if use_lite else None,
                    method_seconds=elapsed["adapter"] + elapsed["window_build"], total_inference_seconds=wall,
                    samples_per_second=sum(len(b) for b in batches) / wall,
                    peak_gpu_mib=peak, max_abs_logit_difference=max(differences),
                ))
    return outputs
