"""Evaluate original NEO and single-view PLEAD on NEO paper-style protocols."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev
import sys

import torch
from scipy.stats import t
from torch import nn
from tqdm import tqdm

from cap_experiments.evaluation import Metrics, _loader
from cap_experiments.ir_evaluation import Timings
from cap_experiments.protocol import load_dataset, load_imagenet_dataset, manifest, natural_stream
from run_cap_neo import CLASS_COUNTS, SEEDS, Tee, _configuration, all_corruptions, load_backbone, write_csv, write_json
from run_plead_v2 import MODELS, preflight
from tta_library.cap_neo import CAPConfig
from tta_library.neo_cont import NEO_Cont
from tta_library.plead_lite import FeatureLogitWindow, PLEADLite, describe_step
from tta_library.plead_v2 import V2Config
from utils.utils import set_reproducible


PROTOCOLS = ("table2", "heldout-512", "continual")
PAPER_TABLE2 = {
    "imagenet-c": (55.6, 59.2),
    "cifar10-c": (80.4, 82.4),
}
METHODS = ("no_adapt", "neo", "plead_single_view")
BUDGET = 512


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CLASS_COUNTS, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="table2")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, help="Continual EMA rate, defaulting to the author repository's 0.1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.protocol != "continual" and args.alpha is not None:
        parser.error("--alpha applies to the continual protocol")
    if args.protocol == "continual":
        args.alpha = 0.1 if args.alpha is None else args.alpha
        if not 0 <= args.alpha <= 1:
            parser.error("--alpha must be in [0, 1]")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.device = torch.device(args.device)
    args.batch_size, args.level, args.workers = 64, 5, 0
    args.corruptions = ["gaussian_noise", "shot_noise"] if args.smoke else list(all_corruptions)
    args.seeds = [2020] if args.smoke else list(SEEDS)
    if not args.data.is_dir():
        parser.error(f"Missing data directory {args.data}")
    return args


def create_adapters(neo, args):
    head = neo.classifier
    if args.protocol == "continual":
        model = nn.Module()
        model.head = head
        neo = NEO_Cont(model, head.out_features, learning_rate=args.alpha)
        neo.feature_extractor = nn.Identity()
        neo.imagenet_mask = None
    neo.reset()
    neo.corrupt_class_center = neo.corrupt_class_center.to(head.weight.device)
    bank = FeatureLogitWindow(head, BUDGET)
    return neo, bank, PLEADLite(bank, V2Config(max_bank=BUDGET))


def evaluation_phases(protocol, stream):
    phases = [("adaptation", stream.adaptation, True)]
    if protocol == "heldout-512":
        phases.append(("heldout", stream.validation["validation"], False))
    return phases


@torch.no_grad()
def evaluate(dataset, stream, extractor, adapters, args, seed):
    neo, bank, lite = adapters
    rows, steps = [], []
    timers = Timings(args.device)
    gpu = timers.gpu
    if gpu:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    for phase, indices, adapt in evaluation_phases(args.protocol, stream):
        meters = {method: Metrics() for method in METHODS}
        for images, labels in _loader(dataset, indices, args, seed):
            with timers.measure(f"{phase}_backbone"):
                h = extractor(images.to(args.device, non_blocking=True)).float()
            with timers.measure(f"{phase}_no_adapt"):
                frozen = neo.classifier(h)
            with timers.measure(f"{phase}_neo"):
                original = neo(h, adapt=adapt)
            with timers.measure(f"{phase}_plead_single_view"):
                if adapt:
                    bank.append(h)
                    correction, diag = lite.update(bank, len(h))
                else:
                    correction = frozen - lite.q
            outputs = torch.stack((frozen, original, correction)).cpu()
            if not torch.isfinite(outputs).all():
                raise ValueError(f"Nonfinite logits during {phase}")
            for method, logits in zip(METHODS, outputs):
                meters[method].add(logits, labels)
            if adapt:
                steps.append({
                    key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in describe_step(diag.tolist()).items()
                })
        rows.extend({"phase": phase, "method": method, **meter.result()} for method, meter in meters.items())
    times = timers.result()
    peak = torch.cuda.max_memory_allocated(args.device) / 1024**2 if gpu else None
    for row in rows:
        row["method_seconds"] = times[f"{row['phase']}_{row['method']}"]
        row["inference_seconds"] = row["method_seconds"] + times[f"{row['phase']}_backbone"]
        row["shared_peak_gpu_mib"] = peak
        row["fallback_rate"] = (
            sum(bool(step["fallback_reasons"]) for step in steps) / len(steps)
            if row["method"] == "plead_single_view" else None
        )
    return rows, steps


def combined_stats(values, name):
    return {f"{name}_mean": mean(values), f"{name}_std": stdev(values) if len(values) > 1 else None}


def summarize(records, args):
    rows = [row for record in records for row in record["rows"]]
    lookup = {(row["seed"], row["corruption"], row["phase"], row["method"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate paper protocol result")
    summaries, pairs = [], []
    continual = args.protocol == "continual"
    paper_method = "neo_continual" if continual else "neo"
    phases = ("adaptation", "heldout") if args.protocol == "heldout-512" else ("adaptation",)
    methods = ("no_adapt", paper_method, "plead_single_view")
    for seed in args.seeds:
        for corruption in args.corruptions:
            for phase in phases:
                baseline = lookup[seed, corruption, phase, paper_method]
                candidate = lookup[seed, corruption, phase, "plead_single_view"]
                if (baseline["split_sha256"], baseline["samples"]) != (candidate["split_sha256"], candidate["samples"]):
                    raise ValueError("Paired experiment samples differ")
                pairs.append(dict(seed=seed, corruption=corruption, phase=phase,
                                  split_sha256=baseline["split_sha256"], top1_delta_pp=candidate["top1"] - baseline["top1"]))
    for phase in phases:
        for corruption in [*args.corruptions, "overall"]:
            scope = args.corruptions if corruption == "overall" else [corruption]
            for method in methods:
                cell = dict(phase=phase, corruption=corruption, method=method, n_seeds=len(args.seeds))
                for metric in ("top1", "top5", "ece", "entropy", "method_seconds", "inference_seconds"):
                    values = [mean(lookup[seed, c, phase, method][metric] for c in scope) for seed in args.seeds]
                    cell.update(combined_stats(values, metric))
                if method != "no_adapt":
                    gains = [mean(lookup[s, c, phase, method]["top1"] - lookup[s, c, phase, "no_adapt"]["top1"]
                                  for c in scope) for s in args.seeds]
                    cell.update(combined_stats(gains, "gain_vs_no_adapt_pp"))
                if method == "plead_single_view":
                    gains = [mean(lookup[s, c, phase, method]["top1"] - lookup[s, c, phase, paper_method]["top1"]
                                  for c in scope) for s in args.seeds]
                    cell.update(combined_stats(gains, "gain_vs_neo_pp"))
                    if len(gains) > 1:
                        half_width = t.ppf(.975, len(gains) - 1) * stdev(gains) / len(gains) ** .5
                        cell.update(paired_ci95_low=mean(gains) - half_width, paired_ci95_high=mean(gains) + half_width)
                if (not args.smoke and args.protocol == "table2" and args.dataset in PAPER_TABLE2
                        and corruption == "overall" and method in ("no_adapt", "neo")):
                    cell["published_table2_top1"] = PAPER_TABLE2[args.dataset][method == "neo"]
                    cell["difference_vs_published_pp"] = cell["top1_mean"] - cell["published_table2_top1"]
                summaries.append(cell)
    write_csv(args.output / "per_stream.csv", rows)
    write_csv(args.output / "paired_differences.csv", pairs)
    write_csv(args.output / "summary.csv", summaries)
    lines = [f"# {args.dataset} — {args.protocol}", "",
             f"Scope: {'smoke' if args.smoke else 'full'}, {len(args.corruptions)} corruptions, seeds {args.seeds}.",
             "Each corruption contributes 512 distinct adaptation images (64 per batch).",
             "Values are macro-averaged over corruptions within seed, then reported as mean ± sample SD.", "",
             "| Phase | Method | Top-1 (%) | Top-5 (%) | ECE | Published Table 2 Top-1 (%) |",
             "|---|---|---:|---:|---:|---:|"]
    labels = {"neo": "NEO", "no_adapt": "No Adapt", "neo_continual": "NEO-Continual", "plead_single_view": "PLEAD (Single-View)"}
    for row in summaries:
        if row["corruption"] == "overall":
            values = [f"{row[m + '_mean']:.3f}" + (f" ± {row[m + '_std']:.3f}" if row[m + "_std"] is not None else "")
                      for m in ("top1", "top5", "ece")]
            lines.append(f"| {row['phase']} | {labels[row['method']]} | " + " | ".join(values)
                         + f" | {row.get('published_table2_top1', '—')} |")
    for phase in phases:
        lines += ["", f"## {phase.title()} Top-1 by corruption", "",
                  f"| Corruption | {labels['no_adapt']} | {labels[paper_method]} | {labels['plead_single_view']} | Paired gain (pp) |",
                  "|---|---:|---:|---:|---:|"]
        for corruption in [*args.corruptions, "overall"]:
            metrics = {r["method"]: r for r in summaries if r["phase"] == phase and r["corruption"] == corruption}
            accuracy = []
            for method in methods:
                row = metrics[method]
                sd = row["top1_std"]
                accuracy.append(f"{row['top1_mean']:.3f}" + (f" ± {sd:.3f}" if sd is not None else ""))
            lite = metrics["plead_single_view"]
            gain = f"{lite['gain_vs_neo_pp_mean']:+.3f}"
            if "paired_ci95_low" in lite:
                gain += f" [{lite['paired_ci95_low']:+.3f}, {lite['paired_ci95_high']:+.3f}]"
            lines.append(f"| {corruption} | " + " | ".join(accuracy) + f" | {gain} |")
    lines += ["", "Intervals use the three seed-level paired means. Per-stream rows and paired differences are in the accompanying CSVs."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(args.output / "verification.json", dict(
        scope="smoke" if args.smoke else "full", seeds=args.seeds, corruptions=args.corruptions,
        protocol=args.protocol, stream_count=len(records), method_rows=len(rows),
    ))


def run(args):
    preflight(args)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    model, args.model_revision, processor_id, args.processor_revision = MODELS[args.dataset]
    config = _configuration(args, CAPConfig(), model, processor_id)
    del config["cap"]
    config.update(adaptation_samples=BUDGET, official_seed_manifest=False,
                  dataset_scope="extension" if args.dataset == "cifar100-c" else "paper_dataset")
    config_path = args.output / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError("Saved configuration differs; use a new output directory")
    write_json(config_path, config)
    set_reproducible(args.seeds[0])
    neo, extractor, processor = load_backbone(args, model, processor_id)
    classes = CLASS_COUNTS[args.dataset]
    continual = args.protocol == "continual"
    records = []
    for seed in args.seeds:
        order = list(args.corruptions)
        if continual:
            random.Random(seed).shuffle(order)
        # A continual seed is saved atomically since every corruption depends on its prefix.
        sequence_path = args.output / "runs" / str(seed) / "sequence.json"
        if continual and sequence_path.is_file():
            saved = json.loads(sequence_path.read_text())
            if saved["order"] != order:
                raise ValueError("Continual order mismatch")
            records.extend(saved["records"])
            continue
        adapters = create_adapters(neo, args) if continual else None
        sequence = []
        for position, corruption in enumerate(tqdm(order, desc=f"{args.dataset} {seed}", unit="corruption")):
            path = args.output / "runs" / str(seed) / f"{corruption}.json"
            if not continual and path.is_file():
                records.append(json.loads(path.read_text()))
                continue
            set_reproducible(seed)
            if args.dataset == "imagenet-c":
                dataset, labels, ids = load_imagenet_dataset(args.data, corruption, args.level)
            else:
                dataset, labels, ids = load_dataset(args.data, corruption, args.level, processor, classes)
            stream = natural_stream(len(dataset), BUDGET, seed)
            splits = manifest(stream, labels, ids, classes)
            if args.protocol != "heldout-512":
                del splits["validation"]
            digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
            if not continual:
                adapters = create_adapters(neo, args)
            rows, steps = evaluate(dataset, stream, extractor, adapters, args, seed)
            for row in rows:
                if row["samples"] != len(splits[row["phase"] if row["phase"] == "adaptation" else "validation"]["indices"]):
                    raise ValueError("Incorrect number of evaluated samples")
                if continual and row["method"] == "neo":
                    row["method"] = "neo_continual"
                row.update(dataset=args.dataset, protocol=args.protocol, seed=seed, corruption=corruption,
                           level=args.level, corruption_position=position, split_sha256=digest)
            record = dict(splits=splits, rows=rows, diagnostics=steps)
            if not continual:
                write_json(path, record)
            sequence.append(record)
            records.append(record)
        if continual:
            write_json(sequence_path, dict(order=order, records=sequence))
    if len(records) != len(args.seeds) * len(args.corruptions):
        raise ValueError("Incomplete corruption coverage")
    summarize(records, args)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"An experiment already owns {args.output}") from None
        with (args.output / "run.log").open("a", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                run(args)


if __name__ == "__main__":
    main()
