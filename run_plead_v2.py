"""Run PLEAD-V2 ablation and DART-inspired long-tail evaluation for one dataset."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from cap_experiments.evaluation import evaluate_stream
from cap_experiments.longtail import ablation_stream, longtail_stream
from cap_experiments.plead_reporting import generate
from cap_experiments.protocol import imagenet_directory, load_dataset, load_imagenet_dataset, manifest
from run_cap_neo import (
    CIFAR100_MODEL, CIFAR100_REVISION, CLASS_COUNTS, IMAGENET_MODEL, MODEL_ID, MODEL_REVISION,
    PROCESSOR_ID, PROCESSOR_REVISION, SEEDS, Tee, _configuration, all_corruptions,
    load_backbone, write_json,
)
from tta_library.cap_neo import CAPConfig
from tta_library.plead_v2 import ABLATION_METHODS, V2Config
from utils.utils import set_reproducible


MODELS = {
    "cifar10-c": (MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION),
    "cifar100-c": (CIFAR100_MODEL, CIFAR100_REVISION, CIFAR100_MODEL, CIFAR100_REVISION),
    "imagenet-c": (IMAGENET_MODEL, None, "te_transforms_imageC", None),
}
RHOS = (1, 10, 100)
LONGTAIL_METHODS = ("neo", "plead_v2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CLASS_COUNTS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="gaussian_noise, seed 2020; use a separate output")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or not args.data.is_dir():
        parser.error("Provide an existing data directory and nonnegative worker count")
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.device = torch.device(args.device)
    args.batch_size = 64
    args.level = 5
    args.adapt_samples = 512
    args.seeds = [2020] if args.smoke else list(SEEDS)
    args.corruptions = ["gaussian_noise"] if args.smoke else list(all_corruptions)
    args.validation_per_class = {"cifar10-c": 50, "cifar100-c": 5, "imagenet-c": 1}[args.dataset]
    args.check_equivalence = False
    return args


def preflight(args):
    for corruption in args.corruptions:
        if args.dataset == "imagenet-c":
            imagenet_directory(args.data, corruption, args.level)
        elif (args.data / "labels.npy").is_file():
            if not (args.data / f"{corruption}.npy").is_file():
                raise FileNotFoundError(args.data / f"{corruption}.npy")
        elif not (args.data / corruption / str(args.level)).is_dir():
            raise FileNotFoundError(args.data / corruption / str(args.level))


def _check_baseline(row, old_rows, old_method):
    reference = next(item for item in old_rows if item["method"] == old_method)
    fields = ["kappa", "lambda", "fallback_counts"]
    fields.extend(field for field in reference if field.endswith(("_top1", "_top5", "_ece", "_entropy")))
    if any(row[field] != reference[field] for field in fields):
        raise AssertionError(f"Smoke legacy metrics differ for {old_method}")


def run(args):
    preflight(args)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    model_id, args.model_revision, processor_id, args.processor_revision = MODELS[args.dataset]
    cap_cfg, v2_cfg = CAPConfig(), V2Config()
    config = _configuration(args, cap_cfg, model_id, processor_id)
    config.update(v2=asdict(v2_cfg), rhos=list(RHOS), ablation_methods=list(ABLATION_METHODS),
                  longtail_methods=list(LONGTAIL_METHODS))
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Configuration changed. Use a new output directory")
    write_json(config_path, config)

    set_reproducible(args.seeds[0])
    neo, extractor, processor = load_backbone(args, model_id, processor_id)
    rows = {"ablation": [], "longtail": []}
    jobs = [(seed, corruption) for seed in args.seeds for corruption in args.corruptions]
    progress = tqdm(jobs, desc=f"{args.dataset} seed/corruption", disable=args.no_progress)
    for seed, corruption in progress:
        set_reproducible(seed)
        classes = CLASS_COUNTS[args.dataset]
        if args.dataset == "imagenet-c":
            dataset, labels, ids = load_imagenet_dataset(args.data, corruption, args.level)
        else:
            dataset, labels, ids = load_dataset(args.data, corruption, args.level, processor, classes)
        sampling_args = (args.adapt_samples, args.validation_per_class)
        streams = [("ablation", ablation_stream(labels, classes, seed, *sampling_args), {})]
        for rho in RHOS:
            stream, audit = longtail_stream(labels, classes, seed, rho, *sampling_args)
            streams.append(("longtail", stream, audit))

        for experiment, stream, audit in streams:
            methods = ABLATION_METHODS if experiment == "ablation" else LONGTAIL_METHODS
            path = args.output / "runs" / str(seed) / corruption / f"{stream.name}.json"
            splits = manifest(stream, labels, ids, classes)
            views = [set(indices) for indices in stream.validation.values()]
            if len(views) == 2 and views[0].intersection(views[1]):
                raise ValueError("Matched and balanced validation overlap")
            digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
            if path.exists():
                record = json.loads(path.read_text())
                if record["splits"] != splits or record["sampling"] != audit:
                    raise ValueError(f"Saved sampling differs in {path}")
                rows[experiment].extend(record["rows"])
                continue

            description = f"{seed} {corruption} {stream.name}"
            progress.set_postfix_str(description)
            parity = args.smoke and (experiment == "ablation" or stream.ratio == 1)
            new_method = "plead_original" if experiment == "ablation" else "neo"
            saved_logits = {}

            def capture(method, phase, offset, logits):
                if method == new_method:
                    saved_logits[phase, offset] = logits

            def compare(method, phase, offset, logits):
                torch.testing.assert_close(logits, saved_logits[phase, offset], atol=0, rtol=0)

            results, histories, _ = evaluate_stream(
                dataset, stream, extractor, neo, methods, cap_cfg, args, seed, description,
                v2_config=v2_cfg, observe_logits=capture if parity else None,
            )
            if parity:
                old_method = "cap_neo" if experiment == "ablation" else "neo"
                old_rows, old_history, _ = evaluate_stream(
                    dataset, stream, extractor, neo, (old_method,), cap_cfg, args, seed,
                    f"{description} legacy parity", observe_logits=compare,
                )
                _check_baseline(next(row for row in results if row["method"] == new_method), old_rows, old_method)
                if histories[new_method] != old_history[old_method]:
                    raise AssertionError(f"Smoke legacy histories differ for {old_method}")
                tqdm.write(f"Legacy parity passed for {new_method}")

            counts = splits["adaptation"]["label_counts"]
            for row in results:
                history = histories[row["method"]]
                row.update(
                    dataset=args.dataset, experiment=experiment, seed=seed, corruption=corruption,
                    severity=args.level, rho=stream.ratio, split_sha256=digest,
                    adaptation_class_counts=counts, missing_classes=counts.count(0),
                    adaptation_unique_images=len(set(stream.adaptation)),
                    balanced_class_counts=splits["balanced_validation"]["label_counts"],
                    matched_class_counts=splits.get("matched_validation", {}).get("label_counts"),
                    matched_missing_classes=(
                        splits["matched_validation"]["label_counts"].count(0)
                        if "matched_validation" in splits else None
                    ),
                    target_class_prior=audit.get("target_prior"),
                    adaptation_redistributed=audit.get("adaptation_sampling", {}).get("redistributed_count"),
                    matched_redistributed=audit.get("matched_sampling", {}).get("redistributed_count"),
                    fallback_rate=row["fallback_updates"] / len(history),
                    fit_fallback_rate=(
                        sum(bool(step["fit_fallback_reason"]) for step in history) / len(history)
                        if "fit_fallback_reason" in history[-1] else None
                    ),
                    gate_off_rate=(
                        sum(step["gamma"] == 0 for step in history) / len(history)
                        if row["gamma_gate"] else None
                    ),
                )
                view = "matched_validation" if experiment == "longtail" else "balanced_validation"
                tqdm.write(
                    f"{description} {row['method']} top1={row[f'{view}_top1']:.2f} "
                    f"kappa={row['kappa']:.4g} lambda={row['lambda']:.4g} "
                    f"gamma={row['gamma']} fallback={row['fallback_rate']:.1%}"
                )
            write_json(path, {"splits": splits, "sampling": audit, "rows": results, "diagnostics": histories})
            rows[experiment].extend(results)
    for experiment, records in rows.items():
        expected = len(jobs) * (len(ABLATION_METHODS) if experiment == "ablation" else len(RHOS) * len(LONGTAIL_METHODS))
        if len(records) != expected:
            raise ValueError(f"Expected {expected} {experiment} results, found {len(records)}")
        generate(args.output, experiment, records)


def main():
    args = parse_args()
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
