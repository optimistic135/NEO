"""Run CAP-NEO experiments independently of main.py."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import timm
import torch
from torch import nn
from tqdm import tqdm
import transformers
import torchvision
from transformers import ViTForImageClassification, ViTImageProcessor

from cap_experiments.evaluation import evaluate_stream
from cap_experiments.protocol import (
    IMAGENET_CLASSES, PROCESSOR_ID, PROCESSOR_REVISION, controlled_imagenet_stream,
    controlled_stream, imagenet_directory, load_dataset, load_imagenet_dataset, manifest,
    natural_stream,
)
from factories import all_corruptions
from tta_library.cap_neo import CAPConfig, METHODS
from tta_library.neo import NEO
from utils.utils import get_device, set_reproducible


MODEL_ID = "nateraw/vit-base-patch16-224-cifar10"
MODEL_REVISION = "b55eeb4221d3a568f627078ed6f27b967810be3d"
IMAGENET_MODEL = "vit_base_patch16_224"
SEEDS = (1234, 2020, 9999)
RATIOS = (10, 30, 50, 70, 90, 100)
STAGE_METHODS = {
    "baseline": ("neo",),
    "ablation": METHODS[:-1],
    "imbalance": METHODS[:-1],
    "oracle": ("cap_oracle_labels",),
}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(*STAGE_METHODS, "all"))
    parser.add_argument("--dataset", choices=("cifar10-c", "imagenet-c"), default="cifar10-c")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--corruptions", nargs="+", choices=all_corruptions)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--ratios", nargs="+", type=int, choices=(0, *RATIOS))
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--adapt-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--controlled-validation-size", type=int)
    parser.add_argument("--confidence", type=float, default=CAPConfig.confidence)
    parser.add_argument("--min-fit-samples", type=int, default=CAPConfig.min_fit_samples)
    parser.add_argument("--min-per-class", type=int, default=CAPConfig.min_per_class)
    parser.add_argument("--min-coverage", type=int, default=CAPConfig.min_coverage)
    parser.add_argument("--min-alignment", type=float, default=CAPConfig.min_alignment)
    parser.add_argument("--eps", type=float, default=CAPConfig.eps)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--processor-revision", default=PROCESSOR_REVISION)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    imagenet = args.dataset == "imagenet-c"
    if args.corruptions is None:
        args.corruptions = [name for name in all_corruptions if not imagenet or name != "zoom_blur"]
    args.ratios = args.ratios or list((0, *RATIOS) if imagenet else RATIOS)
    if args.controlled_validation_size is None:
        args.controlled_validation_size = IMAGENET_CLASSES if imagenet else 480
    if args.stage == "all" and args.methods:
        parser.error("--methods requires a single stage")
    args.methods = args.methods or list(STAGE_METHODS.get(args.stage, ()))
    if not set(args.methods) <= set(STAGE_METHODS.get(args.stage, ())):
        parser.error("The oracle is isolated from unsupervised stages")
    if any(len(items) != len(set(items)) for items in (
        args.methods, args.corruptions, args.seeds, args.ratios,
    )):
        parser.error("Method, corruption, seed and ratio lists must not contain duplicates")
    dataset_size = 50_000 if imagenet else 10_000
    if not 1 <= args.adapt_samples < dataset_size or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid sample, batch or worker count")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if not imagenet and 0 in args.ratios:
        parser.error("CIFAR-10-C uses 10% as its uniform control")
    if args.stage in {"imbalance", "oracle", "all"}:
        size = args.controlled_validation_size
        if imagenet and (not IMAGENET_CLASSES <= size < 50_000 or size % IMAGENET_CLASSES):
            parser.error("ImageNet controlled validation size must be a multiple of 1000 from 1000 to 49000")
        if not imagenet and (size < 10 or size % 10 or args.adapt_samples + size > 1000):
            parser.error("Controlled validation size must be a positive multiple of ten and fit with adaptation")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if not args.data.is_dir():
        parser.error(f"Missing dataset directory: {args.data}")
    if imagenet:
        args.model_revision = None
        args.processor_revision = None
        for corruption in args.corruptions:
            try:
                imagenet_directory(args.data, corruption, args.level)
            except ValueError as error:
                parser.error(str(error))
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    pending = path.with_suffix(".tmp")
    with pending.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    pending.replace(path)


def _configuration(args, cfg, model_id, processor_id):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    cli = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items() if key != "no_progress"
    }
    return {
        "arguments": cli,
        "cap": asdict(cfg),
        "model": model_id,
        "processor": processor_id,
        "git_revision": revision,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
        "paired_clean_shift_metrics": "skipped; no verified clean pair input",
    }


def run(args):
    args.device = get_device() if args.device == "auto" else torch.device(args.device)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    args.check_equivalence = args.stage == "baseline"
    cfg = CAPConfig(
        args.confidence, args.min_fit_samples, args.min_per_class,
        args.min_coverage, args.min_alignment, args.eps,
    )
    imagenet = args.dataset == "imagenet-c"
    model_id = IMAGENET_MODEL if imagenet else MODEL_ID
    processor_id = "te_transforms_imageC" if imagenet else PROCESSOR_ID
    configuration = _configuration(args, cfg, model_id, processor_id)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError(f"Configuration differs from {config_path}; choose a fresh output directory")
    write_json(config_path, configuration)
    print(f"Stage {args.stage}, {args.device}, model {model_id}, output {args.output}")
    set_reproducible(args.seeds[0])
    if imagenet:
        model = timm.create_model(model_id, pretrained=True)
        processor = None
    else:
        model = ViTForImageClassification.from_pretrained(model_id, revision=args.model_revision)
        processor = ViTImageProcessor.from_pretrained(PROCESSOR_ID, revision=args.processor_revision)
    model.to(args.device).eval().requires_grad_(False)
    classes = IMAGENET_CLASSES if imagenet else 10
    neo = NEO(model, classes)
    if neo.classifier.out_features != classes:
        raise ValueError(f"Expected {classes} classifier outputs")
    neo.imagenet_mask = None
    extractor = neo.feature_extractor
    neo.feature_extractor = nn.Identity()

    jobs = [(seed, corruption) for seed in args.seeds for corruption in args.corruptions]
    rows = []
    progress = tqdm(jobs, desc=f"{args.stage} seed/corruption", unit="run", disable=args.no_progress)
    for seed, corruption in progress:
        set_reproducible(seed)
        if imagenet:
            dataset, labels, sample_ids = load_imagenet_dataset(args.data, corruption, args.level)
        else:
            dataset, labels, sample_ids = load_dataset(args.data, corruption, args.level, processor)
        streams = []
        if args.stage != "imbalance":
            streams.append(natural_stream(len(dataset), args.adapt_samples, seed))
        if args.stage in {"imbalance", "oracle"}:
            build_stream = controlled_imagenet_stream if imagenet else controlled_stream
            streams.extend(build_stream(
                labels, args.adapt_samples, args.controlled_validation_size, seed, ratio
            ) for ratio in args.ratios)
        for stream in streams:
            path = args.output / "runs" / str(seed) / corruption / f"{stream.name}.json"
            splits = manifest(stream, labels, sample_ids, classes)
            split_digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
            if path.is_file():
                record = json.loads(path.read_text())
                if record["splits"] != splits:
                    raise ValueError(f"Dataset split differs from {path}")
                rows.extend(record["rows"])
                print(f"Reuse {seed}/{corruption}/{stream.name}")
                continue
            description = f"{seed} {corruption} {stream.name}"
            progress.set_postfix_str(description)
            results, histories, elapsed = evaluate_stream(
                dataset, stream, extractor, neo, args.methods, cfg, args, seed, description
            )
            for result in results:
                counts = splits["adaptation"]["label_counts"]
                prior = result["predicted_prior"]
                result.update({
                    "stage": args.stage,
                    "dataset": args.dataset,
                    "seed": seed,
                    "corruption": corruption,
                    "level": args.level,
                    "stream": stream.name,
                    "ratio": stream.ratio,
                    "dominant_class": stream.dominant_class,
                    "split_sha256": split_digest,
                    "adaptation_label_counts": counts,
                    "adaptation_unique_samples": len(set(stream.adaptation)),
                    "adaptation_updates": len(histories[result["method"]]),
                    "fallback_rate": result["fallback_updates"] / len(histories[result["method"]]),
                    "prior_total_variation": (
                        0.5 * sum(abs(estimate - count / len(stream.adaptation))
                                  for estimate, count in zip(prior, counts))
                        if prior is not None else None
                    ),
                    "observed_dominant_fraction": (
                        counts[stream.dominant_class] / len(stream.adaptation)
                        if stream.dominant_class is not None else None
                    ),
                    "stream_wall_seconds": elapsed,
                })
                views = " ".join(
                    f"{phase}={result[f'{phase}_top1']:.2f}%"
                    for phase in stream.validation
                )
                tqdm.write(
                    f"{description} {result['method']} "
                    f"adaptation={result['adaptation_top1']:.2f}% {views} "
                    f"kappa={result['kappa']:.4f} lambda={result['lambda']:.4f}"
                )
            write_json(path, {"splits": splits, "rows": results, "diagnostics": histories})
            rows.extend(results)
            write_csv(args.output / "per_corruption.csv", rows)
    write_csv(args.output / "per_corruption.csv", rows)


def run_stage(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.log").open("a", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception:
                import traceback

                traceback.print_exc()
                raise SystemExit(1) from None


def main():
    args = parse_args()
    if args.stage != "all":
        run_stage(args)
        return

    from cap_experiments.reporting import generate_report

    stages = []
    for stage, methods in STAGE_METHODS.items():
        output = args.output / stage
        stages.append(output)
        run_stage(argparse.Namespace(**{
            **vars(args), "stage": stage, "methods": list(methods), "output": output,
        }))
    generate_report(stages, args.output / "report")


if __name__ == "__main__":
    main()
