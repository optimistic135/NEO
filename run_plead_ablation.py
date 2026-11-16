"""Run PLEAD ablations and export resumable, paired experiment records."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

from cap_experiments.ablation_evaluation import evaluate
from cap_experiments.ablation_protocol import IR_LEVELS, VALIDATION_PER_CLASS, VARIANTS, experiments
from cap_experiments.ir_evaluation import ArrivalFeatures
from cap_experiments.longtail import ablation_stream
from cap_experiments.online_ir import SPECS, make_stream
from cap_experiments.protocol import load_dataset, load_imagenet_dataset
from run_cap_neo import SEEDS, Tee, _configuration, all_corruptions, load_backbone, write_json
from run_online_ir import write_record
from run_plead_v2 import MODELS, preflight
from tta_library.cap_neo import CAPConfig
from tta_library.plead_lite import FeatureLogitWindow, PLEADLite
from utils.utils import set_reproducible


ROOT = Path("results/plead_ablation")
FOLDERS = {"cifar10-c": "CIFAR-10-C", "cifar100-c": "CIFAR-100-C", "imagenet-c": "ImageNet-C"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=IR_LEVELS)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Gaussian noise, seed 2020, two online subsets")
    parser.add_argument("--report-only", action="store_true", help="Verify and summarize saved streams")
    args = parser.parse_args()
    if not args.report_only and args.dataset is None:
        parser.error("--dataset is required for execution")
    args.output = (ROOT / "smoke" if args.smoke else ROOT).resolve()
    args.device = torch.device(args.device)
    if args.dataset:
        args.data = (args.data or Path("/data/user3/chenc/datasets") / FOLDERS[args.dataset]).resolve()
        args.level = 5
        args.seeds = [2020] if args.smoke else list(SEEDS)
        args.corruptions = ["gaussian_noise"] if args.smoke else list(all_corruptions)
    return args


def data_for(args, corruption, seed, processor):
    set_reproducible(seed)
    if args.dataset == "imagenet-c":
        return load_imagenet_dataset(args.data, corruption, args.level)
    return load_dataset(args.data, corruption, args.level, processor, SPECS[args.dataset][0])


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def save_record(args, suite, corruption, seed, ir, splits, groups, cache, labels, head, ir_audit=None):
    base = Path(args.dataset) / corruption / str(seed) / ("standard.json.gz" if ir is None else f"ir_{ir}.json.gz")
    raw = args.output / "data" / "raw"
    path = raw / suite / base
    sha = digest(splits)
    existed = path.is_file()
    if existed:
        with gzip.open(path, "rt") as handle:
            old = json.load(handle)
        if old["split_sha256"] != sha or old["experiments"] != {k: list(v) for k, v in groups.items()}:
            raise ValueError(f"Completed stream conflicts with requested inputs in {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        variants = {name: VARIANTS[name] for name in dict.fromkeys(m for group in groups.values() for m in group)}
        title = f"{args.dataset}/{corruption}/{seed}/{'standard' if ir is None else ir}"
        rows, updates, subset_rows = evaluate(
            splits["adaptation"], splits.get("balanced_validation"), labels, cache, head, variants, args, title,
        )
        identity = dict(dataset=args.dataset, corruption=corruption, severity=args.level, seed=seed, IR=ir)
        for row in rows:
            row.update(**identity, split_sha256=sha)
        write_record(path, dict(
            identity=identity, split_sha256=sha, splits=splits, experiments=groups,
            rows=rows, updates=updates, subsets=subset_rows, ir_audit=ir_audit,
        ))
        tqdm.write(f"{title} Full={rows[0]['top1']:.3f}% saved")
    # Each secondary study references exact raw rows, avoiding re-running Full or duplicating logits.
    for experiment, variants in groups.items():
        if experiment != suite:
            write_json(raw / experiment / base.with_suffix("").with_suffix(".json"),
                       {"raw_record": str(path.relative_to(raw)), "methods": list(variants), "split_sha256": sha})


@torch.no_grad()
def run(args):
    preflight(args)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    classes, subset_size, online_batch, _ = SPECS[args.dataset]
    model, args.model_revision, processor_id, args.processor_revision = MODELS[args.dataset]
    config = _configuration(args, CAPConfig(), model, processor_id)
    config.pop("cap")
    config.update(
        standard={"budget": 512, "batch_size": 64, "validation_per_class": VALIDATION_PER_CLASS[args.dataset]},
        online={"IR": list(IR_LEVELS[args.dataset]), "subsets_per_stream": classes,
                "subset_size": subset_size, "batch_size": online_batch, "smoke_subsets": 2},
        oracle_logit_shift={"status": "skipped", "reason": "No verified clean/corrupt pairing was supplied."},
        variants={name: asdict(value) for name, value in VARIANTS.items()},
    )
    config_path = args.output / "configs" / f"{args.dataset}.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Run configuration changed at {config_path}")
    write_json(config_path, config)
    set_reproducible(2020)
    neo, extractor, processor = load_backbone(args, model, processor_id)
    head = neo.classifier
    for batch_size in sorted({64, online_batch}):
        features = extractor(torch.zeros((batch_size, 3, 224, 224), device=args.device)).float()
        for variant in VARIANTS.values():
            window = FeatureLogitWindow(head, variant.window)
            logits = window.append(features, allow_oversize=True)
            PLEADLite(window, variant.fit(), variant.components).update(
                window, batch_size, batch_logits=logits, audit=True,
            )
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    for suite in ("standard", "online_ir"):
        args.batch_size = 64 if suite == "standard" else online_batch
        raw = args.output / "data" / "raw" / suite / args.dataset
        for corruption in tqdm(args.corruptions, desc=f"{args.dataset} {suite}", unit="corruption"):
            if suite == "standard":
                # Standard split and image-order seed match the earlier Balanced ablation protocol.
                for seed in args.seeds:
                    dataset, labels, ids = data_for(args, corruption, seed, processor)
                    stream = ablation_stream(labels, classes, seed, 512, VALIDATION_PER_CLASS[args.dataset])
                    splits = dict(adaptation=[stream.adaptation],
                                  balanced_validation=stream.validation["balanced_validation"],
                                  adaptation_sample_ids=[ids[index] for index in stream.adaptation],
                                  validation_sample_ids=[ids[index] for index in stream.validation["balanced_validation"]],
                                  sample_id_sha256=digest(ids))
                    cache = ArrivalFeatures(dataset, extractor, head, args.device)
                    save_record(args, suite, corruption, seed, None, splits,
                                experiments(args.dataset, None), cache, labels, head)
                    del cache
            else:
                dataset, labels, ids = data_for(args, corruption, 2020, processor)
                sample_hash = digest(ids)
                cache = ArrivalFeatures(dataset, extractor, head, args.device)
                index_path = raw / corruption / "sample_ids.json.gz"
                index_path.parent.mkdir(parents=True, exist_ok=True)
                if not index_path.exists():
                    write_record(index_path, ids)
                else:
                    with gzip.open(index_path, "rt") as handle:
                        if json.load(handle) != ids:
                            raise ValueError(f"Sample ID mapping changed at {index_path}")
                for seed in args.seeds:
                    for ir in IR_LEVELS[args.dataset]:
                        stream = make_stream(labels, classes, subset_size, ir, seed)
                        reference = Path("results/online_ir") / args.dataset / "runs" / corruption / str(seed) / f"ir_{ir}.json.gz"
                        if reference.is_file():
                            with gzip.open(reference, "rt") as handle:
                                sampling = json.load(handle)["sampling"]
                            if (sampling["sha256"], sampling["sample_id_sha256"]) != (stream.sha256, sample_hash):
                                raise ValueError(f"Online IR order differs from verified historical manifest {reference}")
                        count = 2 if args.smoke else len(stream.indices)
                        splits = dict(adaptation=stream.indices[:count].tolist(), sample_id_sha256=sample_hash,
                                      full_sampling_sha256=stream.sha256)
                        ir_audit = dict(dominant_classes=stream.dominant_classes[:count].tolist(),
                                        histograms=[[[int(c), int(row[c])] for c in np.flatnonzero(row)]
                                                    for row in stream.counts[:count]],
                                        p_max=stream.p_max, p_min=stream.p_min, repeated_images=0)
                        save_record(args, suite, corruption, seed, ir, splits,
                                    experiments(args.dataset, ir), cache, labels, head, ir_audit)
                del cache


def main():
    args = arguments()
    if args.report_only:
        from cap_experiments.ablation_reporting import export

        export(args.output, args.smoke)
        return
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{args.dataset}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"An experiment already owns {args.dataset} in {args.output}") from None
        with (logs / f"{args.dataset}.log").open("a", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    run(args)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None


if __name__ == "__main__":
    main()
