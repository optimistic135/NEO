"""Run a full Online IR protocol for one dataset, or a two-subset smoke run."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

from cap_experiments.ir_evaluation import ArrivalFeatures, benchmark, evaluate_online, sparse_histogram
from cap_experiments.online_ir import SPECS, make_stream
from cap_experiments.protocol import load_dataset, load_imagenet_dataset
from run_cap_neo import SEEDS, Tee, _configuration, all_corruptions, load_backbone, write_json
from run_plead_v2 import MODELS, preflight
from tta_library.cap_neo import CAPConfig
from tta_library.plead_v2 import V2Config
from utils.utils import set_reproducible


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SPECS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.data = args.data.resolve()
    if not args.data.is_dir():
        parser.error(f"Missing data directory {args.data}")
    args.output = args.output.resolve()
    args.device = torch.device(args.device)
    args.level = 5
    args.classes, args.subset_size, args.batch_size, args.irs = SPECS[args.dataset]
    args.seeds = [2020] if args.smoke else list(SEEDS)
    args.corruptions = ["gaussian_noise"] if args.smoke else list(all_corruptions)
    args.smoke_subsets = 2
    return args


def write_record(path, record):
    pending = path.with_suffix(".tmp")
    with gzip.open(pending, "wt", compresslevel=2) as handle:
        json.dump(record, handle, allow_nan=False, separators=(",", ":"))
    pending.replace(path)


def _load_data(args, corruption, processor):
    set_reproducible(2020)
    if args.dataset == "imagenet-c":
        return load_imagenet_dataset(args.data, corruption, args.level)
    return load_dataset(args.data, corruption, args.level, processor, args.classes)


def run(args):
    preflight(args)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index if args.device.index is not None else 0)
    model, args.model_revision, processor_id, args.processor_revision = MODELS[args.dataset]
    v2 = V2Config(max_bank=min(128, args.subset_size))
    config = _configuration(args, CAPConfig(), model, processor_id)
    config["arguments"]["irs"] = list(args.irs)
    fit = asdict(v2)
    fit.pop("augmentation")
    config.update(lite=fit, protocol="batch_transductive", view="single_original",
                  reuse="lazily populated frozen features, current arrival IDs only")
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Saved configuration changed. Use a new output directory")
    write_json(path, config)
    set_reproducible(2020)
    neo, extractor, processor = load_backbone(args, model, processor_id)
    classifier = neo.classifier
    bench_path = args.output / "benchmark.json"
    with torch.no_grad():
        # Warm up GPU kernels for the adapter benchmark and cache-backed Online runs.
        if not bench_path.is_file():
            dataset, labels, _ = _load_data(args, "gaussian_noise", processor)
            bench_stream = make_stream(labels, args.classes, args.subset_size, 1, 2020)
            limit = (args.subset_size + args.batch_size - 1) // args.batch_size * 2
            batches = list(bench_stream.batches(args.batch_size))[:limit]
            write_json(bench_path, benchmark(dataset, batches, extractor, classifier, args))
            del dataset

        for corruption in tqdm(args.corruptions, desc=args.dataset, unit="corruption"):
            dataset, labels, ids = _load_data(args, corruption, processor)
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate sample identifiers for {corruption}")
            sample_id_sha256 = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            index_path = args.output / "runs" / corruption / "sample_ids.json.gz"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            if index_path.exists():
                with gzip.open(index_path, "rt") as handle:
                    if json.load(handle) != ids:
                        raise ValueError(f"Sample ID mapping changed for {corruption}")
            else:
                write_record(index_path, ids)
            cache = ArrivalFeatures(dataset, extractor, classifier, args.device)
            for seed in args.seeds:
                for ir in args.irs:
                    base = args.output / "runs" / corruption / str(seed)
                    base.mkdir(parents=True, exist_ok=True)
                    path = base / f"ir_{ir}.json.gz"
                    stream = make_stream(labels, args.classes, args.subset_size, ir, seed)
                    if path.exists():
                        with gzip.open(path, "rt") as handle:
                            previous = json.load(handle)
                        if (previous["sampling"]["sha256"] != stream.sha256
                                or previous["sampling"]["sample_id_sha256"] != sample_id_sha256):
                            raise ValueError(f"Saved sample order changed for {path}")
                        continue

                    description = f"{args.dataset}/{corruption}/seed={seed}/IR={ir}"
                    rows, subsets, steps = evaluate_online(stream, labels, cache, classifier, args, v2, description)
                    evaluated = stream.indices[:len(subsets)]
                    expected = len(evaluated) * args.subset_size
                    if any(row["samples"] != expected for row in rows):
                        raise RuntimeError(f"Online predictions missing in {description}")
                    counts = sparse_histogram(labels[evaluated.ravel()], args.classes)
                    identity = dict(dataset=args.dataset, seed=seed, corruption=corruption, ir=ir)
                    for row in rows:
                        row.update(**identity, subsets=len(subsets), sampling_sha256=stream.sha256,
                                   sample_id_sha256=sample_id_sha256)
                    record = {
                        "identity": identity,
                        "sampling": dict(
                            sha256=stream.sha256, index_sha256=stream.index_sha256,
                            sample_id_sha256=sample_id_sha256, full_size=stream.indices.size,
                            evaluated_size=expected, p_max=stream.p_max, p_min=stream.p_min,
                            repeated_images=0, evaluated_class_counts=counts,
                            class_order=stream.class_order.tolist(),
                            dominant_classes=stream.dominant_classes.tolist(),
                            indices=stream.indices.tolist(),
                        ),
                        "rows": rows, "subsets": subsets, "diagnostics": steps,
                    }
                    write_record(path, record)
                    tqdm.write(f"{description} NEO={rows[0]['top1']:.3f} Lite={rows[1]['top1']:.3f}")
            del cache
    from cap_experiments.ir_reporting import export_report

    export_report([args.output], args.output)


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
