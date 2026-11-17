"""Run the Simplified PLEAD studies, resuming at the method/stream level."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import gzip
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

from cap_experiments.ir_evaluation import ArrivalFeatures
from cap_experiments.longtail import ablation_stream
from cap_experiments.online_ir import SPECS, make_stream
from cap_experiments.simplified_evaluation import evaluate
from cap_experiments.simplified_protocol import IR_LEVELS, VALIDATION_PER_CLASS, SUITES, VARIANTS, experiments
from run_cap_neo import SEEDS, Tee, _configuration, all_corruptions, load_backbone, write_json
from run_online_ir import write_record
from run_plead_ablation import FOLDERS, ROOT, data_for, digest
from run_plead_v2 import MODELS, preflight
from tta_library.cap_neo import CAPConfig
from tta_library.plead_lite import FeatureLogitWindow
from tta_library.plead_simplified import PLEAD, MIN_BANK
from utils.utils import set_reproducible


SCHEMA = "simplified-plead-v1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=IR_LEVELS)
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--experiment", choices=("all", *SUITES), default="all")
    p.add_argument("--ir", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    p.add_argument("--corruptions", nargs="+", choices=all_corruptions)
    p.add_argument("--methods", nargs="+", choices=VARIANTS)
    p.add_argument("--cold", action="store_true", help="Single-method, isolated, fresh-cache measurements")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--gzip-csv", action="store_true", help="Compress the large CSVs during reporting")
    a = p.parse_args()
    if not a.report_only and not a.dataset:
        p.error("--dataset is required")
    if a.ir and (not a.dataset or set(a.ir) - set(IR_LEVELS[a.dataset])):
        p.error("--ir must belong to the selected dataset")
    if a.cold and len(a.methods or ["plead"]) != 1:
        p.error("--cold requires a single method")
    if (a.output / "configs" / "method_variants.json").exists():
        p.error("output belongs to the historical ablation. Select a new directory with --output")
    a.output = a.output / "smoke" if a.smoke else a.output
    a.device = torch.device(a.device)
    if a.dataset:
        a.data = (a.data or Path("/data/user3/chenc/datasets") / FOLDERS[a.dataset]).resolve()
    a.seeds = list(dict.fromkeys(a.seeds or ([2020] if a.smoke else SEEDS)))
    a.corruptions = list(dict.fromkeys(a.corruptions or (["gaussian_noise"] if a.smoke else all_corruptions)))
    a.level = 5
    return a


def selected(args, ir):
    groups = experiments(args.dataset, ir)
    if args.experiment != "all":
        groups = {name: variants for name, variants in groups.items() if name == args.experiment}
    methods = dict.fromkeys(name for group in groups.values() for name in group)
    if args.methods or args.cold:
        requested = args.methods or ["plead"]
        methods = {name: None for name in methods if name in requested}
    return methods


def read_record(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def save(args, dataset, corruption, seed, ir, subsets, holdout, labels, ids_hash, head, cache, stream=None):
    methods = selected(args, ir)
    if not methods:
        return
    suite = "standard" if ir is None else "online_ir"
    stem = "standard" if ir is None else f"ir_{ir}"
    base = Path(args.dataset) / corruption / str(seed) / f"{stem}.json.gz"
    raw = args.output / "data" / "raw"
    folder = f"cold/{next(iter(methods))}" if args.cold else suite
    path = raw / folder / base
    manifest = dict(adaptation=subsets, balanced_validation=holdout, sample_ids_sha256=ids_hash,
                    historical_ir_sha256=stream.sha256 if stream else None)
    split_sha = digest(manifest)
    identity = dict(dataset=args.dataset, corruption=corruption, severity=args.level, seed=seed, IR=ir)
    old = read_record(path) if path.exists() else None
    if old and (old["schema"] != SCHEMA or old["split_sha256"] != split_sha or old["identity"] != identity):
        raise ValueError(f"Conflicting saved stream: {path}")
    available = {r["method"] for r in old["rows"]} if old else set()
    pending = {m: VARIANTS[m] for m in methods if m not in available}
    if pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        title = f"{args.dataset}/{corruption}/{seed}/{stem}"
        if cache is None:
            cache = ArrivalFeatures(dataset, args.extractor, head, args.device)
        audited = () if args.cold else experiments(args.dataset, ir).get("mechanism", ())
        rows, updates, subset_rows, priors = evaluate(
            subsets, holdout, labels, cache, head, pending, args, title, audited,
        )
        if ir is not None:
            for row in subset_rows:
                sid = row["subset_id"]
                row["dominant_class"] = int(stream.dominant_classes[sid])
                counts = stream.counts[sid]
                row["true_histogram"] = [[int(k), int(counts[k])] for k in np.flatnonzero(counts)]
        if len(priors):
            prior_path = path.with_name(f"{stem}_soft_prior.npz")
            temp = prior_path.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, soft_prior=priors)
            temp.replace(prior_path)
            for row in updates:
                if "soft_prior_row" in row:
                    row["soft_prior_file"] = str(prior_path.relative_to(args.output))
        record = old or dict(schema=SCHEMA, identity=identity, manifest=manifest, split_sha256=split_sha,
                             rows=[], updates=[], subsets=[])
        for key, values in (("rows", rows), ("updates", updates), ("subsets", subset_rows)):
            record[key].extend(values)
        write_record(path, record)
        available.update(pending)
        tqdm.write(f"{title} saved " + ", ".join(f"{row['method']}={row['top1']:.3f}%" for row in rows))
    if not args.cold:
        for group, group_methods in experiments(args.dataset, ir).items():
            present = [m for m in group_methods if m in available]
            if group != suite and present:
                write_json(raw / group / base.with_name(f"{stem}.json"),
                           dict(raw_record=str(path.relative_to(raw)), methods=present, split_sha256=split_sha))


@torch.no_grad()
def run(args):
    preflight(args)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device.index or 0)
    c, size, online_batch, _ = SPECS[args.dataset]
    model_id, args.model_revision, processor_id, args.processor_revision = MODELS[args.dataset]
    config = _configuration(args, CAPConfig(), model_id, processor_id)
    config.pop("cap")
    config.pop("paired_clean_shift_metrics")
    config["arguments"] = {k: v for k, v in config["arguments"].items()
                           if k in ("dataset", "data", "device", "model_revision", "processor_revision", "level", "smoke")}
    config.update(schema=SCHEMA, batch_online=online_batch, subset_size=size, min_bank=MIN_BANK)
    config_path = args.output / "configs" / f"{args.dataset}.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        # Reporting and subset flags can change on resume. Model identity and data cannot.
        if previous != config:
            raise ValueError(f"Configuration differs at {config_path}")
    else:
        write_json(config_path, config)
    set_reproducible(2020)
    neo, args.extractor, processor = load_backbone(args, model_id, processor_id)
    head = neo.classifier
    for batch_size in sorted({64, online_batch}):
        f = args.extractor(torch.zeros((batch_size, 3, 224, 224), device=args.device)).float()
        for v in VARIANTS.values():
            window = FeatureLogitWindow(head, v.window)
            PLEAD(window, v).update(window, window.append(f))
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    if args.experiment in ("standard", "all") and not args.ir and selected(args, None):
        args.batch_size = 64
        for corruption in tqdm(args.corruptions, desc=f"{args.dataset} standard", unit="corruption"):
            for seed in args.seeds:
                data, labels, ids = data_for(args, corruption, seed, processor)
                split = ablation_stream(labels, c, seed, 512, VALIDATION_PER_CLASS[args.dataset])
                save(args, data, corruption, seed, None, [split.adaptation], split.validation["balanced_validation"],
                     labels, digest(ids), head, None)
    if args.experiment == "standard":
        return
    levels = [ir for ir in (args.ir or IR_LEVELS[args.dataset]) if selected(args, ir)]
    args.batch_size = online_batch
    if not levels:
        return
    for corruption in tqdm(args.corruptions, desc=f"{args.dataset} online", unit="corruption"):
        data, labels, ids = data_for(args, corruption, 2020, processor)
        ids_hash = digest(ids)
        cache = None if args.cold else ArrivalFeatures(data, args.extractor, head, args.device)
        for seed in args.seeds:
            for ir in levels:
                stream = make_stream(labels, c, size, ir, seed)
                reference = Path("results/online_ir") / args.dataset / "runs" / corruption / str(seed) / f"ir_{ir}.json.gz"
                if reference.exists():
                    verified = read_record(reference)["sampling"]
                    if (stream.sha256, ids_hash) != (verified["sha256"], verified["sample_id_sha256"]):
                        raise ValueError(f"Historical sampling order differs: {reference}")
                subsets = (stream.indices[:2] if args.smoke else stream.indices).tolist()
                save(args, data, corruption, seed, ir, subsets, [], labels, ids_hash, head, cache, stream)


def main():
    args = arguments()
    if args.report_only:
        from cap_experiments.simplified_reporting import export

        export(args.output, args.smoke, args.gzip_csv)
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
