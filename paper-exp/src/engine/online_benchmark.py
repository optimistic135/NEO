"""Evaluate official baselines using the saved PEA/PLEAD Online IR protocol."""

import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from src.data.dart import data_digest
from src.data.streams import load_data
from src.engine.evaluator import evaluate, sync
from src.methods.eata import SourceRequired
from src.methods.registry import setup
from src.models.loading import load
from src.utils.seed import seed_everything
from src.utils.settings import HOME, REPO, config, revision, save, sha256

# The index generator is shared verbatim with the earlier PEA/PLEAD runner.
sys.path.insert(0, str(REPO / "ir-benchmark"))
from sampling import CapacityError, DESIGN, build_stream


def save_indices(path, stream):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as out:
        np.savez_compressed(out, indices=stream.indices, counts=stream.counts,
                            dominant_classes=stream.dominant_classes, class_order=stream.class_order)
    temporary.replace(path)


def run(args):
    from src.engine.online_report import summarize

    cfg = config("experiments/online_ir/baselines.yaml")
    output = Path(args.output or HOME / cfg["output"]) / args.capacity_policy
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = args.methods or cfg["methods"]
        if not set(selected) <= set(cfg["methods"]):
            raise ValueError("Unknown baseline")
        devices = torch.device(args.device)
        failed = False
        chosen = list(DESIGN) if args.dataset == "all" else [args.dataset]
        for name in chosen:
            try:
                failed |= run_dataset(args, cfg, output, selected, name, devices)
            except Exception as error:
                save(output / "logs" / f"{name}.json", dict(error=str(error), traceback=traceback.format_exc()))
                print(f"{name}: {error}", file=sys.stderr, flush=True)
                failed = True
            summarize(output, cfg, args.reference)
        return int(failed)


def run_dataset(args, cfg, output, selected, name, device):
    classes, subset_size, ratios = DESIGN[name]
    spec = dict(cfg["datasets"][name], classes=classes)
    model, transform, audit = load(name, "vit_base", classes)
    if audit["sha256"] != spec["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}")
    model.to(device)
    clean = args.imagenet_clean if name == "imagenet-c" else Path(cfg["clean_cifar_root"])
    if name == "imagenet-c" and clean:
        corrupted = Path(spec["root"]).resolve()
        if clean.resolve().is_relative_to(corrupted) or corrupted.is_relative_to(clean.resolve()):
            raise ValueError("ImageNet source must be independent of ImageNet-C")
    corruptions = config("datasets/corruptions.yaml")["corruptions"]
    seed_runs = cfg["seeds"]
    method_cfg = {method: config(f"methods/{method}.yaml") for method in cfg["methods"]}
    pinned = HOME / ".deployed-revision"
    code_git = pinned.read_text().strip() if pinned.exists() else revision(REPO)
    snapshot = dict(code_git=code_git, checkpoint=audit, capacity=args.capacity_policy, spec=spec,
                    batch_size=cfg["batch_size"], severity=cfg["severity"], ratios=ratios,
                    seeds=seed_runs, source_root=str(clean) if clean is not None else None,
                    sampler_sha256=sha256(REPO / "ir-benchmark/sampling.py"),
                    cifar_sampler_sha256=sha256(REPO / "cap_experiments/online_ir.py"),
                    methods=method_cfg, device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    versions={p: version(p) for p in ("torch", "torchvision", "transformers", "timm", "numpy")})
    path = output / "configuration" / f"{name}.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(snapshot)):
        raise ValueError(f"Configuration changed for {name}; use a fresh --output")
    save(path, snapshot)
    with torch.no_grad():
        model(torch.zeros(cfg["batch_size"], 3, 224, 224, device=device))
    sync(device)
    context = dict(dataset=name, clean_root=clean, transform=transform, batch_size=cfg["batch_size"], fisher_cache={})

    def phase(smoke):
        failed = False
        seeds = (2020,) if smoke else seed_runs
        for corruption in corruptions[:1] if smoke else corruptions:
            data, ids = load_data(spec, corruption, cfg["severity"], transform)
            ids_sha = data_digest(ids)
            for seed in seeds:
                context["seed"] = seed
                for ir in ratios:
                    key = f"{corruption}-{seed}-{ir}"
                    dest = output / ("smoke" if smoke else "full") / "raw" / name / key
                    base = dict(dataset=name, seed=seed, corruption=corruption, ir=ir,
                                checkpoint_sha256=audit["sha256"], dataset_ids_sha256=ids_sha)
                    try:
                        stream, capped = build_stream(data.targets, name, ir, seed, args.capacity_policy)
                    except CapacityError as error:
                        for method in selected:
                            save(dest / f"{method}.json", dict(**base, method=method, status="capacity_blocked", error=str(error)))
                        continue
                    indices = stream.indices[:2] if smoke else stream.indices
                    active_count = len(indices)
                    batch_list = list(stream.batches(cfg["batch_size"]))
                    batch_list = batch_list[:active_count * ((subset_size + cfg["batch_size"] - 1) // cfg["batch_size"])]
                    base.update(stream_sha256=stream.sha256,
                                evaluated_indices_sha256=index_hash(indices),
                                p_max=stream.p_max, p_min=stream.p_min, capped=capped,
                                realized_dominant_fraction=float(stream.counts[np.arange(active_count), stream.dominant_classes[:active_count]].mean() / subset_size),
                                repeat_across_subsets=int(indices.size - np.unique(indices).size))
                    split = output / ("smoke" if smoke else "full") / "splits" / name / f"{key}.npz"
                    if not split.exists():
                        save_indices(split, stream)
                    for method in selected:
                        path = dest / f"{method}.json"
                        row = dict(**base, method=method)
                        if path.exists():
                            prior = json.loads(path.read_text())
                            if prior["status"] == "ok":
                                if any(prior[field] != row[field] for field in ("checkpoint_sha256", "dataset_ids_sha256", "stream_sha256", "evaluated_indices_sha256")):
                                    raise ValueError(f"Resumed stream mismatch: {path}")
                                continue
                        if method == "eata" and name == "imagenet-c" and clean is None:
                            save(path, dict(**row, status="source_blocked", error="Independent clean ImageNet for Fisher unavailable"))
                            continue
                        if not smoke:
                            preflight = output / "smoke" / "raw" / name / f"{corruptions[0]}-2020-{ir}" / f"{method}.json"
                            if not preflight.exists() or json.loads(preflight.read_text())["status"] != "ok":
                                save(path, dict(**row, status="smoke_blocked", error="Method/IR smoke test did not pass"))
                                failed = True
                                continue
                        try:
                            seed_everything(seed)
                            start = perf_counter()
                            adapter = setup(model, method_cfg[method], context)
                            adapter.reset()
                            sync(device)
                            seconds = perf_counter() - start
                            source_seconds = getattr(adapter, "source_seconds", 0.)
                            label = f"{name}/{method}/{key}"
                            metrics = evaluate(adapter, data, None, cfg["batch_size"], device, "adaptation", label,
                                               batch_sampler=batch_list)
                            if metrics["n"] != indices.size:
                                raise ValueError("Incomplete evaluated stream")
                            save(path, dict(**row, status="ok", top1=metrics["top1"], samples=metrics["n"],
                                            batch_ms=metrics["latency_ms"], batches=len(metrics["batches"]),
                                            throughput=metrics["throughput"], peak_gpu_mib=metrics["peak_gpu_mib"],
                                            setup_seconds=max(0.,seconds-source_seconds), source_seconds=source_seconds,
                                            state=adapter.get_state(), batch_details=metrics["batches"]))
                            print(f"{label}: {metrics['top1']:.2f}%", flush=True)
                        except SourceRequired as error:
                            save(path, dict(**row, status="source_blocked", error=str(error)))
                        except Exception as error:
                            save(path, dict(**row, status="failed", error=str(error), traceback=traceback.format_exc()))
                            print(f"{name}/{method}/{key}: {error}", file=sys.stderr, flush=True)
                            failed = True
                        finally:
                            adapter = None
        return failed

    failed = phase(smoke=True)
    if not args.smoke_only:
        failed |= phase(smoke=False)
    return failed


def index_hash(indices):
    from hashlib import sha256 as digest
    return digest(indices.astype("<i8", copy=False).tobytes()).hexdigest()
