import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(REPO / "balanced-512")]

import numpy as np
import torch
from torch.utils.data import DataLoader

from run import BATCH, SEEDS, FULL, FrozenPEA, PLEAD, config, load, load_data, run_phase, save, seed_everything, sync
from sampling import CapacityError, DESIGN, build_stream
from src.data.dart import data_digest
from src.utils.settings import revision


def write_split(path, stream):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fp:
        np.savez_compressed(fp, indices=stream.indices, dominant_classes=stream.dominant_classes,
                            class_order=stream.class_order, counts=stream.counts)
    tmp.replace(path)


def methods(dataset):
    return ("plead",) if dataset == "imagenet-c" else ("pea", "plead")


def run(args):
    cfg = config("datasets/corruptions.yaml")
    model_spec = dict(cfg["datasets"][args.dataset], pea_source=str(args.cifar_source))
    classes, subset_size, ratios = DESIGN[args.dataset]
    corruptions = ["gaussian_noise"] if args.smoke else cfg["corruptions"]
    seeds = (2020,) if args.smoke else SEEDS
    root = args.output / args.capacity_policy / ("smoke" if args.smoke else "full")
    device = torch.device(args.device)
    pea_cfg = dict(config("methods/official.yaml")["pea"], source_samples=50000, augment_microbatch=1)
    audit_path = root / "configuration" / f"{args.dataset}.json"
    model, transform, audit = load(args.dataset, "vit_base", classes)
    snapshot = dict(dataset=args.dataset, severity=5, subset_size=subset_size, subsets=classes,
                    ratios=ratios, seeds=seeds, batch_size=BATCH, no_batch_crossing=True,
                    capacity_policy=args.capacity_policy, checkpoint=audit, source=str(args.cifar_source),
                    checkout_git=revision(REPO), pea=pea_cfg,
                    pea_upstream=config("methods/official.yaml")["upstream"]["pea"],
                    code_sha256={str(p.relative_to(REPO)): sha256(p.read_bytes()).hexdigest()
                                 for p in (HERE / "evaluate.py", HERE / "sampling.py",
                                           REPO / "balanced-512/run.py", REPO / "cap_experiments/online_ir.py")},
                    torch_cuda=torch.version.cuda, device=str(device),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    dependencies={k: version(k) for k in ("torch", "torchvision", "timm", "transformers", "numpy")})
    if audit_path.exists() and json.loads(audit_path.read_text()) != json.loads(json.dumps(snapshot)):
        raise ValueError(f"Configuration changed for {audit_path}. Use a new --output directory.")
    save(audit_path, snapshot)
    model.to(device)
    with torch.no_grad():
        model(torch.zeros(BATCH, 3, 224, 224, device=device))
    sync(device)
    failed = False
    for corruption in corruptions:
        data, ids = load_data(model_spec, corruption, 5, transform)
        ids_hash = data_digest(ids)
        for seed in seeds:
            for ratio in ratios:
                key = f"{corruption}-{seed}-{ratio}"
                if not args.smoke:
                    smoke_dir = args.output / args.capacity_policy / "smoke/raw" / args.dataset
                base = dict(dataset=args.dataset, corruption=corruption, seed=seed, ir=ratio,
                            checkpoint_sha256=audit["sha256"], dataset_ids_sha256=ids_hash)
                folder = root / "raw" / args.dataset / key
                try:
                    stream, capped = build_stream(data.targets, args.dataset, ratio, seed, args.capacity_policy)
                except CapacityError as error:
                    for method in methods(args.dataset):
                        save(folder / f"{method}.json", dict(**base, method=method, status="capacity_blocked", error=str(error)))
                    print(f"{args.dataset} {key}: {error}", flush=True)
                    continue
                selection = list(stream.batches(BATCH))
                active = stream.indices[:2] if args.smoke else stream.indices
                if args.smoke:
                    selection = selection[:2 * ((subset_size + BATCH - 1) // BATCH)]
                active_hash = sha256(active.astype("<i8", copy=False).tobytes()).hexdigest()
                base.update(stream_sha256=stream.sha256, evaluated_indices_sha256=active_hash,
                            p_max=stream.p_max, p_min=stream.p_min, capped=capped,
                            realized_dominant_fraction=float(stream.counts[np.arange(len(active)), stream.dominant_classes[:len(active)]].mean() / subset_size),
                            repeat_across_subsets=int(active.size - np.unique(active).size), subsets=len(active))
                split_path = root / "splits" / args.dataset / f"{key}.npz"
                if not split_path.exists():
                    write_split(split_path, stream)
                for method in methods(args.dataset):
                    path = folder / f"{method}.json"
                    row = dict(**base, method=method)
                    if path.exists() and json.loads(path.read_text())["status"] == "ok":
                        old = json.loads(path.read_text())
                        if any(old[k] != row[k] for k in ("checkpoint_sha256", "dataset_ids_sha256", "evaluated_indices_sha256", "stream_sha256")):
                            raise ValueError(f"Completed stream has incompatible provenance: {path}")
                        continue
                    if not args.smoke:
                        smoke = smoke_dir / f"gaussian_noise-2020-{ratio}" / f"{method}.json"
                        if not smoke.exists() or json.loads(smoke.read_text())["status"] != "ok":
                            save(path, dict(**row, status="smoke_blocked", error="Method and IR did not pass smoke test"))
                            failed = True
                            continue
                    try:
                        seed_everything(seed)
                        start = perf_counter()
                        adapter = (FrozenPEA(model, args.dataset, model_spec, transform, seed,
                                             args.upstream, pea_cfg, args.smoke)
                                   if method == "pea" else PLEAD(model, FULL))
                        sync(device)
                        setup = perf_counter() - start
                        loader = DataLoader(data, batch_sampler=selection, pin_memory=device.type == "cuda")
                        title = f"{args.dataset} {method} {key}"
                        result = run_phase(loader, adapter.update, adapter, device, classes, title)
                        if result["samples"] != active.size or result["batches"] != len(selection):
                            raise ValueError("Online scoring did not cover the planned batches")
                        save(path, dict(**row, status="ok", setup_seconds=setup, **result))
                        print(f"{title}: {result['top1']:.2f}% on {result['samples']} online images", flush=True)
                    except Exception as error:
                        save(path, dict(**row, status="failed", error=str(error)))
                        print(f"{args.dataset} {method} {key}: {error}", file=sys.stderr, flush=True)
                        failed = True
                    finally:
                        adapter = None
    return int(failed)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=DESIGN)
    cli.add_argument("--output", type=Path, default=REPO / "results/ir-benchmark")
    cli.add_argument("--upstream", type=Path, default=REPO / "paper-exp/.upstream")
    cli.add_argument("--cifar-source", type=Path, default=Path("/data/user3/chenc/datasets/CIFAR-clean"))
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--capacity-policy", choices=("strict", "cap"), default="strict")
    cli.add_argument("--smoke", action="store_true")
    raise SystemExit(run(cli.parse_args()))
