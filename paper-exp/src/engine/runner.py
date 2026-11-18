from collections import Counter, defaultdict
from importlib.metadata import version
import random
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cap_experiments.simplified_protocol import VARIANTS
from src.data.dart import data_digest, load_manifest
from src.data.streams import balanced, load_data
from src.methods.bn_adapt import BNAdapt
from src.methods.neo import NEO
from src.methods.no_adapt import NoAdapt
from src.methods.pea import PEA
from src.methods.plead import PLEAD
from src.methods.roid import ROID
from src.methods.sar import SAR
from src.methods.t3a import T3A
from src.methods.tent import TENT
from src.metrics.scoring import Metrics
from src.models.loading import load
from src.utils.settings import REPO, config, revision, save
from src.utils.upstream import load_upstream


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def adapter_for(method, model, dataset, spec, transform, seed, upstream, cfg, smoke):
    if method in VARIANTS:
        return PLEAD(model, VARIANTS[method])
    if method in ("no_adapt", "bn_adapt", "neo"):
        return {"no_adapt": NoAdapt, "bn_adapt": BNAdapt, "neo": NEO}[method](model)
    imagenet = dataset == "imagenet-c"
    if method == "tent":
        return TENT(model, cfg[method], imagenet)
    if method == "sar":
        return SAR(model, cfg[method], spec["batch"])
    if method == "t3a":
        return T3A(model, cfg[method]["filter_k"])
    if method == "pea":
        return PEA(model, dataset, spec, transform, seed, upstream, cfg[method], smoke)
    if method == "roid":
        upstream_aug = load_upstream(upstream, "roid", "classification/augmentations/transforms_cotta.py")
        transform_aug = upstream_aug.get_tta_transforms(224, padding_mode="reflect", cotta_augs=False)
        def augment(images):
            mean, std = (images.new_tensor(x)[None, :, None, None] for x in model.normalization)
            return (transform_aug(images*std+mean)-mean)/std
        return ROID(model, cfg[method], imagenet, spec["batch"], augment)
    raise ValueError(method)


def batches(indices, batch):
    blocks = indices if indices.ndim == 2 else [indices]
    return [sub[start:start+batch].tolist() for sub in blocks for start in range(0, len(sub), batch)]


@torch.no_grad()
def score(adapter, loader, classes, device, warmup=0):
    adapter.reset()
    if warmup:
        for i, (images, _) in enumerate(loader):
            if i == warmup:
                break
            adapter.update(images.to(device))
        sync(device)
        adapter.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metric = Metrics(classes)
    reasons = Counter()
    diagnostics = defaultdict(list)
    adapter_seconds = 0.
    count = 0
    sync(device)
    start = perf_counter()
    for images, labels in tqdm(loader, leave=False, unit="batch", mininterval=2):
        time = perf_counter()
        logits = adapter.update(images.to(device, non_blocking=True))
        sync(device)
        adapter_seconds += perf_counter()-time
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite logits at batch {count}")
        metric.add(logits, labels)
        events = getattr(adapter, "events", {})
        if events.get("fallback"):
            reasons[events["fallback"]] += 1
        for name in ("kappa", "gamma", "valid_fit"):
            if name in events:
                diagnostics[name].append(events[name])
        count += 1
    sync(device)
    seconds = perf_counter()-start
    metrics = metric.result()
    return dict(**metrics, update_timing=adapter.timing, batches=count,
                end_to_end_seconds=seconds, method_seconds=adapter_seconds,
                throughput=metrics["samples"]/seconds,
                peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
                forward_passes=adapter.forward_passes, backward_passes=adapter.backward_passes,
                trainable_parameters=adapter.trainable_parameters, fallback_rate=sum(reasons.values())/count,
                fallback_reasons=dict(reasons), **{k: float(np.mean(v)) for k,v in diagnostics.items()})


def run(args):
    data_cfg = config("datasets/corruptions.yaml")
    balanced_cfg = config("protocols/balanced.yaml")
    online = config("protocols/dart_online_ir.yaml")
    methods = config("methods/official.yaml")
    method_key = {"balanced": "methods", "ablation": "ablations", "efficiency": "efficiency"}
    requested = args.methods or (online["methods"] if args.stage == "online_ir" else balanced_cfg.get(method_key.get(args.stage), []))
    online_stage = args.stage == "online_ir" or (args.stage == "ablation" and args.online)
    seeds = [data_cfg["seeds"][0]] if args.smoke else data_cfg["seeds"]
    device = torch.device(args.device)
    snapshot = dict(git=revision(REPO), dependencies={p: version(p) for p in
                    ("torch", "torchvision", "timm", "transformers", "numpy", "PyYAML")},
                    data=data_cfg, balanced=balanced_cfg, online=online, methods=methods, precision="float32")
    for dataset in args.datasets:
        spec = data_cfg["datasets"][dataset]
        for arch in args.architectures:
            try:
                model, transform, audit = load(dataset, arch, spec["classes"])
                model.to(device)
            except Exception as error:
                save(args.output / "checkpoints" / f"{dataset}-{arch}.json",
                     dict(status="failed", error=str(error), traceback=traceback.format_exc()))
                print(f"{dataset}/{arch} checkpoint failed: {error}", flush=True)
                continue
            save(args.output / "checkpoints" / f"{dataset}-{arch}.json", dict(status="ok", **audit))
            if args.stage == "validate":
                continue
            corruptions = ([balanced_cfg["efficiency_corruption"]] if args.smoke or args.stage == "efficiency"
                           else data_cfg["corruptions"])
            ratios = online["datasets"][dataset]["ratios"] if online_stage else [None]
            for seed in seeds:
                unsupported = {}
                for corruption in corruptions:
                    data, ids = load_data(spec, corruption, data_cfg["severity"], transform)
                    sample_hash = data_digest(ids)
                    for ratio in ratios:
                        key = f"{corruption}-{seed}-{ratio}"
                        split = args.output / "splits" / dataset / f"{key}.json"
                        try:
                            if ratio is None:
                                indices, provenance = balanced(len(data), seed), {}
                            else:
                                indices, provenance = load_manifest(args.dart_manifests, dataset, seed, ratio, ids,
                                     online["datasets"][dataset]["subset"], spec["classes"])
                        except (FileNotFoundError, ValueError, KeyError) as error:
                            save(args.output / "blocked" / dataset / f"{key}.json", dict(error=str(error)))
                            continue
                        split_sha = data_digest(indices.tolist())
                        split_record = dict(indices=indices.tolist(), sample_ids=ids, sample_ids_sha256=sample_hash,
                                            provenance=provenance, split_sha256=split_sha,
                                            repeated_images=int(indices.size-np.unique(indices).size))
                        if not split.exists():
                            save(split, split_record)
                        selections = batches(indices, spec["batch"])
                        if args.smoke or args.stage == "efficiency":
                            limit = 2 if args.smoke else balanced_cfg["efficiency_batches"]
                            selections = selections[:limit]
                        actual_batch_hash = data_digest(selections)
                        loader = DataLoader(data, batch_sampler=selections, num_workers=balanced_cfg["workers"],
                                            pin_memory=device.type == "cuda", persistent_workers=balanced_cfg["workers"] > 0)
                        for method in requested:
                            row_path = args.output / "raw" / args.stage / dataset / arch / key / f"{method}.json"
                            if row_path.exists():
                                import json
                                old = json.loads(row_path.read_text())
                                if old["checkpoint_sha256"] != audit["sha256"] or old["batch_sha256"] != actual_batch_hash:
                                    raise ValueError(f"Checkpoint or input changed for {row_path}")
                                if old["status"] == "ok":
                                    continue
                            row = dict(dataset=dataset, architecture=arch, corruption=corruption,
                                       severity=data_cfg["severity"], seed=seed, ir=ratio, method=method,
                                       batch_size=spec["batch"], batch_sha256=actual_batch_hash,
                                       split_sha256=split_sha, checkpoint_sha256=audit["sha256"], stage=args.stage,
                                       timing_scope="single_method_fp32", smoke=args.smoke, config=snapshot)
                            if method == "bn_adapt" and arch == "vit_base":
                                save(row_path, dict(**row, status="na", error="ViT has LayerNorm"))
                                continue
                            seed_everything(seed)
                            adapter = None
                            try:
                                if method in unsupported:
                                    raise FileNotFoundError(unsupported[method])
                                adapter = adapter_for(method, model, dataset, spec, transform, seed, args.upstream, methods, args.smoke)
                                warmup = balanced_cfg["warmup"] if args.stage == "efficiency" else 0
                                result = score(adapter, loader, spec["classes"], device, warmup)
                                save(row_path, dict(**row, status="ok", **result))
                                print(f"{args.stage} {dataset} {arch} {key} {method}: {result['top1']:.3f}%", flush=True)
                            except Exception as error:
                                if adapter is None and isinstance(error, FileNotFoundError):
                                    unsupported[method] = str(error)
                                save(row_path, dict(**row, status="failed", error=str(error), traceback=traceback.format_exc()))
                                print(f"{dataset}/{arch}/{method}: {error}", flush=True)
                            finally:
                                del adapter
