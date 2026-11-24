import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import torch

from src.datasets.cifar_c import load_cifar_c
from src.engine.benchmark_report import summarize
from src.engine.evaluator import evaluate, sync
from src.utils.seed import seed_everything
from src.methods.registry import setup
from src.models.registry import MODELS
from src.protocols.balanced_corruption import build_split, digest
from src.utils.official import official_file
from src.utils.settings import HOME, REPO, config, revision, save
from src.utils.source_stats import clean_digest


def experiment(name, output=None):
    cfg = config(name)
    dataset, protocol = config(cfg["dataset"]), config(cfg["protocol"])
    if protocol["type"] != "balanced_corruption":
        raise ValueError("This runner supports the balanced_corruption protocol")
    method_cfg = {m: config(f"methods/{m}.yaml") for m in cfg["methods"]}
    destination = Path(output or HOME / cfg["output"]).resolve()
    return cfg, dataset, protocol, method_cfg, destination


def validate(cfg, dataset, protocol, device):
    start = perf_counter()
    model, transform, audit = MODELS[cfg["model"]](config(cfg["model"]), cfg["model_profile"], dataset["classes"])
    model.to(device)
    sync(device)
    elapsed = perf_counter() - start
    files = {}
    for corruption in protocol["corruptions"]:
        load_cifar_c(dataset, corruption, protocol["severity"], transform)
        path = Path(dataset["root"]) / f"{corruption}.npy"
        stat = path.stat()
        files[corruption] = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return model, transform, audit, elapsed, files


def run(args):
    cfg, dataset, protocol, method_cfg, destination = experiment(args.config, args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = run_locked(args, cfg, dataset, protocol, method_cfg, destination)
        if not args.validate:
            summarize(destination)
        return status


def run_locked(args, cfg, dataset, protocol, method_cfg, destination):
    selected = args.methods or cfg["methods"]
    if not set(selected) <= method_cfg.keys():
        raise ValueError("Selected method is outside this experiment")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected Python environment")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        model, transform, audit, model_seconds, files = validate(cfg, dataset, protocol, device)
    except Exception as error:
        save(destination / "preflight.json", dict(status="failed", error=str(error),
                                                 traceback=traceback.format_exc()))
        raise
    source_file = Path(dataset["clean_root"]) / "cifar-100-python/train"
    source_hash = clean_digest(dataset["clean_root"]) if source_file.is_file() else None
    checkout_git = revision(REPO)
    deployed = HOME / ".deployed-revision"
    code_git = deployed.read_text().strip() if deployed.is_file() else checkout_git
    snapshot = dict(git=code_git, checkout_git=checkout_git, source_sha256=source_hash, checkpoint=audit, dataset=dataset, protocol=protocol,
                    dataset_audit=files, methods=method_cfg, experiment=cfg,
                    dependencies={k: version(k) for k in ("torch", "torchvision", "transformers",
                                                            "numpy", "matplotlib", "timm")},
                    cuda=torch.version.cuda, device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    snapshot_path = destination / "configuration.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != json.loads(json.dumps(snapshot)):
        raise ValueError(f"Configuration or checkpoint changed. Select a new --output; existing run is {snapshot_path}")
    save(snapshot_path, snapshot)
    save(destination / "preflight.json", dict(status="ok", model_load_seconds=model_seconds))
    if args.validate:
        return 0
    context = dict(clean_root=dataset["clean_root"], transform=transform, audit=audit,
                   output=destination, batch_size=protocol["batch_size"], source_sha256=source_hash)
    with torch.no_grad():
        model(torch.zeros(protocol["batch_size"], 3, 224, 224, device=device))
    sync(device)
    failures = False
    for name in selected:
        log = destination / "logs" / f"{name}.json"
        settings = method_cfg[name]
        try:
            for file in settings.get("upstream", {}).get("files", {}):
                official_file(settings, file)
            if name in {"foa", "fozo"} and not source_hash:
                raise FileNotFoundError(f"Clean source training file required: {source_file}")
            # Smoke and pilot gate each method separately. Pilot streams are reused in full coverage.
            first = protocol["corruptions"][0]
            if not run_one(model, transform, dataset, protocol, device, destination, name, settings,
                           context, first, 2020, smoke=True, model_seconds=model_seconds):
                raise RuntimeError("Smoke test failed")
            for seed in protocol["seeds"]:
                if not run_one(model, transform, dataset, protocol, device, destination, name, settings,
                               context, first, seed, smoke=False, model_seconds=model_seconds):
                    raise RuntimeError(f"Pilot corruption failed at seed {seed}")
            ready = True
            if not args.pilot:
                for corruption in protocol["corruptions"][1:]:
                    for seed in protocol["seeds"]:
                        ready &= run_one(model, transform, dataset, protocol, device, destination, name, settings,
                                         context, corruption, seed, smoke=False, model_seconds=model_seconds)
            status = ("pilot_complete" if args.pilot else "complete") if ready else "partial"
            save(log, dict(status=status))
            failures |= not ready
        except Exception as error:
            save(log, dict(status="failed", error=str(error), traceback=traceback.format_exc()))
            print(f"{name}: {error}", flush=True)
            failures = True
    return int(failures)


def run_one(model, transform, dataset, protocol, device, destination, name, settings, context,
            corruption, seed, smoke, model_seconds):
    data = load_cifar_c(dataset, corruption, protocol["severity"], transform)
    adapt, valid = build_split(data.targets, dataset["classes"], seed, protocol["adaptation"], protocol["validation"])
    key = f"{corruption}-{seed}"
    index_path = destination / "splits" / f"{key}.json"
    indices = dict(adaptation=adapt.tolist(), validation=valid.tolist(),
                   adaptation_sha256=digest(adapt), validation_sha256=digest(valid))
    if index_path.exists() and json.loads(index_path.read_text()) != indices:
        raise ValueError(f"Split changed: {index_path}")
    if not index_path.exists():
        save(index_path, indices)
    path = destination / ("smoke" if smoke else "raw") / name / f"{key}.json"
    if path.exists() and json.loads(path.read_text())["status"] == "ok":
        return True
    batch_size = protocol["batch_size"]
    if smoke:
        adapt, valid = adapt[:batch_size], valid[:batch_size]
    identity = dict(method=name, corruption=corruption, seed=seed,
                    checkpoint_sha256=context["audit"]["sha256"], adaptation_sha256=digest(adapt),
                    validation_sha256=digest(valid))
    title = f"{name} / {key}"
    try:
        seed_everything(seed)
        start = perf_counter()
        method = setup(model, settings, context)
        method.reset()
        sync(device)
        source = getattr(method, "source_seconds", 0.)
        init = max(0., perf_counter() - start - source)
        a = evaluate(method, data, adapt, batch_size, device, "adaptation", title + " adaptation")
        v = evaluate(method, data, valid, batch_size, device, "validation", title + " validation")
        state = method.get_state()
        if smoke:
            method.reset()
            counters = method.get_state()
            if counters["forward_passes"] or counters["backward_passes"]:
                raise ValueError("Method did not reset its counters")
        save(path, dict(**identity, status="ok", adaptation=a, validation=v, state=state,
                        source_seconds=source, model_load_seconds=model_seconds, setup_seconds=init))
        print(f"{title}: adaptation {a['top1']:.2f}% · validation {v['top1']:.2f}%", flush=True)
        return True
    except Exception as error:
        save(path, dict(**identity, status="failed", error=str(error), traceback=traceback.format_exc()))
        print(f"{title}: {error}", flush=True)
        return False
