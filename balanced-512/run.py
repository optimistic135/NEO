import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / ".runtime" / "paper-exp"
if not RUNTIME.is_dir():
    raise SystemExit("Run bash balanced-512/setup.sh first.")
sys.path[:0] = [str(RUNTIME), str(HERE.parent)]

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cap_experiments.simplified_protocol import FULL
from protocol import split, split_hash
from src.data.dart import data_digest
from src.data.streams import load_data
from src.engine.runner import seed_everything
from src.methods.pea import PEA
from src.methods.plead import PLEAD
from src.metrics.scoring import Metrics
from src.models.loading import load
from src.utils.settings import config, revision, save


SEEDS = (1234, 2020, 9999)
BATCH = 64


class FrozenPEA(PEA):
    @torch.no_grad()
    def update(self, images):
        # Same inference and reset rules as main_vit.py at the pinned revision.
        logits, self.views = self.infer(self.network, self.aligner, images.cpu(),
                                       self.device, microbatch_variants=self.microbatch,
                                       **self.options)
        self.forward_passes += len(self.views) + (len(self.views) + self.microbatch - 1) // self.microbatch
        ent = self.upstream.entropy_from_logits(logits)
        reset = self.stats.entropy_spike(ent)
        if reset:
            self.stats = self.stats.reset()
            self.aligner.stats = self.stats
            self.aligner.reset_state()
            self.aligner.norm_weights = {}
        self.stats.update_entropy(ent)
        self.events = {"fallback": "entropy_reset"} if reset else {}
        return logits

    def freeze(self):
        # Keep the last adaptation batch's weights and final EMA moments.
        # Cache materialization during validation depends only on those moments.
        self.aligner._cache.clear()
        self.aligner.update_every = sys.maxsize
        self.stats.update = lambda *_: False
        align = self.aligner._align_block
        self.aligner._align_block = lambda feat, key: align(feat, key) if key in self.stats.mu else feat

    @torch.no_grad()
    def predict(self, images):
        variants = [self.upstream._apply_variant_cpu(images.cpu(), v, 0.9) for v in self.views]
        outputs = []
        for start in range(0, len(variants), self.microbatch):
            chunk = variants[start:start + self.microbatch]
            outputs.append(self.aligner(torch.cat(chunk).to(self.device)).reshape(len(chunk), len(images), -1))
            self.forward_passes += 1
        return torch.cat(outputs).mean(0)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_phase(loader, predict, adapter, device, classes, title):
    metrics, reasons = Metrics(classes), Counter()
    process = psutil.Process()
    stop = Event()
    peak_rss = process.memory_info().rss

    def sample_rss():
        nonlocal peak_rss
        while not stop.wait(.02):
            peak_rss = max(peak_rss, process.memory_info().rss)

    thread = Thread(target=sample_rss, daemon=True)
    thread.start()
    seconds = []
    passes = adapter.forward_passes
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    try:
        for images, labels in tqdm(loader, desc=title, unit="batch", mininterval=3):
            begin = perf_counter()
            logits = predict(images if isinstance(adapter, PEA) else images.to(device, non_blocking=True))
            sync(device)
            seconds.append(perf_counter() - begin)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits")
            metrics.add(logits, labels)
            if predict == adapter.update and adapter.events.get("fallback"):
                reasons[adapter.events["fallback"]] += 1
    finally:
        stop.set()
        thread.join()
    elapsed = perf_counter() - start
    return dict(**metrics.result(), batches=len(seconds), batch_ms=1e3 * float(np.mean(seconds)),
                batch_ms_p95=1e3 * float(np.quantile(seconds, .95)), end_to_end_seconds=elapsed,
                peak_rss_mib=max(peak_rss, process.memory_info().rss)/2**20,
                peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
                forward_passes=adapter.forward_passes - passes, fallback_reasons=dict(reasons))


def main():
    parser = argparse.ArgumentParser()
    cfg = config("datasets/corruptions.yaml")
    parser.add_argument("--dataset", required=True, choices=cfg["datasets"])
    parser.add_argument("--method", required=True, choices=("pea", "plead"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-root", help="PEA clean dataset root (ImageNet root contains train/).")
    parser.add_argument("--source-samples", type=int, default=50000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.source_samples <= 0:
        parser.error("--source-samples must be positive")
    device = torch.device(args.device)
    spec = dict(cfg["datasets"][args.dataset])
    if args.source_root:
        spec["pea_source"] = args.source_root
    args.output = args.output / ("smoke" if args.smoke else "full")
    status = args.output / "status" / args.dataset / f"{args.method}.json"
    if args.method == "pea" and not spec.get("pea_source"):
        save(status, dict(status="blocked", error="Missing verified clean ImageNet training root (--source-root)"))
        raise SystemExit("PEA requires verified clean ImageNet training images.")
    cfg_pea = dict(config("methods/official.yaml")["pea"], source_samples=args.source_samples, augment_microbatch=1)
    pea_pin = config("methods/official.yaml")["upstream"]["pea"]
    code_hash = sha256(b"".join((HERE / name).read_bytes() for name in ("run.py", "protocol.py"))).hexdigest()
    settings = dict(dataset=args.dataset, method=args.method, severity=5, batch_size=BATCH,
                    adaptation_size=512, seeds=SEEDS, pea=pea_pin, pea_config=cfg_pea if args.method=="pea" else None,
                    data_spec=spec, git=revision(HERE.parent), runtime_pin="73c7669984cf57e58d943d2649c2ad1e240d9b01",
                    code_sha256=code_hash, dependencies={k: version(k) for k in ("torch", "torchvision", "timm", "transformers", "psutil")},
                    device=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu", precision="fp32")
    path = args.output / "config" / args.dataset / f"{args.method}.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(settings)):
        raise SystemExit(f"Configuration changed for {path}. Use a fresh --output directory.")
    save(path, settings)
    save(status, dict(status="running"))
    try:
        model, transform, audit = load(args.dataset, "vit_base", spec["classes"])
        model.to(device)
        save(args.output / "checkpoints" / f"{args.dataset}.json", audit)
        corruptions = ["gaussian_noise"] if args.smoke else cfg["corruptions"]
        seeds = (2020,) if args.smoke else SEEDS
        with torch.no_grad():
            model(torch.zeros(BATCH, 3, 224, 224, device=device))
        sync(device)
        for seed in seeds:
            for corruption in corruptions:
                key = f"{corruption}-{seed}"
                dest = args.output / "raw" / args.dataset / key / f"{args.method}.json"
                if dest.exists() and json.loads(dest.read_text())["status"] == "ok":
                    continue
                data, ids = load_data(spec, corruption, 5, transform)
                adapt, validation = split(data.targets, seed)
                record = dict(adapt=adapt.tolist(), validation=validation.tolist(), dataset_ids_sha256=data_digest(ids),
                              adapt_sha256=split_hash(adapt), validation_sha256=split_hash(validation))
                split_file = args.output / "splits" / args.dataset / f"{key}.json"
                if split_file.exists() and json.loads(split_file.read_text()) != record:
                    raise ValueError(f"Split mismatch for {key}")
                save(split_file, record)
                seed_everything(seed)
                sync(device)
                start = perf_counter()
                adapter = (FrozenPEA(model, args.dataset, spec, transform, seed, args.upstream, cfg_pea, args.smoke)
                           if args.method == "pea" else PLEAD(model, FULL))
                sync(device)
                setup_seconds = perf_counter() - start
                validation_indices = validation[:128] if args.smoke else validation
                loaders = [DataLoader(Subset(data, indices.tolist()), batch_size=BATCH, pin_memory=device.type=="cuda")
                           for indices in (adapt, validation_indices)]
                title = f"{args.dataset} {args.method} {key}"
                a = run_phase(loaders[0], adapter.update, adapter, device, spec["classes"], title+" adapt")
                if args.method == "pea":
                    adapter.freeze()
                if args.smoke:
                    state = ({k: t.clone() for k, t in adapter.stats.mu.items()},
                             {k: t.clone() for k, t in adapter.stats.cov.items()},
                             dict(adapter.aligner.norm_weights), dict(adapter.stats.drift_smooth),
                             adapter.stats.entropy_smooth) if args.method == "pea" else adapter.corrector.q.clone()
                v = run_phase(loaders[1], adapter.predict, adapter, device, spec["classes"], title+" validation")
                if args.smoke:
                    current = (adapter.stats.mu, adapter.stats.cov, adapter.aligner.norm_weights,
                               adapter.stats.drift_smooth, adapter.stats.entropy_smooth
                               ) if args.method == "pea" else adapter.corrector.q
                    torch.testing.assert_close(current, state, rtol=0, atol=0)
                row = dict(status="ok", dataset=args.dataset, method=args.method, seed=seed, corruption=corruption,
                           checkpoint_sha256=audit["sha256"], adapt_sha256=record["adapt_sha256"],
                           validation_sha256=split_hash(validation_indices), setup_seconds=setup_seconds,
                           adaptation=a, validation=v)
                save(dest, row)
                print(f"{title}: adaptation={a['top1']:.2f}% validation={v['top1']:.2f}%", flush=True)
                del adapter
        save(status, dict(status="complete"))
    except Exception as error:
        save(status, dict(status="failed", error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
