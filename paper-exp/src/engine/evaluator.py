from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(method, data, indices, batch_size, device, phase, title):
    loader = DataLoader(Subset(data, indices.tolist()), batch_size=batch_size)
    step = method.step if phase == "adaptation" else method.predict
    if phase == "validation":
        method.model.eval()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    rows = []
    for images, labels in tqdm(loader, unit="batch", desc=title, mininterval=2):
        before = method.forward_passes
        start_batch = perf_counter()
        scores = step(images.to(device))
        sync(device)
        latency = (perf_counter() - start_batch) * 1000.
        if scores.shape != (len(images), method.model.head.out_features) or not torch.isfinite(scores).all():
            raise ValueError(f"Invalid {phase} scores at batch {len(rows)}")
        correct = int((scores.argmax(1).cpu() == labels).sum())
        rows.append(dict(batch=len(rows), samples=len(labels), correct=correct, latency_ms=latency,
                         forwards=method.forward_passes - before,
                         events=method.get_state()["events"] if phase == "adaptation" else {}))
    elapsed = perf_counter() - start
    n = sum(row["samples"] for row in rows)
    return dict(n=n, top1=100 * sum(row["correct"] for row in rows) / n,
                latency_ms=sum(row["latency_ms"] for row in rows) / len(rows),
                forwards_per_batch=sum(row["forwards"] for row in rows) / len(rows),
                throughput=n / elapsed, elapsed_seconds=elapsed,
                peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
                batches=rows)
