import csv
from itertools import combinations
import json
from statistics import mean, stdev


def csv_save(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def plots(folder, summary, efficiency, per, corruptions, methods):
    if not efficiency:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    if summary:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.bar([x["method"] for x in summary], [x["validation_mean"] for x in summary],
               yerr=[x["validation_std"] for x in summary], capsize=3)
        ax.set_ylabel("Validation Top-1 (%)")
        fig.savefig(folder / "overall_accuracy.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    for method in methods:
        values = [per.get((method, c), float("nan")) for c in corruptions]
        if any(x == x for x in values):
            ax.plot(corruptions, values, marker="o", label=method)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Validation Top-1 (%)")
    if ax.lines:
        ax.legend()
    fig.savefig(folder / "per_corruption_accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    rows = [r for r in efficiency if r["phase"] == "adaptation"]
    ax.bar([x["method"] for x in rows], [x["latency_ms"] for x in rows])
    ax.set_ylabel("Adaptation latency (ms/batch)")
    fig.savefig(folder / "efficiency.pdf")
    plt.close(fig)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows]


def summarize(root):
    cfg = json.loads((root / "configuration.json").read_text())
    protocol, audit = cfg["protocol"], cfg["checkpoint"]
    corruptions, seeds, methods = protocol["corruptions"], protocol["seeds"], list(cfg["methods"])
    expected = len(corruptions) * len(seeds)
    phases = ("adaptation", "validation")
    raw = [json.loads(p.read_text()) for p in sorted((root / "raw").glob("*/*.json"))]
    ok = [r for r in raw if r["status"] == "ok"]
    cache = {}
    detail, summary, efficiency, pairs = [], [], [], []
    batches = {p: [] for p in phases}
    for r in ok:
        name, corruption, seed = (r[k] for k in ("method", "corruption", "seed"))
        split = json.loads((root / "splits" / f"{corruption}-{seed}.json").read_text())
        if r["checkpoint_sha256"] != audit["sha256"] or any(r[p+"_sha256"] != split[p+"_sha256"] for p in phases):
            raise ValueError(f"Pairing mismatch: {name}/{corruption}/{seed}")
        row = dict(method=name, corruption=corruption, seed=seed)
        for phase in phases:
            stats = r[phase]
            n, correct = (sum(b[key] for b in stats["batches"]) for key in ("samples", "correct"))
            if n != stats["n"] or abs(100. * correct / n - stats["top1"]) > 1e-8:
                raise ValueError(f"Batch scores disagree with stream score: {name}/{corruption}/{seed}")
            row[phase] = stats["top1"]
            for b in stats["batches"]:
                batches[phase].append(dict(method=name, corruption=corruption, seed=seed, **b))
        detail.append(row)
        cache[name, corruption, seed] = r
    seed_scores, per, coverage = {}, {}, {}
    for name in methods:
        rows = [r for r in ok if r["method"] == name]
        coverage[name] = len(rows)
        for c in corruptions:
            if all((name, c, seed) in cache for seed in seeds):
                per[name, c] = mean(cache[name, c, seed]["validation"]["top1"] for seed in seeds)
        if len(rows) == expected and all((name,c,seed) in cache for c in corruptions for seed in seeds):
            total = dict(method=name, streams=len(rows))
            for phase in phases:
                values = [mean(cache[name,c,s][phase]["top1"] for c in corruptions) for s in seeds]
                seed_scores[name,phase] = values
                total[phase+"_mean"], total[phase+"_std"] = mean(values), stdev(values)
            summary.append(total)
        for phase in phases:
            if not rows:
                continue
            stats = [r[phase] for r in rows]
            count = sum(len(s["batches"]) for s in stats)
            peak = [s["peak_gpu_mib"] for s in stats if s["peak_gpu_mib"] is not None]
            efficiency.append(dict(method=name, phase=phase, streams=len(rows),
                                   latency_ms=sum(s["latency_ms"]*len(s["batches"]) for s in stats)/count,
                                   throughput=sum(s["n"] for s in stats)/sum(s["elapsed_seconds"] for s in stats),
                                   peak_gpu_mib=max(peak) if peak else None,
                                   forwards_per_batch=sum(s["forwards_per_batch"]*len(s["batches"]) for s in stats)/count,
                                   trainable_parameters=rows[0]["state"]["trainable_parameters"],
                                   mean_model_load_s=mean(r["model_load_seconds"] for r in rows),
                                   mean_setup_s=mean(r["setup_seconds"] for r in rows)))
    for first, second in combinations(methods, 2):
        for phase in phases:
            a, b = seed_scores.get((first,phase)), seed_scores.get((second,phase))
            if a and b:
                diffs = [x-y for x,y in zip(a,b)]
                avg, radius = mean(diffs), 4.30265273*stdev(diffs)/len(diffs)**.5
                pairs.append(dict(first=first, second=second, phase=phase, delta_pp=avg,
                                  ci_low=avg-radius, ci_high=avg+radius))

    for phase in phases:
        csv_save(root/"raw"/f"{phase}_batches.csv", batches[phase],
                 ("method", "corruption", "seed", "batch", "samples", "correct", "latency_ms", "forwards", "events"))
    csv_save(root/"per_corruption"/"metrics.csv", detail, ("method", "corruption", "seed", *phases))
    csv_save(root/"summary"/"overall_metrics.csv", summary,
             ("method", "streams", "adaptation_mean", "adaptation_std", "validation_mean", "validation_std"))
    csv_save(root/"summary"/"paired_comparisons.csv", pairs, ("first", "second", "phase", "delta_pp", "ci_low", "ci_high"))
    efficiency_columns = ("method", "phase", "streams", "latency_ms", "throughput", "peak_gpu_mib",
                          "forwards_per_batch", "trainable_parameters", "mean_model_load_s", "mean_setup_s")
    csv_save(root/"summary"/"efficiency.csv", efficiency, efficiency_columns)
    plots(root/"figures", summary, efficiency, per, corruptions, methods)

    lines = ["# ViT-B/16 · CIFAR-100-C", "", "## Experimental Setup", "",
             "与 balanced-512 共享分层划分。每流 512 张适应，随后在剩余 9,488 张上验证。"
             f"Severity {protocol['severity']}，batch {protocol['batch_size']}，种子 {seeds}。", "",
             "## Model and Checkpoint", "", f"- Checkpoint {audit['path']}",
             f"- SHA256 {audit['sha256']}", f"- Preprocessing {audit['preprocessing']}", "",
             "## Implementation Details", ""]
    implementation = []
    for method in methods:
        m = cfg["methods"][method]
        up = m.get("upstream")
        ref = f"[Code]({up['url']}/tree/{up['commit']}) · [Paper]({up['paper']})" if up else "Project baseline"
        implementation.append((method, ref, m["description"]))
    lines += table(("Method", "Source", "Implementation"), implementation)
    complete = {r["method"]:r for r in summary}
    def score(name, phase):
        r = complete.get(name)
        return f"{r[phase+'_mean']:.2f} ± {r[phase+'_std']:.2f}" if r else "—"
    lines += ["", "## Main Results", ""] + table(("Method", "Streams", "Adaptation Top-1 (%)", "Validation Top-1 (%)"),
        [(m, f"{coverage[m]}/{expected}", score(m, "adaptation"), score(m, "validation")) for m in methods])
    lines += ["", "## Per-Corruption Results", ""] + table(("Corruption", *methods),
        [(c, *(f"{per[m,c]:.2f}" if (m,c) in per else "—" for m in methods)) for c in corruptions])
    lines += ["", "## Efficiency Analysis", ""] + table(("Method","Phase","ms/batch","Images/s","GPU MiB","Params","Forwards/batch"),
        [(r["method"], r["phase"], f"{r['latency_ms']:.2f}", f"{r['throughput']:.2f}",
          f"{r['peak_gpu_mib']:.1f}" if r["peak_gpu_mib"] is not None else "—", r["trainable_parameters"],
          f"{r['forwards_per_batch']:.2f}") for r in efficiency])
    smoke = [json.loads(p.read_text()) for p in (root/"smoke").glob("*/*.json")]
    sources = sum(r.get("source_seconds",0.) for r in smoke + ok)
    lines += ["", f"离线源统计累计 {sources:.2f} 秒。时延包含数据到设备和适应或预测。吞吐包含加载及评分。",
              "模型加载和初始化见 summary/efficiency.csv。", "",
              "## Statistical Analysis", "",
              "先对每个种子的 15 种 corruption 求平均，再报告三个种子的 mean ± sample std。"
              "配对差按同 corruption 和 seed 计算，95% CI 使用三个种子的配对均值和 Student t（df=2）。"
              "完整配对比较见 summary/paired_comparisons.csv。", "",
              "## Reproducibility Audit", "", f"- GPU {cfg['device']} · CUDA {cfg['cuda']} · Git {cfg['git']}",
              f"- Dependencies {cfg['dependencies']}", "- Checked split hashes, checkpoint and batch totals.", "",
              "## Limitations", "",
              "LAME 依官方方法对每个验证批执行无状态图推理。其余方法用最终状态做冻结预测。"
              "统一 ViT-B/16 和 balanced-512 是本实验的适配设置。"]
    for fp in sorted((root/"logs").glob("*.json")):
        state = json.loads(fp.read_text())
        lines.append(f"- {fp.stem} · {state['status']} {state.get('error','')}")
    (root/"report.md").write_text("\n".join(lines) + "\n")
