"""Validate saved records and build the PLEAD ablation report."""

from collections import Counter, defaultdict
import csv
import fcntl
import gzip
import hashlib
import json
from math import isfinite, sqrt
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import t

from dataclasses import asdict

from cap_experiments.ablation_protocol import (
    COMPONENTS, IR_LEVELS, RIDGE_SWEEP, VALIDATION_PER_CLASS, VARIANTS, WINDOW_SWEEP, experiments,
)
from cap_experiments.online_ir import SPECS
from run_cap_neo import SEEDS, all_corruptions, write_csv, write_json


NAMES = {
    "plead": "Full PLEAD", "plead_no_debias": "No Debiasing", "plead_no_weight": "Uniform Weight",
    "plead_no_gate": "No Gate", "plead_no_shrink": "No Shrinkage", "plead_no_ridge": "Unregularized",
    "hard_prior": "Hard Pseudo-label", "window_32": "Window 32", "window_64": "Window 64",
    "window_256": "Window 256", "ridge_0.001": "Ridge 0.001", "ridge_0.1": "Ridge 0.1",
}
DATASETS = {"cifar10-c": "CIFAR-10-C", "cifar100-c": "CIFAR-100-C", "imagenet-c": "ImageNet-C"}
METRICS = ("top1", "ece", "fallback_rate", "gamma_zero_rate", "class_prior_tv_error",
           "effective_debiasing_rate", "kappa_mean", "kappa_update_std", "gamma_mean", "lambda_mean",
           "adaptation_top1", "adapter_seconds", "total_seconds", "throughput", "peak_gpu_mib")


def table(headers, rows):
    def line(row):
        return "| " + " | ".join(map(str, row)) + " |"
    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def stats(values, field):
    return {f"{field}_mean": mean(values), f"{field}_std": stdev(values) if len(values) >= 2 else None}


def seed_means(records, key):
    buckets = defaultdict(list)
    for row in records:
        buckets[row["seed"]].append(row[key])
    return [mean(values) for values in buckets.values()]


def verify(record, smoke):
    identity = record["identity"]
    dataset, ir = identity["dataset"], identity["IR"]
    adaptation = record["splits"]["adaptation"]
    flat = [value for subset in adaptation for value in subset]
    holdout = record["splits"].get("balanced_validation", [])
    digest = hashlib.sha256(json.dumps(record["splits"], separators=(",", ":")).encode()).hexdigest()
    if digest != record["split_sha256"] or len(flat) != len(set(flat)) or set(flat).intersection(holdout):
        raise ValueError(f"Invalid or overlapping sampling indices for {identity}")
    classes, size, online_batch, _ = SPECS[dataset]
    limit = 1 if ir is None else (2 if smoke else classes)
    expected_size = 512 if ir is None else size
    batch_size = 64 if ir is None else online_batch
    if len(adaptation) != limit or any(len(s) != expected_size for s in adaptation):
        raise ValueError(f"Invalid subset lengths for {identity}")
    if len(holdout) != len(set(holdout)):
        raise ValueError(f"Validation repeats for {identity}")
    groups = {key: list(value) for key, value in experiments(dataset, ir).items()}
    methods = {m for names in groups.values() for m in names}
    if record["experiments"] != groups or {row["method"] for row in record["rows"]} != methods:
        raise ValueError(f"Missing method coverage for {identity}")
    if len(record["rows"]) != len(methods):
        raise ValueError(f"Duplicate method row for {identity}")
    expected_batches = limit * ((expected_size + batch_size - 1) // batch_size)
    updates_by_method = defaultdict(list)
    for row in record["updates"]:
        updates_by_method[row["method"]].append(row)
    for row in record["rows"]:
        if (row["split_sha256"] != digest or row["n_samples"] != (len(holdout) or len(flat))
                or len(updates_by_method[row["method"]]) != expected_batches):
            raise ValueError(f"Incomplete or unpaired outputs for {identity}")
        arrivals = 0
        for idx, update in enumerate(updates_by_method[row["method"]]):
            bps = (expected_size + batch_size - 1) // batch_size
            expected_batch = min(batch_size, expected_size - (idx % bps) * batch_size)
            arrivals += expected_batch
            if (update["update_idx"] != idx or update["subset_id"] != idx // bps
                    or update["batch_size"] != expected_batch
                    or update["bank_size"] != min(row["window"], arrivals)):
                raise ValueError(f"FIFO or batch-boundary verification failed for {identity}")
        for field in ("top1", "top5", "ece", "fallback_rate", "adapter_seconds", "total_seconds"):
            if not isfinite(row[field]):
                raise ValueError(f"Nonfinite {field} for {identity}")
    return identity


def _fmt(row, field, scale=1, digits=3):
    avg, sd = row.get(f"{field}_mean"), row.get(f"{field}_std")
    if avg is None:
        return "—"
    return f"{avg*scale:.{digits}f}" + (f" ± {sd*scale:.{digits}f}" if sd is not None else "")


def _ci(row):
    if row is None or row.get("ci95_low") is None:
        return "—"
    return f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"


def _write_if_rows(path, rows):
    if rows:
        write_csv(path, rows)
    else:
        path.write_text("status\n未完成\n")


def _plot(output, name, rows, part):
    fig = Figure(figsize=(9, 4.4), constrained_layout=True)
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    if not rows:
        axis.text(.5, .5, "No measured data", transform=axis.transAxes, ha="center")
    elif part == "sensitivity":
        found = {row["method"]: row for row in rows}
        fig.clear()
        axes = fig.subplots(1, 2)
        for axis, order, xs, xlabel in zip(
            axes, (WINDOW_SWEEP, RIDGE_SWEEP), ((32, 64, 128, 256), (0, .001, .01, .1)), ("FIFO window", "Ridge ratio"),
        ):
            available = [(x, found[key]) for x, key in zip(xs, order) if key in found]
            axis.errorbar([x for x, _ in available], [row["top1_mean"] for _, row in available],
                          yerr=[row["top1_std"] or 0 for _, row in available], marker="o", capsize=3)
            axis.set(xlabel=xlabel, ylabel="Online Top-1 (%)")
            axis.grid(axis="y", alpha=.2)
    elif part == "mechanism":
        for dataset in DATASETS:
            subset = [row for row in rows if row["dataset"] == dataset]
            if subset:
                axis.scatter([row["class_prior_tv_error"] for row in subset], [row["top1"] for row in subset],
                             s=6, alpha=.2, label=DATASETS[dataset], rasterized=True)
        axis.set(xlabel="Class-prior TV error", ylabel="Online batch Top-1 (%)")
        axis.legend(frameon=False)
    else:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["dataset"], row["IR"]].append(row)
        x = np.arange(len(COMPONENTS))
        for (ds, ir), group in grouped.items():
            present = {row["method"]: row for row in group}
            if "plead" not in present:
                continue
            reference = present["plead"]["top1_mean"]
            positions = [i for i, method in enumerate(COMPONENTS) if method in present]
            axis.plot(positions, [present[tuple(COMPONENTS)[i]]["top1_mean"] - reference for i in positions],
                      marker="o", label=DATASETS[ds] + (f" IR={ir}" if ir is not None else ""))
        axis.axhline(0, color="#555555", linewidth=.8)
        axis.set_xticks(x, [NAMES[m] for m in COMPONENTS], rotation=15)
        axis.set_ylabel("Top-1 change vs Full PLEAD (pp)")
        axis.legend(frameon=False, ncol=2 if part == "online_ir" else 1, fontsize=8)
    axis.grid(axis="y", alpha=.2)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=200)


def _markdown(output, summary, pairs, coverage, expected, mechanism, efficiency, failures, smoke):
    def section(title, rows):
        if not rows:
            return [f"## {title}", "", "未完成。", ""]
        paired = {(r["experiment"], r["dataset"], r["IR"], r["method"]): r for r in pairs}
        content = []
        for r in rows:
            pair = paired.get((r["experiment"], r["dataset"], r["IR"], r["method"]))
            delta = f"{pair['delta_mean']:+.3f}" if pair else "—"
            content.append([DATASETS[r["dataset"]], r["IR"] if r["IR"] is not None else "—", NAMES[r["method"]],
                            f"{r['n_streams']}/{r['expected_streams']}", _fmt(r, "top1"), _fmt(r, "ece", digits=4),
                            _fmt(r, "fallback_rate", 100, 2), _fmt(r, "gamma_zero_rate", 100, 2),
                            _fmt(r, "class_prior_tv_error", digits=4), _fmt(r, "kappa_mean"),
                            _fmt(r, "effective_debiasing_rate", 100, 2),
                            _fmt(r, "adapter_seconds"), delta, _ci(pair)])
        return [f"## {title}", "", *table(
            ("数据集", "IR", "方法", "流", "Top-1 (%)", "ECE", "Fallback (%)", "γ=0 (%)",
             "先验 TV", "κ（流均值）", "有效去偏 (%)", "适应耗时 (s)", "相对 Full (pp)", "配对 95% CI"),
            content,
        ), ""]

    lines = ["# PLEAD Ablation Experiment Report", "", "## 1. Overview", "",
             "冻结 ViT-Base 和分类头。标准污染用 512 张无标签图适应，主指标为独立 Balanced Validation Top-1。",
             "Online IR 在同一批更新后评分，子流切换时延续状态。默认 FIFO 为 128。所有对比共享样本顺序。",
             "统计以 seed 为单位，先在 seed 内平均同一组扰动。只有全量覆盖的三 seed 对比给出 Student t 95% CI（df=2）。",
             ""]
    config_dir = output.parent / "configs"
    lines += table(("数据集", "Checkpoint 配置", "Online IR"), [
        [DATASETS[dataset], f"[固定权重与预处理](../configs/{dataset}.json)"
         if (config_dir / f"{dataset}.json").is_file() else "未完成", ", ".join(map(str, levels))]
        for dataset, levels in IR_LEVELS.items()
    ])
    lines += ["", "## 2. Completion", "",
             f"运行范围为 {'smoke' if smoke else '全量协议'}。实际保存 {len(coverage)} / {len(expected)} 条配对流。", ""]
    lines += table(("数据集", "标准污染", "Online IR", "未完成流"), [
        [name, f"{sum(d == ds and ir is None for d, _, _, ir in coverage)} / {sum(d == ds and ir is None for d, _, _, ir in expected)}",
         f"{sum(d == ds and ir is not None for d, _, _, ir in coverage)} / {sum(d == ds and ir is not None for d, _, _, ir in expected)}",
         sum(d == ds for d, _, _, _ in expected - coverage)] for ds, name in DATASETS.items()])
    lines += ["", *([f"- {item}" for item in failures] if failures else ["当前日志中没有已记录的异常终止。"]), ""]
    for title, name in (("3. Standard Corruption Ablation", "standard"), ("4. Online IR Ablation", "online_ir"),
                        ("5. Estimation Strategy", "estimation"), ("6. Sensitivity Analysis", "sensitivity")):
        lines += section(title, [r for r in summary if r["experiment"] == name])
    lines += ["## 7. Mechanism Analysis", "",
              "Oracle Logit Shift Error 已跳过。本次配置没有经核验的干净图像配对清单。", ""]
    if mechanism:
        groups = defaultdict(list)
        for row in mechanism:
            groups[row["dataset"], row["IR"]].append(row)
        lines += table(("数据集", "IR", "更新数", "平均先验 TV", "平均类别校正范数", "γ=0 (%)",
                        "切换后首批 − 切换前末批（pp）", "先验 TV 与逐批 Top-1 的相关系数"),
                       [[DATASETS[d], ir, len(rs), f"{mean(r['class_prior_tv_error'] for r in rs):.4f}",
                         f"{mean(r['class_composition_correction_norm'] for r in rs):.4f}",
                         f"{100*mean(r['gamma'] == 0 for r in rs):.2f}",
                         f"{mean(r['boundary_delta_pp'] for r in rs if r['boundary_delta_pp'] is not None):+.3f}"
                         if any(r["boundary_delta_pp"] is not None for r in rs) else "—",
                         _correlation(rs, "class_prior_tv_error", "top1")]
                        for (d, ir), rs in groups.items()])
        lines += ["", "类别切换前后可能由不同类别组成，差值描述观测到的变化。逐更新 κ、γ、λ 及原因见 mechanism.csv。", ""]
    else:
        lines += ["未完成。", ""]
    lines += ["## 8. Efficiency", "",
              "GPU 先预热，再用 CUDA 事件并在结束时同步。总推理耗时含输入传输、实际主干前向、缓存读取和适应。",
              "CPU 图片解码和指标汇总在计时范围外。峰值 GPU 内存含共享特征缓存和同驻留的消融方法。", ""]
    timing = defaultdict(list)
    for item in efficiency:
        if item["IR"] is not None:
            timing[item["dataset"], item["cache_scope"]].append(item)
    if timing:
        lines += table(("数据集", "缓存状态", "流数", "适应秒/流", "推理秒/流", "图像/秒", "显存 MiB", "前向次数/流"),
                       [[DATASETS[d], scope, len(rs), *(f"{mean(row[k] for row in rs):.3f}" for k in
                         ("adapter_seconds", "total_seconds", "throughput")),
                         f"{mean(r['peak_gpu_mib'] for r in rs if r['peak_gpu_mib'] is not None):.1f}"
                         if any(r["peak_gpu_mib"] is not None for r in rs) else "—",
                         f"{mean(r['backbone_forward_count'] for r in rs):.1f}"] for (d, scope), rs in timing.items()])
        lines += ["", "冷缓存与跨配置缓存分别汇报。当前主实验只有 PLEAD 消融，数据未用于推算与不同协议下方法的加速比。", ""]
    else:
        lines += ["未完成。", ""]
    lines += ["## 9. Key Findings", ""]
    for name, caption in (("standard", "标准污染"), ("online_ir", "Online IR"),
                          ("estimation", "估计策略"), ("sensitivity", "敏感性")):
        available = [r for r in pairs if r["experiment"] == name]
        if available:
            lo, hi = min(available, key=lambda r: r["delta_mean"]), max(available, key=lambda r: r["delta_mean"])
            def claim(row):
                return (f"{DATASETS[row['dataset']]}{'' if row['IR'] is None else ' IR=' + str(row['IR'])} "
                        f"{NAMES[row['method']]}（{row['delta_mean']:+.3f} pp）")
            lines.append(f"- {caption}中，相对 Full 的最低和最高配对变化分别为 {claim(lo)}、{claim(hi)}。")
    lines += ["", f"当前配对置信区间齐备的比较有 {sum(row.get('ci95_low') is not None for row in pairs)} 项。"
              "组件取舍需结合提升方向、置信区间和开销，当前表格只覆盖已有结果。", "",
              "## 10. Files", ""]
    for path in sorted((output.parent / "data").glob("*.csv")):
        lines.append(f"- [{path.name}](../data/{path.name})")
    lines += ["- [verification.json](../verification.json)", "- [figures/](figures/)",
              "- 原始逐流清单位于 `../data/raw/`，失败日志位于 `../logs/`。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def _correlation(rows, a, b):
    x, y = [row[a] for row in rows], [row[b] for row in rows]
    if len(x) < 2 or min(np.std(x), np.std(y)) == 0:
        return "—"
    return f"{np.corrcoef(x, y)[0, 1]:+.3f}"


def export(root, smoke):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _export(root, smoke)


def _export(root, smoke):
    raw = root / "data" / "raw"
    data, report = root / "data", root / "report"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    corruptions = ("gaussian_noise",) if smoke else all_corruptions
    seeds = (2020,) if smoke else SEEDS
    write_json(root / "configs" / "experiment_config.json", dict(
        scope="smoke" if smoke else "full", severity=5, seeds=seeds, corruptions=corruptions,
        IR=IR_LEVELS, online_specs=SPECS, online_smoke_subsets=2, standard_adaptation_budget=512,
        validation_per_class=VALIDATION_PER_CLASS, adaptation_semantics="batch_transductive",
        window=128, oracle_logit_shift="skipped: no verified clean/corrupt pairing",
    ))
    write_json(root / "configs" / "method_variants.json",
               {name: asdict(value) for name, value in VARIANTS.items()})
    expected = {(dataset, c, s, ir) for dataset, irs in IR_LEVELS.items() for c in corruptions
                for s in seeds for ir in (None, *irs)}
    coverage = set()
    groups, pair_groups = defaultdict(list), defaultdict(list)
    per_stream, paired, mechanisms, efficiency, subsets = [], [], [], [], []
    counts = Counter()
    with (data / "per_update.csv.tmp").open("w", newline="") as updates_handle:
        writer = None
        for suite in ("standard", "online_ir"):
            for path in sorted((raw / suite).glob("*/*/*/*.json.gz")):
                with gzip.open(path, "rt") as handle:
                    record = json.load(handle)
                identity = verify(record, smoke)
                d, ir, seed, corruption = (identity[k] for k in ("dataset", "IR", "seed", "corruption"))
                event = (d, corruption, seed, ir)
                if event in coverage or event not in expected:
                    raise ValueError(f"Unexpected stream in {path}")
                coverage.add(event)
                by_name = {r["method"]: r for r in record["rows"]}
                per_stream.extend(record["rows"])
                efficiency.append(by_name["plead"])
                for experiment, methods in record["experiments"].items():
                    counts[experiment] += 1
                    for method in methods:
                        row = by_name[method]
                        groups[experiment, d, ir, method].append(row)
                        if method == "plead":
                            continue
                        pair = dict(**identity, experiment=experiment, method=method, split_sha256=row["split_sha256"],
                                    delta=row["top1"] - by_name["plead"]["top1"])
                        pair_groups[experiment, d, ir, method].append(pair)
                        paired.append(pair)
                last = None
                for step in record["updates"]:
                    row = dict(**identity, **step)
                    row["fallback_reason"] = "|".join(row.pop("fallback_reasons"))
                    row["Neff"] = row.pop("neff")
                    row["Sxx"] = row.pop("sxx")
                    row["fit_correlation"] = row.pop("correlation")
                    if writer is None:
                        writer = csv.DictWriter(updates_handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    if step["method"] == "plead" and "mechanism" in record["experiments"]:
                        boundary = last is not None and step["subset_id"] != last["subset_id"]
                        mechanisms.append({**row, "boundary_delta_pp": step["top1"] - last["top1"] if boundary else None})
                        last = step
                audit = record["ir_audit"]
                for subset in record["subsets"]:
                    row = dict(**identity, **subset)
                    if audit:
                        i = subset["subset_id"]
                        row.update(dominant_class=audit["dominant_classes"][i], class_counts=audit["histograms"][i])
                    subsets.append(row)
    if coverage:
        (data / "per_update.csv.tmp").replace(data / "per_update.csv")
    else:
        (data / "per_update.csv.tmp").unlink()
        (data / "per_update.csv").write_text("status\n未完成\n")
    summary, paired_summary = [], []
    required = len(seeds) * len(corruptions)
    for (experiment, d, ir, method), rows in groups.items():
        item = dict(experiment=experiment, dataset=d, IR=ir, method=method, n_streams=len(rows), expected_streams=required)
        for metric in METRICS:
            valid = [row for row in rows if row.get(metric) is not None]
            if valid:
                item.update(stats(seed_means(valid, metric), metric))
        pairs = pair_groups.get((experiment, d, ir, method))
        if pairs:
            values = seed_means(pairs, "delta")
            res = {**{key: item[key] for key in ("experiment", "dataset", "IR", "method", "n_streams")},
                   **stats(values, "delta"), "n_seeds": len(values)}
            if not smoke and len(pairs) == required and len(values) == 3:
                interval = t.ppf(.975, 2) * stdev(values) / sqrt(3)
                res.update(ci95_low=mean(values) - interval, ci95_high=mean(values) + interval)
            paired_summary.append(res)
            item.update(delta_vs_full_mean=res["delta_mean"], delta_vs_full_std=res["delta_std"])
        summary.append(item)
    outputs = {
        "per_stream": per_stream, "summary": summary, "paired_summary": paired_summary,
        "paired_differences": paired, "per_subset": subsets, "mechanism": mechanisms, "efficiency": efficiency,
        "component_ablation": [r for r in summary if r["experiment"] in ("standard", "online_ir")],
        "estimation_ablation": [r for r in summary if r["experiment"] == "estimation"],
        "sensitivity": [r for r in summary if r["experiment"] == "sensitivity"],
    }
    for filename, rows in outputs.items():
        _write_if_rows(data / f"{filename}.csv", rows)
    for filename, part in (("standard_ablation", "standard"), ("online_ir_ablation", "online_ir"),
                           ("sensitivity", "sensitivity"), ("mechanism", "mechanism")):
        rows = mechanisms if part == "mechanism" else [r for r in summary if r["experiment"] == part]
        _plot(figures, filename, rows, part)
    failures = []
    for path in sorted((root / "logs").glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(("ValueError:", "RuntimeError:", "FileNotFoundError:", "AssertionError:")):
                failures.append(f"`logs/{path.name}` · {line}")
    _markdown(report, summary, paired_summary, coverage, expected, mechanisms, efficiency, failures, smoke)
    write_json(root / "verification.json", dict(
        scope="smoke" if smoke else "full", complete=coverage == expected, expected_paired_streams=len(expected),
        completed_paired_streams=len(coverage), method_rows=len(per_stream), experiment_counts=dict(counts),
        missing_streams=[list(k) for k in sorted(expected - coverage, key=str)], failures=failures,
        checked_sample_hashes=True, checked_pairing=True, checked_fifo_and_batch_boundaries=True,
        oracle_logit_shift="skipped: no verified clean/corrupt pairing",
    ))
