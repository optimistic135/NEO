import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling import DESIGN


SEEDS = (1234, 2020, 9999)
CORRUPTIONS = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
               "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
               "elastic_transform", "pixelate", "jpeg_compression")
FIELDS = ("checkpoint_sha256", "dataset_ids_sha256", "evaluated_indices_sha256", "stream_sha256")


def summarize(values):
    return f"{mean(values):.2f} ± {stdev(values):.2f}" if len(values) == 3 else "—"


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("directory", type=Path, help="Result directory for strict or cap policy")
    root = cli.parse_args().directory
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "full/raw").glob("*/*/*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    lookup = {(r["dataset"], r["corruption"], r["seed"], r["ir"], r["method"]): r for r in rows}
    lines = ["# Online IR · PEA / PLEAD", "",
             f"采样策略 `{root.name}`。ViT-B/16，severity 5，15 种扰动，种子 1234、2020、9999。",
             "当前批加入统计后评分。批次上限 64，子集尾批保留。状态连续传递，PEA 仍执行官方熵重置。",
             "CIFAR 全流无重复。ImageNet-C 子集内无重复，允许跨子集重复。IR 为名义比值，实际比例见下表。",
             "只有完整的 15 种扰动 × 3 种子才填主表的 Mean ± Sample Std。增益为同 corruption、同 seed 的配对差，先按种子平均 15 种扰动，再以 df=2 计算 95% CI。",
             "ms/batch 为包含设备传输和适应推理的同步时延，排除数据读取、评分及离线准备。RAM 和 VRAM 是已完成流的最高峰值（MiB）。", ""]
    summary = []
    for dataset, (_, _, ratios) in DESIGN.items():
        method_list = ("plead",) if dataset == "imagenet-c" else ("pea", "plead")
        lines += [f"## {dataset}", "",
                  "| 名义 IR | 目标主导比例 | 实际主导比例 | PEA Top-1 (%) | PLEAD Top-1 (%) | PLEAD−PEA (pp, 95% CI) |",
                  "| ---: | ---: | ---: | --- | --- | --- |"]
        resources = []
        for ir in ratios:
            scores = {}
            fractions = []
            for method in method_list:
                by_key = {(c, s): lookup[dataset, c, s, ir, method] for c in CORRUPTIONS for s in SEEDS
                          if (dataset, c, s, ir, method) in lookup}
                successful = [r for r in by_key.values() if r["status"] == "ok"]
                counts = Counter(r["status"] for r in by_key.values())
                fractions.extend(r["realized_dominant_fraction"] for r in successful)
                complete = counts["ok"] == len(CORRUPTIONS) * len(SEEDS)
                scores[method] = [mean(by_key[c, s]["top1"] for c in CORRUPTIONS) for s in SEEDS] if complete else []
                avg_ms = (sum(r["batch_ms"] * r["batches"] for r in successful) / sum(r["batches"] for r in successful)
                          if successful else None)
                ram = max((r["peak_rss_mib"] for r in successful), default=None)
                gpu = max((r["peak_gpu_mib"] for r in successful if r["peak_gpu_mib"] is not None), default=None)
                resources.append((ir, method, counts, avg_ms, ram, gpu))
                summary.append(dict(dataset=dataset, ir=ir, method=method, n_ok=counts["ok"],
                                    n_failed=counts["failed"], n_capacity_blocked=counts["capacity_blocked"],
                                    n_smoke_blocked=counts["smoke_blocked"],
                                    top1_mean=mean(scores[method]) if complete else None,
                                    top1_std=stdev(scores[method]) if complete else None,
                                    batch_ms=avg_ms, peak_ram_mib=ram, peak_vram_mib=gpu))
            if len(method_list) == 2:
                for c in CORRUPTIONS:
                    for s in SEEDS:
                        a, b = (lookup.get((dataset, c, s, ir, m)) for m in method_list)
                        if a and b and a["status"] == b["status"] == "ok":
                            if any(a[k] != b[k] for k in FIELDS):
                                raise ValueError(f"Unpaired streams: {dataset}/{ir}/{c}/{s}")
            gain = "—"
            if scores.get("pea") and scores["plead"]:
                paired = [b - a for a, b in zip(scores["pea"], scores["plead"])]
                m = mean(paired)
                radius = 4.30265273 * stdev(paired) / 3**.5
                gain = f"{m:+.2f} [{m-radius:+.2f}, {m+radius:+.2f}]"
            target = f"{ir / (ir + DESIGN[dataset][0] - 1):.2%}"
            actual = f"{mean(fractions):.2%}" if fractions else "—"
            pea = summarize(scores["pea"]) if "pea" in scores else "未安排"
            lines.append(f"| {ir} | {target} | {actual} | {pea} | {summarize(scores['plead'])} | {gain} |")
        lines += ["", "| IR | 方法 | 成功 / 计划 | 容量阻塞 | 运行/冒烟失败 | ms/batch | 峰值 RAM | 峰值 VRAM |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for ir, method, counts, *values in resources:
            metrics = " | ".join("—" if v is None else f"{v:.2f}" for v in values)
            lines.append(f"| {ir} | {method.upper()} | {counts['ok']}/45 | {counts['capacity_blocked']} | "
                         f"{counts['failed']+counts['smoke_blocked']} | {metrics} |")
        lines.append("")
    problems = sorted({(r["dataset"], r["ir"], r["method"], r["error"]) for r in rows if "error" in r})
    if problems:
        lines += ["## 未完成项", ""] + [f"- {d}, IR={ir}, {m.upper()} · {e}" for d, ir, m, e in problems]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    for filename, records in (("per_stream.csv", rows), ("summary.csv", summary)):
        keys = sorted({key for r in records for key in r})
        if keys:
            with (root / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)


if __name__ == "__main__":
    main()
