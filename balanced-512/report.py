import argparse
import csv
import json
from pathlib import Path
import statistics


DATASETS = ("cifar10-c", "cifar100-c", "imagenet-c")
METHODS = ("pea", "plead")
SEEDS = (1234, 2020, 9999)
CORRUPTIONS = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
               "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
               "elastic_transform", "pixelate", "jpeg_compression")


def cell(values):
    if not values:
        return "—"
    if len(values) != len(SEEDS):
        return f"{statistics.mean(values):.2f} ({len(values)}/3 seeds)"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    root = args.output / "full"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(f.read_text()) for f in sorted((root / "raw").glob("*/*/*.json"))]
    fields = ("checkpoint_sha256", "adapt_sha256", "validation_sha256")
    lookup = {(r["dataset"], r["corruption"], r["seed"], r["method"]): r for r in rows if r["status"] == "ok"}
    lines = ["# 6.1 类别均衡场景", "",
             "ViT-B/16 · Severity 5 · 15 种扰动 · Seeds 1234、2020、9999 · Batch size 64。",
             "每条流按类别分层、无放回地抽取 512 张，验证使用其余图片。两种方法共享初始权重、样本及顺序。",
             "ImageNet-C 的 512 张适应图来自 512 个不同类别。验证保留全部类别。",
             "主表为冻结状态后的验证 Top-1 (%)。逐扰动对种子求 Mean ± Sample Std。总体先对每个种子的 15 种扰动平均，再跨种子汇总。",
             "PEA 使用官方 ViT 对齐、增强及熵触发重置。冻结验证使用最后的有效统计和对齐权重，跳过基于验证数据估计权重的前向。"
             "源统计离线计算，使用相同 checkpoint/预处理。", "",
             "资源表中的 ms/batch 包括数据到设备和模型推理或适应，不含 DataLoader、离线源统计和评分。"
             "RSS 是单进程采样峰值，GPU 是 PyTorch 已分配显存峰值，均为 MiB。"
             "表内时延按批数加权，内存显示已完成各流中的最大峰值。", ""]
    for data in DATASETS:
        lines += [f"## {data}", "", "| Corruption | PEA | PLEAD |", "| --- | ---: | ---: |"]
        for corruption in CORRUPTIONS:
            for seed in SEEDS:
                pair = [lookup.get((data, corruption, seed, m)) for m in METHODS]
                if all(pair) and any(pair[0][k] != pair[1][k] for k in fields):
                    raise ValueError(f"Unpaired data or checkpoint: {data}/{corruption}/{seed}")
            scores = [cell([lookup[data, corruption, s, m]["validation"]["top1"] for s in SEEDS
                            if (data, corruption, s, m) in lookup]) for m in METHODS]
            lines.append(f"| {corruption} | {scores[0]} | {scores[1]} |")
        totals = []
        for m in METHODS:
            seeds = [statistics.mean(lookup[data, c, s, m]["validation"]["top1"] for c in CORRUPTIONS)
                     for s in SEEDS if all((data, c, s, m) in lookup for c in CORRUPTIONS)]
            totals.append(cell(seeds))
        lines += [f"| **平均准确率** | **{totals[0]}** | **{totals[1]}** |", "",
                  "| 方法 | 阶段 | 成功流 | 时延 (ms/batch) | 峰值 RAM (MiB) | 峰值 VRAM (MiB) |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for m in METHODS:
            subset = [r for k, r in lookup.items() if k[0] == data and k[-1] == m]
            for stage in ("adaptation", "validation"):
                values = [r[stage] for r in subset]
                if values:
                    ms = sum(v["batch_ms"] * v["batches"] for v in values) / sum(v["batches"] for v in values)
                    ram = max(v["peak_rss_mib"] for v in values)
                    gpu = [v["peak_gpu_mib"] for v in values if v["peak_gpu_mib"] is not None]
                    display = [f"{ms:.2f}", f"{ram:.1f}", f"{max(gpu):.1f}" if gpu else "—"]
                else:
                    display = ["—"] * 3
                lines.append(f"| {m.upper()} | {stage} | {len(subset)}/45 | " + " | ".join(display) + " |")
        lines.append("")
    lines += ["## 运行状态", ""]
    for f in sorted((root / "status").glob("*/*.json")):
        state = json.loads(f.read_text())
        lines.append(f"- {f.parent.name}/{f.stem} — {state['status']} {state.get('error', '')}")
    lines += ["", "逐流结果见 `per_stream.csv`。Smoke Test 结果单独保存在 `smoke/`。"]
    tmp = args.output / "report.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(args.output / "report.md")
    flat = []
    for r in lookup.values():
        row = {k: v for k, v in r.items() if k not in ("adaptation", "validation")}
        for stage in ("adaptation", "validation"):
            row.update({f"{stage}_{k}": v for k, v in r[stage].items()})
        flat.append(row)
    if flat:
        with (args.output / "per_stream.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
