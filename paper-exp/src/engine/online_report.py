from collections import Counter
import json
from statistics import mean, stdev

from src.engine.benchmark_report import csv_save, table
from src.utils.settings import config

KEYS = ("dataset", "corruption", "seed", "ir")
HASHES = ("checkpoint_sha256", "dataset_ids_sha256", "stream_sha256", "evaluated_indices_sha256")


def read_rows(path):
    return [json.loads(p.read_text()) for p in sorted((path / "full/raw").glob("*/*/*.json"))]


def complete_scores(rows, corruptions, seeds):
    if any((c, s) not in rows for c in corruptions for s in seeds):
        return None
    return [mean(rows[c, s]["top1"] for c in corruptions) for s in seeds]


def matches(left, right):
    return all(left.get(key) is not None and left[key] == right.get(key) for key in HASHES)


def summarize(output, cfg, reference):
    from src.engine.online_benchmark import DESIGN

    own = read_rows(output)
    candidates = read_rows(reference / output.name)
    known = {}
    for r in own:
        if r["status"] == "ok":
            key = tuple(r[k] for k in KEYS)
            previous = known.get(key)
            if previous and not matches(previous, r):
                raise ValueError(f"Baseline provenance mismatch for {key}")
            known[key] = r
    admitted, mismatch = [], 0
    for r in candidates:
        if r["method"] not in ("pea", "plead") or r["status"] != "ok":
            continue
        other = known.get(tuple(r[k] for k in KEYS))
        if other and matches(other, r):
            admitted.append(r)
        elif other:
            mismatch += 1
    combined = own + admitted
    out = [{k: v for k, v in r.items() if k != "batch_details"} for r in combined]
    fields = sorted({k for r in out for k in r})
    csv_save(output / "per_stream.csv", out, fields or (*KEYS, "method", "status"))
    corruptions = config("datasets/corruptions.yaml")["corruptions"]
    methods = cfg["methods"] + ["pea", "plead"]
    expected = len(corruptions) * len(cfg["seeds"])
    lines = ["# Online IR · TTA baselines", "",
             "ViT-B/16，severity 5，15 种扰动，三个共享种子。使用 PEA／PLEAD 的在线 IR 采样，64 张批次上限与子集尾批。",
             "适应状态跨子集传递，各方法保留官方预测时序。表中只展示完整 45 流的 mean ± sample std。",
             "完整 EATA 使用独立干净数据估计 Fisher。NEO 使用官方累计重心。ImageNet-C 容量不足的档位单独计数。", "",
             f"采样策略 {output.name}。已匹配历史 PEA／PLEAD 逐流结果 {len(admitted)} 条，排除哈希不匹配的结果 {mismatch} 条。", ""]
    combined_index = {}
    for r in combined:
        combined_index.setdefault((r["dataset"], r["ir"], r["method"]), []).append(r)
    summary, paired = [], []
    for dataset, (_, _, irs) in DESIGN.items():
        lines += [f"## {dataset}", ""]
        main, coverage = [], []
        for ir in irs:
            row = [ir]
            by_method = {}
            for m in methods:
                group = combined_index.get((dataset, ir, m), [])
                success = {(r["corruption"], r["seed"]): r for r in group if r["status"] == "ok"}
                s = complete_scores(success, corruptions, cfg["seeds"])
                by_method[m] = s
                reasons = Counter(r["status"] for r in group)
                timings = [(r["batch_ms"], r["batches"]) for r in success.values()]
                time = sum(ms*n for ms,n in timings)/sum(n for _,n in timings) if timings else None
                mem = max((r["peak_gpu_mib"] for r in success.values() if r["peak_gpu_mib"] is not None), default=None)
                info = dict(dataset=dataset, ir=ir, method=m, streams=len(success), expected=expected,
                            source_blocked=reasons["source_blocked"], capacity_blocked=reasons["capacity_blocked"],
                            failed=reasons["failed"] + reasons["smoke_blocked"],
                            top1_mean=mean(s) if s else None, top1_std=stdev(s) if s else None,
                            ms_per_batch=time, peak_gpu_mib=mem)
                summary.append(info)
                row.append(f"{mean(s):.2f} ± {stdev(s):.2f}" if s else "—")
                coverage.append((ir, m, f"{info['streams']}/{expected}", info["source_blocked"],
                                 info["capacity_blocked"], info["failed"],
                                 f"{time:.2f}" if time is not None else "—",
                                 f"{mem:.1f}" if mem is not None else "—"))
            main.append(row)
            for a in cfg["methods"]:
                for b in ("pea", "plead"):
                    lhs, rhs = by_method[a], by_method[b]
                    if lhs is None or rhs is None:
                        continue
                    delta = [v-w for v,w in zip(lhs, rhs)]
                    avg = mean(delta)
                    radius = 4.30265273 * stdev(delta) / len(delta)**.5
                    paired.append(dict(dataset=dataset, ir=ir, baseline=a, reference=b, delta_pp=avg,
                                       ci_low=avg-radius, ci_high=avg+radius))
        lines += table(("IR", *methods), main)
        lines += [""] + table(("IR","Method","Streams","Source blocked","Capacity blocked","Failed","ms/batch","VRAM MiB"), coverage) + [""]
    lines += ["## 实现与版本", ""]
    for method in cfg["methods"]:
        method_cfg = config(f"methods/{method}.yaml")
        up = method_cfg.get("upstream")
        source = f" [官方代码]({up['url']}/tree/{up['commit']})" if up else ""
        lines.append(f"- {method_cfg['name']} · {method_cfg['description']}{source}")
    lines += ["", "配对增益以 baseline 减参考方法计，先按种子平均 corruption，再计算 df=2 的 95% t 区间。"
              "逐批准确率、时延与 forward 数保存在成功流的 JSON；CSV 含逐流状态和性能。"]
    errors = sorted({(r["dataset"],r["method"],r["ir"],r["error"]) for r in own if "error" in r})
    lines += ["", "## 未完成项", ""] + [f"- {d}/{m}/IR={ir} · {e}" for d,m,ir,e in errors]
    for path in sorted((output/"logs").glob("*.json")):
        lines.append(f"- {path.stem} · {json.loads(path.read_text())['error']}")
    (output/"report.md").write_text("\n".join(lines)+"\n")
    csv_save(output/"summary.csv", summary, tuple(summary[0]))
    csv_save(output/"paired_differences.csv", paired, ("dataset","ir","baseline","reference","delta_pp","ci_low","ci_high"))
