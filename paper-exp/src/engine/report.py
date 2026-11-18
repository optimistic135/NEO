from collections import Counter, defaultdict
import csv
import json
from math import sqrt
from statistics import mean, stdev

from matplotlib.figure import Figure
from scipy.stats import t

from src.utils.settings import config, save


NAMES = dict(no_adapt="No Adapt", bn_adapt="BNAdapt", tent="TENT", sar="SAR", t3a="T3A", neo="NEO",
             pea="PEA", roid="ROID", plead="PLEAD", hard_prior="Hard Prior", norm_only_kappa="Norm-only Fit",
             no_evidence_gate="w/o Evidence Gate", no_composition="w/o Composition")


def table(columns, rows):
    def line(xs):
        return "| " + " | ".join(str(x).replace("|", "\\|") for x in xs) + " |"
    return "\n".join([line(columns), line(["---"]*len(columns)), *(line(row) for row in rows)])


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as output:
        writer = csv.DictWriter(output, keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def report(output):
    params = config("datasets/corruptions.yaml")
    br = config("protocols/balanced.yaml")
    on = config("protocols/dart_online_ir.yaml")
    official = config("methods/official.yaml")
    audits = {p.stem: json.loads(p.read_text()) for p in sorted((output/"checkpoints").glob("*.json"))}
    save(output/"checkpoint_validation.json", audits)
    rows = [json.loads(p.read_text()) for p in sorted((output/"raw").glob("*/*/*/*/*.json"))]
    raw = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k,v in r.items() if k != "config"} for r in rows]
    csv_save(output/"raw"/"per_stream.csv", raw)
    smoke = [json.loads(p.read_text()) for p in sorted((output/"smoke/raw").glob("*/*/*/*/*.json"))]
    csv_save(output/"smoke"/"per_stream.csv", [{k:v for k,v in r.items() if k not in ("config","traceback")} for r in smoke])
    groups = defaultdict(dict)
    lookup = {}
    for r in rows:
        if r["status"] == "ok":
            key = tuple(r[k] for k in ("stage", "dataset", "architecture", "ir", "method"))
            groups[key][r["seed"], r["corruption"]] = r
            lookup[key + (r["seed"], r["corruption"])] = r
    summary, paired = [], []
    for (stage, d, a, ir, m), samples in groups.items():
        item = dict(stage=stage, dataset=d, architecture=a, ir=ir, method=m, streams=len(samples))
        by_seed = defaultdict(list)
        for row in samples.values():
            by_seed[row["seed"]].append(row)
        n = 1 if stage == "efficiency" else len(params["corruptions"])
        complete = len(by_seed) == 3 and all(len(v) == n for v in by_seed.values())
        item["complete"] = complete
        for metric in ("top1", "macro_accuracy", "ece", "kappa", "gamma", "valid_fit", "fallback_rate",
                       "throughput", "peak_gpu_mib", "forward_passes", "backward_passes", "trainable_parameters",
                       "end_to_end_seconds", "method_seconds"):
            vals = [mean(r[metric] for r in xs) for xs in by_seed.values()
                    if all(r.get(metric) is not None for r in xs)]
            if vals:
                item[metric] = mean(vals)
                if complete:
                    item[metric+"_std"] = stdev(vals)
        if stage == "efficiency":
            baseline = [lookup.get((stage,d,a,ir,"no_adapt",s,c)) for s,c in samples]
            if all(baseline):
                item["adaptation_overhead_seconds"] = mean(r["method_seconds"]-base["method_seconds"]
                                                           for r,base in zip(samples.values(), baseline))
        if m != "plead" and stage != "efficiency":
            deltas = defaultdict(list)
            for (seed, corruption), row in samples.items():
                full = lookup.get((stage,d,a,ir,"plead",seed,corruption))
                if full and full["batch_sha256"] == row["batch_sha256"] and full["checkpoint_sha256"] == row["checkpoint_sha256"]:
                    delta = full["top1"]-row["top1"]
                    deltas[seed].append(delta)
                    paired.append(dict(stage=stage,dataset=d,architecture=a,ir=ir,method=m,seed=seed,
                                       corruption=corruption,plead_gain_pp=delta))
            if len(deltas) == 3 and all(len(x) == n for x in deltas.values()):
                seed_deltas = [mean(x) for x in deltas.values()]
                gain = mean(seed_deltas)
                ci = t.ppf(.975, 2)*stdev(seed_deltas)/sqrt(3)
                item.update(plead_gain_pp=gain, ci95_low=gain-ci, ci95_high=gain+ci)
        summary.append(item)
    csv_save(output/"paired_differences.csv", paired)
    for stage in ("balanced","online_ir","ablation","efficiency"):
        csv_save(output/stage/"summary.csv",[r for r in summary if r["stage"] == stage])
    plans = (("balanced",False,br["methods"]), ("online_ir",True,on["methods"]),
             ("ablation",False,br["ablations"]), ("ablation",True,br["ablations"]),
             ("efficiency",False,br["efficiency"]))
    result_lookup = {(r["stage"],r["dataset"],r["architecture"],r["ir"],r["method"]):r for r in summary}
    text = ["# PLEAD Cross-Architecture TTA Benchmark", "",
            "固定 ViT-Base/16 与 ResNet-50 各自的源 checkpoint。全量 Balanced 使用完整均衡 severity 5，"
            "覆盖 15 种扰动。随机种子 137、2029、8671。CIFAR 批大小 200，ImageNet 批大小 64。",
            "", "结果在每个 seed 内先按 corruption 平均，再汇总三个 seed。完整配对使用 Student t，df=2。"
            "带流数的单点为部分结果。Macro Accuracy 仅平均已出现的类别，ECE 为 15 个等宽区间的百分比。",
            "", "## 权重与预处理", "", table(("配置","状态","类别","SHA256","格式"),
                [(k,v["status"],v.get("classes","—"),v.get("sha256","—"),v.get("format","—")) for k,v in audits.items()]),
            "", "完整路径、文件大小及预处理见 [checkpoint_validation.json](checkpoint_validation.json)。",
            "", "## 方法来源", "", table(("方法","来源","固定提交"),
                [(NAMES[k],v["url"],v["commit"]) for k,v in official["upstream"].items()]),
            "", "TENT、SAR 和 ROID 返回本批梯度更新前的 logits。ROID 在返回前使用本批先验校正。"
            "T3A、NEO、PLEAD 使用加入本批后的状态。BNAdapt 直接使用本批 BN 统计。"
            "PEA 使用官方两遍推理和干净源域统计，HF ViT 通过等价模块视图暴露 transformer block。",
            "", "## Online IR 构造", "", "DART §2、A.1、G.5 指定主导类概率 IR/(IR+K−1)、其他类 1/(IR+K−1)，"
            "以及子集随机排序。论文引用的 SAR 生成器对 ImageNet 独立抽样标签，再按标签有放回抽样图像；"
            "正式运行入口要求可核验的 DART 索引清单。子集内独立分批，保留尾批。",
            "", table(("数据集","子集","每子集","总图数","IR"),[
                (d,params["datasets"][d]["classes"],v["subset"],params["datasets"][d]["classes"]*v["subset"],
                 ", ".join(map(str,v["ratios"]))) for d,v in on["datasets"].items()]), ""]
    expected = Counter()
    for stage, is_online, methods in plans:
        header = ("Online IR" if stage == "online_ir" else "PLEAD ablation (Online IR)" if is_online
                  else {"balanced":"Balanced Corruption","ablation":"PLEAD ablation (Balanced)","efficiency":"Efficiency"}[stage])
        text += [f"## {header}", ""]
        cells = []
        for d in params["datasets"]:
            for a in ("vit_base","resnet50"):
                for ir in on["datasets"][d]["ratios"] if is_online else [None]:
                    line = [d,a,ir if ir is not None else "—"]
                    for m in methods:
                        r = result_lookup.get((stage,d,a,ir,m))
                        if m == "bn_adapt" and a == "vit_base":
                            line.append("N/A")
                            continue
                        expected[header] += 3 if stage == "efficiency" else 45
                        if r and r["complete"]:
                            field = "throughput" if stage == "efficiency" else "top1"
                            line.append(f"{r[field]:.2f} ± {r[field+'_std']:.2f}")
                        elif r:
                            field = "throughput" if stage == "efficiency" else "top1"
                            line.append(f"{r[field]:.2f} ({r['streams']} streams)")
                        else:
                            line.append("待运行")
                    if stage == "online_ir":
                        line.append("N/A" if a == "vit_base" else "Pending Reproduction")
                    cells.append(line)
        text += [table(("Dataset","Backbone","IR",*(NAMES[m] for m in methods),*(["DART"] if stage=="online_ir" else [])),cells), ""]
    text += ["## 配对增益、机制与计时", "",
             "各阶段 summary.csv 记录百分点及配对区间，以及 κ、γ、拟合有效率、Fallback、Macro Accuracy 和 ECE。"
             "Efficiency 表使用图像每秒。逐流同时记录同步端到端时间、专属时间、GPU 显存与前后向次数。"
             "适应开销是同配置方法时间减 No Adapt 时间。PEA 的源统计准备在测试计时之前。",
             "", "## 核验与待完成项", ""]
    statuses = Counter((r["stage"], r["status"]) for r in rows)
    text += [table(("阶段","状态","流"), [(s,v,n) for (s,v),n in sorted(statuses.items())]), ""]
    failed = Counter(r.get("error","") for r in rows+smoke if r["status"] == "failed")
    text.extend(f"- {reason}（{n} 条）" for reason,n in failed.most_common())
    if not (output/"splits").exists() or not any(r["stage"] == "online_ir" and r["status"] == "ok" for r in rows):
        text.append("- Online IR 和在线消融等待可核验的 DART 索引。")
    text += ["", "正式实验的逐流 JSON、CSV 和输入索引保存在 raw/ 与 splits/。以下烟雾数据仅用于验证流程。", ""]
    smoketab = [(r["dataset"],r["architecture"],NAMES.get(r["method"],r["method"]),
                 f"{r['top1']:.2f}" if r["status"] == "ok" else r["status"],r.get("samples","—")) for r in smoke]
    text.append(table(("Dataset","Backbone","Method","Smoke Top-1 (%)","样本"), smoketab))
    save(output/"verification.json", dict(validated_checkpoints=sum(a["status"]=="ok" for a in audits.values()),
         expected=dict(expected), successful=sum(r["status"]=="ok" for r in rows),
         failed=sum(r["status"]=="failed" for r in rows), smoke_successes=sum(r["status"]=="ok" for r in smoke),
         complete=sum(r["status"]=="ok" for r in rows)==sum(expected.values())))
    pending = output/"report.tmp"
    pending.write_text("\n".join(text)+"\n")
    pending.replace(output/"report.md")
    complete = [r for r in summary if r["stage"]=="balanced" and r["method"]=="plead" and r["complete"]]
    if complete:
        fig = Figure(figsize=(8,4),layout="constrained")
        ax = fig.subplots()
        ax.bar([r["dataset"]+"\n"+r["architecture"] for r in complete], [r["top1"] for r in complete],
               yerr=[r["top1_std"] for r in complete], capsize=3)
        ax.set_ylabel("Balanced Top-1 (%)")
        fig.savefig(output/"balanced"/"plead_top1.pdf")
