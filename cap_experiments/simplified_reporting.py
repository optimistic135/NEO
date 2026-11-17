"""Verify raw experiment records and export paired statistics and figures."""

from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import csv
import fcntl
import gzip
import json
from math import isfinite, sqrt
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import t

from cap_experiments.ablation_reporting import DATASETS, table
from cap_experiments.online_ir import SPECS
from cap_experiments.simplified_protocol import COMPONENTS, IR_LEVELS, VALIDATION_PER_CLASS, VARIANTS, experiments
from run_cap_neo import SEEDS, all_corruptions, write_csv, write_json
from run_plead_ablation import digest
from tta_library.plead_simplified import FIELDS, MIN_BANK


NAMES = dict(zip(VARIANTS, ("Full PLEAD", "No Composition", "Hard Prior", "Norm-only Fit", "No Evidence Gate",
                            "Current Batch Only", "Fixed γ=0.5", "Window 64", "Window 256", "No Gate, Window 256")))
METRICS = ("top1", "top5", "adaptation_top1", "ece", "fallback_rate", "prior_tv", "kappa", "kappa_update_std",
           "gamma", "gamma_update_std", "d", "v", "a_norm", "b_norm", "q_norm")
IDENTITY = ["dataset", "corruption", "severity", "seed", "IR"]
UPDATE = IDENTITY + ["method", "update_idx", "subset_id", "steps_since_switch", "batch_size", "correct1", "top1",
                     *[key for key in FIELDS if key != "fallback_code"], "fallback_reason", "prior_tv",
                     "true_max_fraction", "adapter_seconds", "true_histogram", "soft_prior_row", "soft_prior_file"]
SUBSET = IDENTITY + ["method", "subset_id", "samples", "top1", "dominant_class", "true_histogram"]


@contextmanager
def csv_output(data, name, cols, compressed):
    path = data / (name + (".csv.gz" if compressed else ".csv"))
    tmp = path.with_suffix(".tmp")
    opener = gzip.open if compressed else open
    options = {"compresslevel": 2} if compressed else {}
    with opener(tmp, "wt", newline="", **options) as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()

        def add(identity, row):
            w.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                        for key, value in (identity | row).items()})

        yield add
    tmp.replace(path)


def verify(raw, smoke):
    info, manifest = raw["identity"], raw["manifest"]
    dataset, ir = info["dataset"], info["IR"]
    c, subset_size, batch, _ = SPECS[dataset]
    a, b = manifest["adaptation"], manifest["balanced_validation"]
    size = 512 if ir is None else subset_size
    limit = 1 if ir is None else (2 if smoke else c)
    batch = 64 if ir is None else batch
    flat = [i for sub in a for i in sub]
    if (digest(manifest) != raw["split_sha256"] or len(a) != limit or any(len(x) != size for x in a)
            or len(flat) != len(set(flat)) or len(b) != len(set(b)) or set(flat).intersection(b)
            or len(b) != (c * VALIDATION_PER_CLASS[dataset] if ir is None else 0)):
        raise ValueError(f"Invalid manifest: {info}")
    methods = {r["method"]: r for r in raw["rows"]}
    allowed = {m for ms in experiments(dataset, ir).values() for m in ms}
    if set(methods) - allowed or len(methods) != len(raw["rows"]):
        raise ValueError(f"Invalid method keys: {info}")
    per = (size + batch - 1) // batch
    todo = Counter()
    arrivals = Counter()
    for u in raw["updates"]:
        m, idx = u["method"], todo[u["method"]]
        expected = min(batch, size - idx % per * batch)
        arrivals[m] += expected
        conf = VARIANTS[m]
        if (m not in methods or u["update_idx"] != idx or u["batch_size"] != expected
                or u["subset_id"] != idx // per or u["steps_since_switch"] != idx % per
                or u["bank_size"] != (expected if conf.batch_only else min(conf.window, arrivals[m]))
                or not 0 <= u["gamma"] <= 1):
            raise ValueError(f"FIFO verification failed: {info}/{m}/{idx}")
        todo[m] += 1
    subset_count = Counter(r["method"] for r in raw["subsets"])
    for m, row in methods.items():
        if (todo[m] != limit * per or subset_count[m] != limit or row["samples"] != (len(b) or len(flat))
                or any(isinstance(val, float) and not isfinite(val) for val in row.values())):
            raise ValueError(f"Incomplete or nonfinite method output: {info}/{m}")
    return methods


def seed_average(rows, field):
    values = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            values[row["seed"]].append(row[field])
    return {s: mean(xs) for s, xs in sorted(values.items())}


def stats(values):
    return (mean(values), stdev(values) if len(values) > 1 else None) if values else (None, None)


def group_statistics(groups, pairs, required):
    summary, paired, seed_deltas = [], [], []
    for key, records in sorted(groups.items(), key=lambda z: str(z[0])):
        suite, dataset, ir, method = key
        row = dict(experiment=suite, dataset=dataset, IR=ir, method=method, streams=len(records), expected=required)
        for k in METRICS:
            row[k + "_mean"], row[k + "_std"] = stats(list(seed_average(records, k).values()))
        deltas = seed_average(pairs.get(key, []), "delta_pp")
        avg, sd = stats(list(deltas.values()))
        ci = t.ppf(.975, 2) * sd / sqrt(3) if len(deltas) == 3 and len(pairs[key]) == required else None
        row.update(delta_pp=avg, ci_low=avg-ci if ci is not None else None, ci_high=avg+ci if ci is not None else None)
        summary.append(row)
        if deltas:
            metadata = {k: row[k] for k in ("experiment", "dataset", "IR", "method", "expected", "delta_pp", "ci_low", "ci_high")}
            paired.append(dict(**metadata, streams=len(pairs[key]), seeds=len(deltas), delta_std=sd))
            n = Counter(d["seed"] for d in pairs[key])
            seed_deltas.extend(dict(experiment=suite, dataset=dataset, IR=ir, method=method, seed=s,
                                    corruption_count=n[s], delta_pp=val) for s, val in deltas.items())
    return summary, paired, seed_deltas


def fmt(x, precision=3):
    return "—" if x is None else f"{x:.{precision}f}"


def report(path, summary, coverage, checks, gate, window, mechanisms, timing, failures, smoke):
    def section(title, kinds):
        rows = []
        for r in summary:
            if r["experiment"] not in kinds:
                continue
            ci = f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] is not None else "—"
            top = fmt(r["top1_mean"]) + (f" ± {fmt(r['top1_std'])}" if r["top1_std"] is not None else "")
            rows.append((DATASETS[r["dataset"]], r["IR"] or "—", NAMES[r["method"]], f"{r['streams']}/{r['expected']}",
                         top, fmt(r["delta_pp"]), ci, fmt(r["ece_mean"]), fmt(r["prior_tv_mean"]),
                         fmt(r["fallback_rate_mean"]), fmt(r["gamma_mean"]), fmt(r["kappa_mean"])))
        return [f"## {title}", ""] + (table(
            ("数据集", "IR", "方法", "流数", "Top-1 (%)", "相对 Full (pp)", "95% CI",
             "ECE", "TV", "Fallback 比例", "γ", "κ"), rows) if rows else ["待完成。"]) + [""]

    lines = ["# Simplified PLEAD Experiment Report", "", "## 1. Method Overview", "",
             "在冻结 ViT-Base 的 FIFO 中用上一次校正求 soft prior。按分类头中心化权重投影，"
             "用无权重最小二乘估计 κ，再用 q = â − γb̂ 更新当前批校正。γ = clip(1 − v/(d+ε), 0, 1)。",
             f"最少 {MIN_BANK} 张图；拟合无效时保留上一次 q。精确均匀且两半一致时，所给公式的 d=v=0，因此 γ=1；此时均匀先验的几何贡献为零。",
             "标准污染为 512 张适应图和独立均衡验证。Online IR 批内更新后评分，子流之间延续状态。ImageNet-C 每子流 50 张无重复图。",
             "每个 seed 内先平均扰动；完整三种子的配对差值使用 Student t（df=2）。相对 Full 为变体减 Full。",
             "", "## 2. Experiment Completion", "",
             f"{'Gaussian Noise 烟雾测试' if smoke else '正式实验'}已完成 {checks['complete_streams']} / {checks['expected_streams']} 条全配置流，"
             f"已保存 {checks['method_rows']} / {checks['expected_method_rows']} 条方法记录。", ""]
    lines += table(("实验", "已完成方法记录", "预期"), [(k, v[0], v[1]) for k, v in coverage.items()])
    lines += ["", *([f"- {x}" for x in failures] or ["未发现已记录的失败。"]), ""]
    for label, kinds in (("3. Standard Corruption Ablation", ("standard",)),
                         ("4. Online IR Ablation", ("online_ir",)),
                         ("5. Prior-Evidence Gate", ("gate",))):
        lines += section(label, kinds)
    if gate:
        lines += table(("数据集", "IR", "方法", "γ / 真实集中度相关系数", "切换前 γ", "切换后 γ", "d", "v"),
                       [(DATASETS[r["dataset"]], r["IR"], NAMES[r["method"]], fmt(r["gamma_concentration_r"]),
                         fmt(r["pre_switch_gamma"]), fmt(r["post_switch_gamma"]), fmt(r["d"]), fmt(r["v"])) for r in gate])
        lines += ["", "γ 分箱准确率、窗口切换后的逐步准确率分别见 gate_diagnostics.csv 和 class_switch.csv。", ""]
    lines += section("6. FIFO Window", ("window",))
    switches = {(r["dataset"], r["IR"], r["method"], r["steps_since_switch"]): r for r in window}
    deltas = [(DATASETS[d], ir, NAMES[m], fmt(r["top1"] - switches[d, ir, m, -1]["top1"]))
              for (d, ir, m, step), r in switches.items() if step == 0 and (d, ir, m, -1) in switches
              and m in ("window_64", "plead", "window_256")]
    if deltas:
        lines += table(("数据集", "IR", "方法", "切换后首批 − 切换前末批 (pp)"), deltas)
        lines += ["", "子流前后类别组成不同，差值描述观察到的变化。", ""]
    lines += ["## 7. Mechanism Analysis", ""]
    if mechanisms:
        lines += table(("数据集", "IR", "方法", "TV", "κ", "γ", "‖â‖", "‖b̂‖", "‖q‖", "Fallback 次数"),
                       [(DATASETS[r["dataset"]], r["IR"], NAMES[r["method"]], *(fmt(r[k]) for k in
                         ("prior_tv", "kappa", "gamma", "a_norm", "b_norm", "q_norm")), r["fallbacks"]) for r in mechanisms])
        lines += ["", "真实集中度与 γ 的相关系数只描述关联。逐更新先验通过 mechanism.csv 的路径与行号读取。", ""]
    else:
        lines += ["待完成。", ""]
    lines += section("组合删减", ("combination",))
    lines += ["耗时只使用单配置冷缓存行。共享推理运行数据标有 shared_experiment。", ""]
    if timing:
        lines += table(("数据集", "IR", "方法", "Adapter 秒", "推理秒", "主干批次"),
                       [(DATASETS[r["dataset"]], r["IR"] or "—", NAMES[r["method"]], fmt(r["adapter_seconds"]),
                         fmt(r["total_seconds"]), r["backbone_forward_count"]) for r in timing]) + [""]
    lines += ["## 8. Final Simplification Decision", ""]
    gate_full = [r for r in summary if r["experiment"] == "gate" and r["method"] == "no_evidence_gate" and r["ci_low"] is not None]
    if len(gate_full) == 6 and all(r["delta_pp"] >= 0 for r in gate_full):
        lines += ["六个完整门控设置中，固定 γ=1 的点估计均达到或超过自适应门控。建议删除门控以简化方法；"
                  "效应大小与不确定性见门控表。", ""]
    elif len(gate_full) == 6:
        lines += ["固定 γ=1 的配对差值在门控表中按 IR 完整列出；是否保留需同时考虑均衡和高 IR。", ""]
    else:
        lines += ["门控去留等待六个条件的完整配对结果。", ""]
    for name in ("no_composition", "hard_prior", "norm_only_kappa", "current_batch_only"):
        ready = [r for r in summary if r["method"] == name and r["experiment"] in ("standard", "online_ir") and r["ci_low"] is not None]
        below, above = sum(r["ci_high"] < 0 for r in ready), sum(r["ci_low"] > 0 for r in ready)
        if len(ready) != 12:
            decision = "待补齐全部条件"
        elif below and not above:
            decision = "建议保留对应机制"
        elif above and not below:
            decision = "建议考虑该简化版本"
        elif below and above:
            decision = "条件间存在方向相反的结果，保留场景差异"
        else:
            decision = "当前区间均覆盖零，独立收益尚未确定"
        lines.append(f"- {NAMES[name]}：完整条件 {len(ready)}/12，区间全低于零 {below} 个，全高于零 {above} 个。{decision}。")
    lines += ["- FIFO 默认保留 128。窗口敏感性用于描述表现与切换响应。", "", "## 9. Key Findings", ""]
    for kind, label in (("standard", "标准污染"), ("online_ir", "Online IR"), ("gate", "门控"), ("window", "窗口")):
        measured = [r for r in summary if r["experiment"] == kind and r["delta_pp"] is not None]
        if measured:
            low, high = min(measured, key=lambda r: r["delta_pp"]), max(measured, key=lambda r: r["delta_pp"])
            lines.append(f"- {label}已测配对差值范围 {low['delta_pp']:+.3f} 至 {high['delta_pp']:+.3f} pp，"
                         f"对应 {NAMES[low['method']]} 与 {NAMES[high['method']]}。")
    lines += ["", "## 10. Files", "", *[f"- [{p.name}](../data/{p.name})" for p in sorted((path.parent / "data").glob("*.csv*"))],
              "- [核验清单](../verification.json)", "- [图片](figures/)", "- 原始流和先验向量在 `../data/raw/`。"]
    (path / "REPORT.md").write_text("\n".join(lines) + "\n")


def diagnostics(gate_moments, bins, boundaries, mechanism):
    gates = []
    for (d, ir, m), x in sorted(gate_moments.items()):
        n, g, y = x["n"], x["g"], x["y"]
        denominator = sqrt(max(0, n*x["gg"]-g*g) * max(0, n*x["yy"]-y*y))
        gates.append(dict(dataset=d, IR=ir, method=m, gamma_concentration_r=(n*x["gy"]-g*y)/denominator
                          if denominator and m == "plead" else None,
                          pre_switch_gamma=x["pre"] / x["pre_n"] if x["pre_n"] else None,
                          post_switch_gamma=x["post"] / x["post_n"] if x["post_n"] else None,
                          d=x["d"]/n, v=x["v"]/n))
    gate_bins = [dict(dataset=d, IR=ir, method=m, gamma_low=i/5, gamma_high=(i+1)/5, samples=x["n"],
                      top1=100*x["correct"]/x["n"]) for (d, ir, m, i), x in sorted(bins.items())]
    changes = [dict(dataset=d, IR=ir, method=m, steps_since_switch=i, batches=x["n"],
                    top1=100*x["correct"]/x["samples"], gamma=x["g"]/x["n"])
               for (d, ir, m, i), x in sorted(boundaries.items())]
    mech = [dict(dataset=d, IR=ir, method=m, fallbacks=x["fallbacks"],
                 **{k: x[k]/x[k+"_n"] if x[k+"_n"] else None for k in ("prior_tv", "kappa", "gamma", "a_norm", "b_norm", "q_norm")})
            for (d, ir, m), x in sorted(mechanism.items())]
    return gates, gate_bins, changes, mech


def plot_csv(data, output, file, name, ylabel, key):
    with (data / file).open() as handle:
        rows = list(csv.DictReader(handle))
    fig = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    groups = defaultdict(list)
    for r in rows:
        if r.get(key):
            label = DATASETS[r["dataset"]] + (f" IR={r['IR']}" if r.get("IR") else " standard")
            groups[label].append(r)
    xticks = {}
    for label, group in groups.items():
        if name == "class_shift_dynamics":
            for method in ("plead", "no_composition"):
                rr = [r for r in group if r["method"] == method]
                if rr:
                    ax.plot([int(r["steps_since_switch"]) for r in rr], [float(r[key]) for r in rr],
                            marker="o", label=f"{label}, {NAMES[method]}")
            ax.set_xlabel("Updates from subset boundary (−1 = last update before switch)")
        else:
            order = (COMPONENTS if name == "component_ablation" else
                     ("no_composition", "gamma_half", "no_evidence_gate", "plead") if name == "gate_analysis"
                     else ("window_64", "plead", "window_256"))
            lookup = {r["method"]: r for r in group}
            active = [(i, lookup[m]) for i, m in enumerate(order) if m in lookup]
            ax.plot([i for i, _ in active], [float(r[key]) for _, r in active], marker="o", label=label)
            xticks.update({i: NAMES[m] for i, m in enumerate(order)})
    if xticks:
        ax.set_xticks(list(xticks), list(xticks.values()), rotation=16)
    if groups:
        ax.legend(ncol=2, frameon=False, fontsize=7)
    else:
        ax.text(.5, .5, "Awaiting measured data", transform=ax.transAxes, ha="center")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(output / f"{name}.png", dpi=180)


def export(root, smoke, compressed=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _export(root, smoke, compressed)


def _export(root, smoke, compressed):
    data, report_dir = root / "data", root / "report"
    raw_root, figures = data / "raw", report_dir / "figures"
    data.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    corruptions = ("gaussian_noise",) if smoke else all_corruptions
    seeds = (2020,) if smoke else SEEDS
    write_json(root / "configs" / "method.json", dict(method="PLEAD", schema="simplified-plead-v1",
        min_bank=MIN_BANK, precision="FP32", window=128, geometry="unweighted nonnegative OLS",
        fallback="retain previous q", gate="clip(1 - v/(d + float32_eps), 0, 1)",
        relative_sxx_floor="max(dtype.tiny, dtype.eps**2 * sum(t**2))"))
    write_json(root / "configs" / "ablations.json", {m: asdict(x) for m, x in VARIANTS.items()})
    write_json(root / "configs" / "experiment.json", dict(
        scope="smoke" if smoke else "full", corruptions=corruptions, seeds=seeds, severity=5,
        settings={d: {str(ir): experiments(d, ir) for ir in (None, *levels)} for d, levels in IR_LEVELS.items()},
        standard=VALIDATION_PER_CLASS, online_ir_specs=SPECS, adaptation_samples=512,
        batch_transductive=True, large_csv_gzip=compressed, smoke_subsets=2 if smoke else None))
    plan = {(d, c, s, ir): experiments(d, ir) for d, levels in IR_LEVELS.items()
            for c in corruptions for s in seeds for ir in (None, *levels)}
    expected = Counter()
    for groups in plan.values():
        expected.update({k: len(v) for k, v in groups.items()})
    seen, observed = {}, Counter()
    groups, pairs = defaultdict(list), defaultdict(list)
    moments, bins, boundary, mech = (defaultdict(Counter) for _ in range(4))
    per_stream, timing, pair_rows = [], [], []
    fallbacks = Counter()
    with ExitStack() as stack:
        add_update = stack.enter_context(csv_output(data, "per_update", UPDATE, compressed))
        add_subset = stack.enter_context(csv_output(data, "per_subset", SUBSET, compressed))
        add_mech = stack.enter_context(csv_output(data, "mechanism", UPDATE, compressed))
        for suite in ("standard", "online_ir", "cold"):
            pattern = "*/*/*/*.json.gz" if suite != "cold" else "*/*/*/*/*.json.gz"
            for path in sorted((raw_root / suite).glob(pattern)):
                with gzip.open(path, "rt") as handle:
                    raw = json.load(handle)
                identity = raw["identity"]
                dataset, corruption, seed, ir = (identity[k] for k in ("dataset", "corruption", "seed", "IR"))
                stream = (dataset, corruption, seed, ir)
                methods = verify(raw, smoke)
                if suite == "cold":
                    if len(methods) != 1 or any(r["timing_scope"] != "single_cold" for r in methods.values()):
                        raise ValueError(f"Invalid cold-cache record {path}")
                    timing.extend(identity | r for r in methods.values())
                    continue
                if stream in seen or stream not in plan or raw["schema"] != "simplified-plead-v1":
                    raise ValueError(f"Unexpected raw stream {path}")
                seen[stream] = set(methods)
                sha = raw["split_sha256"]
                per_stream.extend(identity | r | {"split_sha256": sha} for r in methods.values())
                for kind, variants in plan[stream].items():
                    for m in variants:
                        if m not in methods:
                            continue
                        observed[kind] += 1
                        key = (kind, dataset, ir, m)
                        groups[key].append(identity | methods[m])
                        if m != "plead" and "plead" in methods:
                            pair = dict(**identity, experiment=kind, method=m, split_sha256=sha,
                                        delta_pp=methods[m]["top1"]-methods["plead"]["top1"])
                            pairs[key].append(pair)
                            pair_rows.append(pair)
                prior_file = None
                prior_index = 0
                previous = {}
                for u in raw["updates"]:
                    add_update(identity, u)
                    m = u["method"]
                    key = (dataset, ir, m)
                    if u["fallback_reason"]:
                        fallbacks[dataset, ir, m, u["fallback_reason"]] += 1
                    if "true_histogram" in u:
                        add_mech(identity, u)
                        stats_counter = mech[key]
                        stats_counter["fallbacks"] += bool(u["fallback_reason"])
                        for k in ("prior_tv", "kappa", "gamma", "a_norm", "b_norm", "q_norm"):
                            if u[k] is not None:
                                stats_counter[k] += u[k]
                                stats_counter[k+"_n"] += 1
                    if u.get("soft_prior_file"):
                        prior_file = root / u["soft_prior_file"]
                        if u["soft_prior_row"] != prior_index:
                            raise ValueError(f"Prior row order mismatch: {path}")
                        prior_index += 1
                    if ir is None:
                        continue
                    gate = m in plan[stream].get("gate", ())
                    track_boundary = gate or m in plan[stream].get("window", ()) or m in plan[stream].get("mechanism", ())
                    g, y = u["gamma"], u["true_max_fraction"]
                    last = previous.get(m)
                    switch = last is not None and u["subset_id"] != last["subset_id"]
                    if gate:
                        x = moments[key]
                        x.update(n=1, g=g, y=y, gg=g*g, yy=y*y, gy=g*y, d=u["d"] or 0, v=u["v"] or 0)
                        if switch:
                            x.update(pre=last["gamma"], pre_n=1, post=g, post_n=1)
                        bins[(*key, min(4, int(g*5)))].update(n=u["batch_size"], correct=u["correct1"])
                    if track_boundary:
                        step = u["steps_since_switch"]
                        if switch:
                            boundary[(*key, -1)].update(n=1, correct=last["correct1"],
                                samples=last["batch_size"], g=last["gamma"])
                        if u["subset_id"] > 0 and step < 8:
                            boundary[(*key, step)].update(n=1, correct=u["correct1"], samples=u["batch_size"], g=g)
                    previous[m] = u
                if prior_file is not None:
                    with np.load(prior_file, allow_pickle=False) as values:
                        arr = values["soft_prior"]
                        if arr.shape != (prior_index, SPECS[dataset][0]) or not np.isfinite(arr).all():
                            raise ValueError(f"Invalid saved soft priors: {prior_file}")
                for r in raw["subsets"]:
                    add_subset(identity, r)
    summary, paired, seeds_out = group_statistics(groups, pairs, len(corruptions)*len(seeds))
    gates, gate_bins, changes, mech_summary = diagnostics(moments, bins, boundary, mech)
    exports = dict(per_stream=per_stream, paired_summary=paired, paired_seed_differences=seeds_out,
                   paired_differences=pair_rows, gate_diagnostics=gate_bins, gate_switch_summary=gates,
                   class_switch=changes, mechanism_summary=mech_summary, timing=timing,
                   component_ablation=[r for r in summary if r["experiment"] in ("standard", "online_ir")],
                   gate_ablation=[r for r in summary if r["experiment"] == "gate"],
                   window_sensitivity=[r for r in summary if r["experiment"] == "window"],
                   combination_ablation=[r for r in summary if r["experiment"] == "combination"],
                   fallback_reasons=[dict(dataset=d, IR=ir, method=m, reason=f, count=n)
                                     for (d, ir, m, f), n in fallbacks.items()])
    for name, rows in exports.items():
        dest = data / f"{name}.csv"
        if rows:
            write_csv(dest, rows)
        else:
            dest.write_text("status\n未完成\n")
    for file, name, label, key in (
        ("component_ablation.csv", "component_ablation", "Top-1 (%)", "top1_mean"),
        ("gate_ablation.csv", "gate_analysis", "Online Top-1 (%)", "top1_mean"),
        ("window_sensitivity.csv", "window_sensitivity", "Online Top-1 (%)", "top1_mean"),
        ("class_switch.csv", "class_shift_dynamics", "Online Top-1 (%)", "top1"),
    ):
        plot_csv(data, figures, file, name, label, key)
    failures = [f"{log.name} · {line}" for log in sorted((root / "logs").glob("*.log"))
                for line in log.read_text(errors="replace").splitlines()
                if line.startswith(("RuntimeError:", "ValueError:", "OSError:", "FileNotFoundError:", "PermissionError:"))]
    missing = [dict(dataset=d, corruption=c, seed=s, IR=ir,
                    missing_methods=sorted({m for ms in v.values() for m in ms}-seen.get((d,c,s,ir), set())))
               for (d,c,s,ir), v in plan.items()
               if {m for ms in v.values() for m in ms}-seen.get((d,c,s,ir), set())]
    checks = dict(complete=not missing, expected_streams=len(plan), complete_streams=len(plan)-len(missing),
                  method_rows=len(per_stream), expected_method_rows=sum(len({m for ms in v.values() for m in ms})
                  for v in plan.values()), cold_timing_records=len(timing), failures=failures, missing=missing,
                  finite_metrics_verified=True, fifo_and_split_verified=True)
    write_json(root / "verification.json", checks)
    report(report_dir, summary, {k: (observed[k], v) for k, v in expected.items()},
           checks, gates, changes, mech_summary, timing, failures, smoke)
