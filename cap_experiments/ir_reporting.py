"""Verify and aggregate Online IR runs without loading historical predictions."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import t

from cap_experiments.ir_evaluation import METHODS
from cap_experiments.online_ir import OnlineIR, SPECS
from run_cap_neo import write_csv, write_json


def _load(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _stats(values):
    return mean(values), stdev(values) if len(values) > 1 else None


def _seed_means(rows, field):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row[field])
    return [mean(items) for items in by_seed.values()]


def _fmt(value, std):
    return f"{value:.3f}" if std is None else rf"${value:.3f} \pm {std:.3f}$"


def _write_table(path, headings, rows):
    end = " " + chr(92) * 2
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{tabular}{l" + "r" * (len(headings) - 1) + "}",
             r"\toprule", " & ".join(headings) + end, r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + end for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_accuracy(path, summary):
    datasets = sorted({row["dataset"] for row in summary})
    fig = Figure(figsize=(4.4 * len(datasets), 3.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    for axis, dataset in zip(fig.subplots(1, len(datasets), squeeze=False)[0], datasets):
        for method, color in zip(METHODS, ("#30343b", "#0072b2")):
            rows = sorted((row for row in summary if row["dataset"] == dataset and row["method"] == method),
                          key=lambda row: row["ir"])
            axis.errorbar([row["ir"] for row in rows], [row["top1_mean"] for row in rows],
                          yerr=[row["top1_std"] or 0 for row in rows], label=method,
                          color=color, marker="o", capsize=3)
        axis.set_xscale("log")
        axis.set_xticks([r["ir"] for r in rows], [str(r["ir"]) for r in rows], rotation=25)
        axis.set(title=dataset, xlabel="Nominal IR", ylabel="Online Top-1 (%)")
        axis.grid(axis="y", alpha=.2)
        axis.legend(frameon=False)
    fig.savefig(path)


def _plot_efficiency(path, speed):
    fig = Figure(figsize=(7.5, 3.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    locations = np.arange(len(speed))
    for offset, field, label, color in (
        (-.18, "method_speedup", "Adapter + window", "#0072b2"),
        (.18, "end_to_end_speedup", "Single-view pipeline", "#d55e00"),
    ):
        axis.bar(locations + offset, [row[field] for row in speed], .36, label=label, color=color)
    axis.axhline(1, color="#333333", linewidth=.8)
    axis.set_xticks(locations, [row["dataset"] for row in speed])
    axis.set_ylabel("V2 Fit+Gate time / Lite time")
    axis.legend(frameon=False)
    fig.savefig(path)


def _check(record, dataset, config):
    identity = record["identity"]
    sampling = record["sampling"]
    args = config["arguments"]
    classes, size, batch_size, _ = SPECS[dataset]
    indices = np.asarray(sampling["indices"])
    dominant = np.asarray(sampling["dominant_classes"])
    order = np.asarray(sampling["class_order"])
    stream = OnlineIR(indices, dominant, order, np.empty(0), sampling["p_max"], sampling["p_min"])
    if (indices.shape != (classes, size) or not np.array_equal(np.sort(indices.ravel()), np.arange(indices.size))
            or not np.array_equal(np.sort(dominant), np.arange(classes)) or not np.array_equal(np.sort(order), np.arange(classes))
            or stream.sha256 != sampling["sha256"] or stream.index_sha256 != sampling["index_sha256"]):
        raise ValueError(f"Invalid or repeated sampling indices in {identity}")
    num_subsets = args["smoke_subsets"] if args["smoke"] else classes
    batches = num_subsets * ((size + batch_size - 1) // batch_size)
    if (tuple(row["method"] for row in record["rows"]) != METHODS or len(record["subsets"]) != num_subsets
            or len(record["diagnostics"]) != batches or sampling["evaluated_size"] != num_subsets * size):
        raise ValueError(f"Incomplete paired stream in {identity}")
    counts = Counter()
    expected_window = min(128, size)
    arrived = 0
    for b, step in enumerate(record["diagnostics"]):
        batch_index = b % ((size + batch_size - 1) // batch_size)
        expected_size = min(batch_size, size - batch_index * batch_size)
        arrived += expected_size
        if (step["size"] != expected_size or step["bank_size"] != min(expected_window, arrived)
                or sum(count for _, count in step["window_class_counts"]) != step["bank_size"]):
            raise ValueError(f"Window or batch boundary verification failed in {identity}")
    for i, subset in enumerate(record["subsets"]):
        if subset["dominant_class"] != dominant[i] or sum(num for _, num in subset["class_counts"]) != size:
            raise ValueError(f"Invalid subset histogram in {identity}")
        counts.update(dict(subset["class_counts"]))
    if not args["smoke"] and any(counts[c] != size for c in range(classes)):
        raise ValueError(f"Global class population is imbalanced in {identity}")
    for row in record["rows"]:
        if (row["sampling_sha256"] != sampling["sha256"] or row["samples"] != sampling["evaluated_size"]
                or row["batches"] != batches or any(not isfinite(row[field]) for field in ("top1", "top5", "ece"))):
            raise ValueError(f"Stream rows lack finite, paired results in {identity}")
    return identity


def export_report(directories, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, pairs, benchmarking, verified, configs = [], [], [], {}, {}
    by_dominant = defaultdict(lambda: defaultdict(list))
    diag_counts = defaultdict(Counter)
    sample_counts = defaultdict(list)
    with (output / "per_subset.csv.tmp").open("w", newline="") as subsets_log, (
        output / "sampling_audit.csv.tmp"
    ).open("w", newline="") as audit_log:
        subset_writer = audit_writer = None
        for directory in directories:
            cfg = json.loads((directory / "config.json").read_text())
            args = cfg["arguments"]
            dataset = args["dataset"]
            if dataset in configs:
                raise ValueError(f"Duplicate dataset {dataset}")
            configs[dataset] = cfg
            expected = {(corruption, seed, ir) for corruption in args["corruptions"] for seed in args["seeds"] for ir in args["irs"]}
            found = set()
            for path in sorted((directory / "runs").glob("*/*/*.json.gz")):
                record = _load(path)
                event = _check(record, dataset, cfg)
                group = (event["corruption"], event["seed"], event["ir"])
                if group not in expected or group in found or event["dataset"] != dataset:
                    raise ValueError(f"Unexpected run identity in {path}")
                found.add(group)
                rows.extend(record["rows"])
                neo, lite = record["rows"]
                pairs.append({**event, "samples": neo["samples"], "sampling_sha256": neo["sampling_sha256"],
                              **{f"{field}_delta": lite[field] - neo[field] for field in ("top1", "top5", "ece")}})
                for subset in record["subsets"]:
                    item = {**event, **subset, "class_counts": json.dumps(subset["class_counts"], separators=(",", ":")),
                            "sampling_sha256": neo["sampling_sha256"]}
                    if subset_writer is None:
                        subset_writer = csv.DictWriter(subsets_log, fieldnames=list(item))
                        subset_writer.writeheader()
                    subset_writer.writerow(item)
                    audit = {k: v for k, v in item.items() if not k.endswith("_top1")}
                    if audit_writer is None:
                        audit_writer = csv.DictWriter(audit_log, fieldnames=list(audit))
                        audit_writer.writeheader()
                    audit_writer.writerow(audit)
                    bucket = by_dominant[dataset, event["ir"], subset["dominant_class"]]
                    for method in METHODS:
                        bucket[method].append(subset[f"{method}_top1"])
                    sample_counts[dataset, event["ir"]].append(subset["dominant_fraction"])
                counts = diag_counts[dataset, event["ir"]]
                for step in record["diagnostics"]:
                    counts["batches"] += 1
                    counts["gamma_zero"] += step["gamma"] == 0
                    counts["fallback_any"] += bool(step["fallback_reasons"])
                    counts.update(step["fallback_reasons"])
            if found != expected:
                raise ValueError(f"Missing {len(expected - found)} requested streams for {dataset}")
            verified[dataset] = dict(
                scope="smoke" if args["smoke"] else "full",
                seeds=args["seeds"], corruptions=args["corruptions"], irs=args["irs"], streams=len(found),
                distinct_indices=True, paired_streams=True, checked_window_and_batch_boundaries=True,
            )
            benchmarking.extend({"dataset": dataset, **row}
                                for row in json.loads((directory / "benchmark.json").read_text()))

    for stem in ("per_subset.csv", "sampling_audit.csv"):
        (output / (stem + ".tmp")).replace(output / stem)
    summary = []
    grouped = defaultdict(list)
    paired_grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"], row["ir"], row["method"]].append(row)
    for pair in pairs:
        paired_grouped[pair["dataset"], pair["ir"]].append(pair)
    for (dataset, ir, method), entries in sorted(grouped.items()):
        gain = _seed_means(paired_grouped[dataset, ir], "top1_delta")
        row = dict(dataset=dataset, ir=ir, method=method, seeds=len(gain), streams=len(entries),
                   actual_dominant_fraction=mean(sample_counts[dataset, ir]))
        for field in ("top1", "top5", "ece", "method_seconds", "total_inference_seconds", "peak_gpu_mib",
                      "samples_per_second", "kappa", "lambda", "gamma", "neff", "sxx", "correlation",
                      "fallback_rate", "gamma_zero_rate"):
            usable = [entry for entry in entries if entry.get(field) is not None]
            if usable:
                row[f"{field}_mean"], row[f"{field}_std"] = _stats(_seed_means(usable, field))
        if method == "plead_lite":
            avg, sd = _stats(gain)
            ci = sd * t.ppf(.975, len(gain) - 1) / sqrt(len(gain)) if sd is not None else None
            row.update(gain_pp=avg, gain_std=sd, gain_ci95_low=avg - ci if ci is not None else None,
                       gain_ci95_high=avg + ci if ci is not None else None)
        summary.append(row)

    speed = []
    for dataset in configs:
        old, lite = ([r for r in benchmarking if r["dataset"] == dataset and r["method"] == method]
                     for method in ("v2_fit_gate", "plead_lite"))
        record = {"dataset": dataset, "method": "speedup", "repeats": len(lite)}
        for field in ("method_seconds", "total_inference_seconds"):
            record[f"v2_{field}"] = mean(r[field] for r in old)
            record[f"lite_{field}"] = mean(r[field] for r in lite)
        record["method_speedup"] = record["v2_method_seconds"] / record["lite_method_seconds"]
        record["end_to_end_speedup"] = record["v2_total_inference_seconds"] / record["lite_total_inference_seconds"]
        record["max_abs_logit_difference"] = max(r["max_abs_logit_difference"] for r in lite)
        speed.append(record)

    write_csv(output / "per_stream.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_differences.csv", pairs)
    write_csv(output / "efficiency.csv", benchmarking + speed)
    write_csv(output / "dominant_class_summary.csv", [
        {"dataset": dataset, "ir": ir, "dominant_class": c, "n_subsets": len(bucket[METHODS[0]]),
         **{f"{m}_top1": mean(values) for m, values in bucket.items()},
         "top1_gain_pp": mean(bucket["plead_lite"]) - mean(bucket["neo_online"])}
        for (dataset, ir, c), bucket in sorted(by_dominant.items())
    ])
    write_csv(output / "fit_gate_diagnostics.csv", [
        dict(dataset=dataset, ir=ir, diagnostic=reason, count=count, batches=counts["batches"],
             fraction=count / counts["batches"])
        for (dataset, ir), counts in sorted(diag_counts.items()) for reason, count in counts.items() if reason != "batches"
    ])
    write_json(output / "verification.json", verified)
    if len(configs) > 1:
        write_json(output / "config.json", configs)
    table = []
    for lite in (r for r in summary if r["method"] == "plead_lite"):
        neo = next(r for r in summary if r["dataset"] == lite["dataset"] and r["ir"] == lite["ir"] and r["method"] == "neo_online")
        ci = (f"[{lite['gain_ci95_low']:+.3f}, {lite['gain_ci95_high']:+.3f}]"
              if lite["gain_ci95_low"] is not None else "--")
        table.append([lite["dataset"], lite["ir"], _fmt(neo["top1_mean"], neo["top1_std"]),
                      _fmt(lite["top1_mean"], lite["top1_std"]), _fmt(neo["top5_mean"], neo["top5_std"]),
                      _fmt(lite["top5_mean"], lite["top5_std"]), f"{lite['gain_pp']:+.3f}", ci])
    _write_table(output / "online_ir_table.tex",
                 ("Dataset", "IR", "NEO Top-1", "Lite Top-1", "NEO Top-5", "Lite Top-5", "Gain (pp)", r"95\% CI"), table)
    _write_table(output / "efficiency_table.tex",
                 ("Dataset", "V2 total (s)", "Lite total (s)", "Pipeline speedup", "Adapter speedup"),
                 [[row["dataset"], f"{row['v2_total_inference_seconds']:.4f}",
                   f"{row['lite_total_inference_seconds']:.4f}", f"{row['end_to_end_speedup']:.3f}",
                   f"{row['method_speedup']:.3f}"] for row in speed])
    _plot_accuracy(output / "online_ir_accuracy.pdf", summary)
    _plot_efficiency(output / "efficiency_comparison.pdf", speed)
    lines = ["# Online IR results", "",
             "The current batch enters the FIFO before its predictions. Strict prequential scoring predicts before updating.",
             "For ImageNet-C, this protocol uses 50 unique images per subset. The DART paper uses 100.",
             "IR describes the target prior. The cyclic quota rounds the diagonal and allocates the remaining integers, so finite subset proportions may differ.",
             "Intervals are paired across identical corruption and seed, averaging corruptions within seed before applying a Student t interval.", "",
             "| Dataset | Scope | IR | NEO Top-1 | Lite Top-1 | Gain (pp) | Paired 95% CI |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset, ir, neo1, lite1, _, __, gain, ci in table:
        lines.append(f"| {dataset} | {verified[dataset]['scope']} | {ir} | {neo1.replace('$', '').replace(chr(92)+'pm', '±')} | "
                     f"{lite1.replace('$', '').replace(chr(92)+'pm', '±')} | {gain} | {ci} |")
    lines += ["", "## Efficiency", "",
              "The benchmark reports synchronized single-view pipeline time including transfer of preprocessed batches to the device. It warms up both methods before measurement. Method time includes window construction. The Lite cache reading and update time is recorded separately from cache construction.",
              "Cross-configuration feature hits in main experiments are recorded separately in per_stream.csv. The first full stream per corruption constructs the feature cache. Subsequent streams read only the current arrival's frozen features.", ""]
    for row in speed:
        lines.append(f"- {row['dataset']}: pipeline {row['end_to_end_speedup']:.3f}×, adapter {row['method_speedup']:.3f}×, max absolute logit difference {row['max_abs_logit_difference']:.3g}.")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_report(args.runs, args.output)


if __name__ == "__main__":
    main()
