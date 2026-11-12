"""Summarize CAP-NEO runs using seeds as the unit of replication."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


PHASES = ("adaptation", "validation", "balanced_validation", "matched_validation")
METRICS = tuple(f"{phase}_{metric}" for phase in PHASES for metric in ("top1", "top5", "entropy", "ece"))
DIAGNOSTICS = (
    "kappa", "lambda", "correction_norm", "feature_shift_norm", "fallback_updates", "fallback_rate",
    "prior_total_variation", "inference_seconds", "shared_backbone_seconds", "method_seconds",
    "stream_wall_seconds", "peak_gpu_mib", "method_peak_extra_gpu_mib",
)
COLORS = {
    "neo": "#343a40", "neo_shrink": "#0072b2", "cap_pa": "#009e73",
    "cap_neo": "#d55e00", "cap_oracle_labels": "#8b58a1",
}
LEGACY_PHASES = {"adaptation": "adapt", "validation": "val"}
LEGACY_METRICS = {"top1": "accuracy1", "top5": "accuracy5", "entropy": "entropy", "ece": "ece"}


def _write_csv(output, name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path = output / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_stage(directory):
    configuration = json.loads((directory / "config.json").read_text())
    args = configuration["arguments"]
    stage = args["stage"]
    with (directory / "per_corruption.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scenarios = [] if stage == "imbalance" else ["natural"]
    if stage in {"imbalance", "oracle"}:
        scenarios.extend(f"ratio_{ratio:03d}" for ratio in args["ratios"])
    expected = {
        (seed, corruption, stream, method)
        for seed in args["seeds"] for corruption in args["corruptions"]
        for stream in scenarios for method in args["methods"]
    }
    found = set()
    for row in rows:
        row["seed"] = int(row["seed"])
        row["ratio"] = int(row["ratio"]) if row["ratio"] else None
        key = (row["seed"], row["corruption"], row["stream"], row["method"])
        if key in found or row["stage"] != stage:
            raise ValueError(f"Duplicate or misidentified record {key} in {directory}")
        found.add(key)
        for metric in (*METRICS, *DIAGNOSTICS):
            value = row.get(metric)
            row[metric] = float(value) if value else None
            if row[metric] is not None and not math.isfinite(row[metric]):
                raise ValueError(f"Nonfinite {metric} for {key} in {directory}")
        required = ("adaptation", "validation") if row["ratio"] is None else (
            "adaptation", "balanced_validation", "matched_validation"
        )
        if not row["split_sha256"] or any(
            row[f"{phase}_{metric}"] is None for phase in required for metric in LEGACY_METRICS
        ):
            raise ValueError(f"Incomplete metrics or split for {key} in {directory}")
    if expected != found:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"Incomplete stage {directory}: missing {missing[:5]}, extra {extra[:5]}")
    return configuration, rows


def _comparable(first, other):
    fields = (
        "seeds", "corruptions", "level", "adapt_samples", "batch_size",
        "model_revision", "processor_revision",
    )
    return (
        all(first["arguments"][key] == other["arguments"][key] for key in fields)
        and first["model"] == other["model"]
        and first["processor"] == other["processor"]
        and first["cap"] == other["cap"]
    )


def _verify_split_identity(rows):
    split_hashes = {}
    for row in rows:
        key = (row["seed"], row["corruption"], row["stream"])
        digest = row["split_sha256"]
        if split_hashes.setdefault(key, digest) != digest:
            raise ValueError(f"Compared methods used different ordered splits: {key}")


def _statistics(values):
    return mean(values), stdev(values) if len(values) > 1 else None


def _aggregate(rows, keys):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = tuple(row[key] for key in keys)
        groups[group][row["seed"]].append(row)
    summary = []
    for group, by_seed in sorted(groups.items()):
        result = dict(zip(keys, group))
        result["n_seeds"] = len(by_seed)
        result["n_corruptions"] = len({row["corruption"] for subset in by_seed.values() for row in subset})
        for metric in (*METRICS, *DIAGNOSTICS):
            values = []
            for subset in by_seed.values():
                available = [row[metric] for row in subset if row[metric] is not None]
                if available:
                    values.append(mean(available))
            if values:
                result[f"{metric}_mean"], result[f"{metric}_std"] = _statistics(values)
        summary.append(result)
    return summary


def _ablation(rows):
    by_method = {
        (row["seed"], row["corruption"], row["method"]): row for row in rows
        if row["stage"] == "ablation"
    }
    contrasts = (
        ("neo_shrink", "neo"), ("cap_pa", "neo"), ("cap_neo", "neo"),
        ("cap_neo", "neo_shrink"), ("cap_neo", "cap_pa"),
    )
    result = []
    for method, reference in contrasts:
        per_seed = defaultdict(list)
        for (seed, corruption, candidate), row in by_method.items():
            baseline = by_method.get((seed, corruption, reference))
            if candidate == method and baseline is not None:
                per_seed[seed].append({
                    metric: row[metric] - baseline[metric]
                    for metric in METRICS if row[metric] is not None
                })
        if not per_seed:
            continue
        item = {"method": method, "reference": reference, "n_seeds": len(per_seed)}
        for metric in per_seed[next(iter(per_seed))][0]:
            deltas = [mean(record[metric] for record in subset) for subset in per_seed.values()]
            item[f"delta_{metric}_mean"], item[f"delta_{metric}_std"] = _statistics(deltas)
        result.append(item)
    return result


def _paired_diagnostics(rows):
    baseline = {}
    for row in rows:
        if row["method"] == "neo":
            key = (row["seed"], row["corruption"], row["stream"])
            baseline[key] = row
    paired = []
    for row in rows:
        key = (row["seed"], row["corruption"], row["stream"])
        reference = baseline.get(key)
        if reference is None or row["method"] == "neo":
            continue
        for phase in PHASES[1:]:
            metric = f"{phase}_top1"
            if row[metric] is None:
                continue
            paired.append({
                "method": row["method"], "seed": row["seed"], "corruption": row["corruption"],
                "stream": row["stream"], "ratio": row["ratio"], "phase": phase,
                "kappa": row["kappa"], "lambda": row["lambda"],
                "delta_top1": row[metric] - reference[metric],
                "prior_total_variation": row["prior_total_variation"],
                "fallback_updates": row["fallback_updates"],
            })
    return paired


def _save_figure(figure, output, name):
    FigureCanvasAgg(figure)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=240)


def _plot_imbalance(summary, output):
    controlled = [row for row in summary if row["stream"].startswith("ratio_")]
    if not controlled:
        return
    figure = Figure(figsize=(10.4, 4.4), constrained_layout=True)
    axes = figure.subplots(1, 2, sharey=True)
    for axis, phase, title in zip(
        axes, ("balanced_validation", "matched_validation"),
        ("Balanced validation", "Prior-matched validation"),
    ):
        for method, color in COLORS.items():
            series = sorted(
                (int(row["stream"].split("_")[-1]), row)
                for row in controlled if row["method"] == method
            )
            if series:
                axis.errorbar(
                    [ratio for ratio, _ in series],
                    [row[f"{phase}_top1_mean"] for _, row in series],
                    yerr=[row[f"{phase}_top1_std"] or 0 for _, row in series],
                    label=method, color=color, marker="o", capsize=3,
                    linestyle="--" if method == "cap_oracle_labels" else "-",
                )
        axis.set_title(title)
        axis.set_xlabel("Target dominant class share in adaptation (%)")
        axis.set_xticks(sorted({int(row["stream"].split("_")[-1]) for row in controlled}))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation Top-1 accuracy (%)")
    axes[1].legend(frameon=False)
    _save_figure(figure, output, "imbalance_accuracy")


def _plot_diagnostics(paired, output):
    points = [row for row in paired if row["phase"] in {"validation", "balanced_validation"}]
    if not points:
        return
    figure = Figure(figsize=(10.4, 4.4), constrained_layout=True)
    axes = figure.subplots(1, 2, sharey=True)
    for axis, field, xlabel in zip(axes, ("kappa", "lambda"), ("Prototype scale κ", "Shrinkage λ")):
        for method, color in COLORS.items():
            subset = [row for row in points if row["method"] == method]
            if subset:
                axis.scatter(
                    [row[field] for row in subset], [row["delta_top1"] for row in subset],
                    label=method, c=color, alpha=0.45, s=13, edgecolors="none",
                )
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation Top-1 change vs NEO (pp)")
    axes[1].legend(frameon=False)
    _save_figure(figure, output, "correction_diagnostics")


def _compare_legacy(rows, directory):
    records = []
    for row in rows:
        if row["stage"] != "baseline":
            continue
        corruption, seed = row["corruption"], row["seed"]
        file = directory / f"neo-cifar10-base-{seed}-{corruption}.json"
        if not file.is_file():
            continue
        original = json.loads(file.read_text())[corruption]
        for phase, short_phase in LEGACY_PHASES.items():
            for metric, short_metric in LEGACY_METRICS.items():
                value = original.get(f"avg_{short_phase}_{short_metric}")
                if value is None or value == -1:
                    continue
                measured = row[f"{phase}_{metric}"]
                records.append({
                    "seed": seed, "corruption": corruption, "phase": phase, "metric": metric,
                    "original": value, "reproduction": measured, "difference": measured - value,
                    "split_verified": False, "original_file": str(file),
                })
    return records


def _compare_baselines(rows):
    seen = {}
    for row in rows:
        if row["method"] != "neo" or row["stream"] != "natural":
            continue
        key = (row["seed"], row["corruption"])
        if key not in seen:
            seen[key] = row
            continue
        original = seen[key]
        for metric in METRICS:
            value = row[metric]
            if value is not None and not math.isclose(value, original[metric], abs_tol=1e-4):
                raise ValueError(f"Baseline and ablation NEO disagree on {key} {metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Optional original main.py results directory")
    args = parser.parse_args()

    configs, rows = {}, []
    for directory in args.runs:
        configuration, stage_rows = _load_stage(directory)
        stage = configuration["arguments"]["stage"]
        if stage in configs:
            raise ValueError(f"Provide one output directory per stage, found duplicate {stage}")
        if configs and not _comparable(next(iter(configs.values())), configuration):
            raise ValueError(f"Incompatible model, split or CAP settings in {directory}")
        configs[stage] = configuration
        rows.extend(stage_rows)
    _verify_split_identity(rows)
    _compare_baselines(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    overall = _aggregate(rows, ("stage", "stream", "method"))
    per_corruption = _aggregate(rows, ("stage", "stream", "corruption", "method"))
    natural_stage = "ablation" if "ablation" in configs else "baseline"
    methods = [row for row in overall if row["stream"] == "natural" and row["stage"] in {natural_stage, "oracle"}]
    diagnostics = _paired_diagnostics(rows)
    _write_csv(args.output, "overall_summary.csv", overall)
    _write_csv(args.output, "per_corruption_accuracy.csv", per_corruption)
    _write_csv(args.output, "method_comparison.csv", methods)
    _write_csv(args.output, "ablation_summary.csv", _ablation(rows))
    _write_csv(args.output, "paired_diagnostics.csv", diagnostics)
    if args.reference_dir is not None:
        legacy = _compare_legacy(rows, args.reference_dir)
        _write_csv(args.output, "legacy_baseline_comparison.csv", legacy)
        print(f"Original main.py comparisons: {len(legacy)} metric pairs (split identity unverified)")
    _plot_imbalance(overall, args.output)
    _plot_diagnostics(diagnostics, args.output)
    (args.output / "report_config.json").write_text(
        json.dumps({
            "stages": configs,
            "run_directories": [str(directory.resolve()) for directory in args.runs],
            "reference_directory": str(args.reference_dir.resolve()) if args.reference_dir else None,
        }, indent=2) + "\n"
    )
    print(f"Summarized {len(rows)} method/stream records in {args.output}")


if __name__ == "__main__":
    main()
