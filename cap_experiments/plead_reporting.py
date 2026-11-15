"""Publication tables and paired seed-level comparisons for PLEAD-V2."""

from collections import defaultdict
from statistics import mean, stdev

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import t

from run_cap_neo import write_csv


AB_METRICS = (
    "adaptation_top1", "balanced_validation_top1", "balanced_validation_top5",
    "adaptation_ece", "balanced_validation_ece", "fallback_rate",
    "kappa", "lambda", "gamma", "neff", "sxx", "correlation",
    "method_seconds", "inference_seconds", "peak_gpu_mib",
)
LT_METRICS = (
    "matched_validation_top1", "matched_validation_top5", "balanced_validation_top1",
    "matched_validation_ece", "balanced_validation_ece", "fallback_rate", "method_seconds",
)


def _stats(values, field):
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else None
    return {f"{field}_mean": average, f"{field}_std": deviation}


def summarize(rows, experiment):
    fields = AB_METRICS if experiment == "ablation" else LT_METRICS
    reference = "plead_original" if experiment == "ablation" else "neo"
    score = "balanced_validation_top1" if experiment == "ablation" else "matched_validation_top1"
    reference_rows = {
        (r["rho"], r["seed"], r["corruption"]): r for r in rows if r["method"] == reference
    }
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["rho"], row["method"]][row["seed"]].append(row)
    result = []
    for (rho, method), seeds in groups.items():
        summary = {"rho": rho, "method": method, "n_seeds": len(seeds), "reference": reference}
        paired = []
        for seed_rows in seeds.values():
            gains = []
            for row in seed_rows:
                base = reference_rows[row["rho"], row["seed"], row["corruption"]]
                if row["split_sha256"] != base["split_sha256"]:
                    raise ValueError("Paired comparison has mismatched splits")
                gains.append(row[score] - base[score])
            paired.append(mean(gains))
        summary.update(_stats(paired, "gain_pp"))
        if len(paired) > 1:
            half = t.ppf(.975, len(paired) - 1) * stdev(paired) / len(paired) ** .5
            summary.update(gain_ci95_low=mean(paired) - half, gain_ci95_high=mean(paired) + half)
        for field in fields:
            per_seed = [[r[field] for r in rs if r.get(field) is not None] for rs in seeds.values()]
            values = [mean(rs) for rs in per_seed if rs]
            if values:
                summary.update(_stats(values, field))
        result.append(summary)
    return result


def _cell(row, field):
    value, std = row.get(f"{field}_mean"), row.get(f"{field}_std")
    if value is None:
        return "--"
    return f"{value:.3f}" if std is None else rf"${value:.3f} \\pm {std:.3f}$"


def write_table(path, rows, experiment):
    specs = (
        (("Adapt Top-1", "adaptation_top1"), ("Balanced Top-1", "balanced_validation_top1"),
         ("Top-5", "balanced_validation_top5"), ("ECE", "balanced_validation_ece"),
         ("Fallback", "fallback_rate"), ("Method (s)", "method_seconds"))
        if experiment == "ablation" else
        (("Matched Top-1", "matched_validation_top1"), ("Matched Top-5", "matched_validation_top5"),
         ("Balanced Top-1", "balanced_validation_top1"), ("ECE", "matched_validation_ece"),
         ("Fallback", "fallback_rate"), ("Method (s)", "method_seconds"))
    )
    include_rho = experiment == "longtail"
    columns = ([r"$\rho$"] if include_rho else []) + ["Method", *(title for title, _ in specs), "Gain (pp), 95\\% CI"]
    end = " " + chr(92) * 2
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{tabular}{" + ("r" if include_rho else "") +
             "l" + "r" * (len(specs) + 1) + "}", r"\toprule", " & ".join(columns) + end, r"\midrule"]
    for row in rows:
        gain = f"{row['gain_pp_mean']:+.3f}"
        if "gain_ci95_low" in row:
            gain += f" [{row['gain_ci95_low']:+.3f}, {row['gain_ci95_high']:+.3f}]"
        cells = ([str(row["rho"])] if include_rho else [])
        cells += [row["method"].replace("_", r"\_"), *(_cell(row, field) for _, field in specs), gain]
        lines.append(" & ".join(cells) + end)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def plot_longtail(path, rows):
    fig = Figure(figsize=(8, 3.5), constrained_layout=True)
    FigureCanvasAgg(fig)
    for method, color in (("neo", "#343a40"), ("plead_v2", "#0072b2")):
        series = sorted((row["rho"], row) for row in rows if row["method"] == method)
        for axis, phase, label in zip(
            fig.axes or fig.subplots(1, 2), ("matched_validation", "balanced_validation"),
            ("Matched validation", "Balanced validation"),
        ):
            axis.errorbar(
                [rho for rho, _ in series], [row[f"{phase}_top1_mean"] for _, row in series],
                yerr=[row[f"{phase}_top1_std"] or 0 for _, row in series],
                label=method, color=color, marker="o", capsize=3,
            )
            axis.set_xscale("log")
            axis.set_xticks([1, 10, 100], labels=["1", "10", "100"])
            axis.set_xlabel("Nominal imbalance ratio ρ")
            axis.set_ylabel("Top-1 accuracy (%)")
            axis.set_title(label)
            axis.grid(axis="y", alpha=.25)
    fig.axes[-1].legend(frameon=False)
    fig.savefig(path)


def generate(output, experiment, rows):
    if not rows:
        raise ValueError(f"No {experiment} results")
    summary = summarize(rows, experiment)
    write_csv(output / f"{experiment}_per_stream.csv", rows)
    write_csv(output / f"{experiment}_summary.csv", summary)
    write_table(output / f"{experiment}_table.tex", summary, experiment)
    if experiment == "longtail":
        plot_longtail(output / "longtail_accuracy.pdf", summary)
