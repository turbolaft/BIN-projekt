import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy import stats
except ImportError:
    stats = None


@dataclass(slots=True)
class RunSummary:
    seed: int
    quant_mode: str
    best_genotype: str
    architecture: str
    best_fitness: float
    val_accuracy: float
    test_accuracy: float
    model_bits: int
    compute_cost: int
    output_file: Path
    log_file: Path


SUMMARY_PATTERNS = {
    "best_genotype": re.compile(r"^Best genotype:\s*(.+)$", re.MULTILINE),
    "architecture": re.compile(r"^Architecture\s*:\s*(.+)$", re.MULTILINE),
    "best_fitness": re.compile(r"^Best fitness\s*:\s*([0-9.]+)$", re.MULTILINE),
    "val_accuracy": re.compile(r"^Val accuracy\s*:\s*([0-9.]+)$", re.MULTILINE),
    "test_accuracy": re.compile(r"^Test accuracy:\s*([0-9.]+)$", re.MULTILINE),
    "efficiency": re.compile(r"^Efficiency\s*:\s*model_bits=(\d+), compute_cost=(\d+)$", re.MULTILINE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated quantization experiments and save recaps.")
    parser.add_argument("--dataset", type=int, default=144)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp32", "mixed", "binary"],
        choices=["fp32", "mixed", "binary"],
    )
    return parser.parse_args()


def run_single_experiment(args: argparse.Namespace, quant_mode: str, seed: int) -> RunSummary:
    mode_dir = args.logs_root / quant_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    log_file = mode_dir / f"run_{seed}.txt"
    output_file = mode_dir / f"run_output_{seed}.txt"

    command = [
        sys.executable,
        "main.py",
        "--quant-mode",
        quant_mode,
        "--dataset",
        str(args.dataset),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--log-file",
        str(log_file),
        "--population-size",
        str(args.population_size),
        "--generations",
        str(args.generations),
        "--epochs",
        str(args.epochs),
        "--quiet",
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    output_text = completed.stdout
    if completed.stderr:
        output_text = f"{output_text}\n[stderr]\n{completed.stderr}"
    output_file.write_text(output_text, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for mode={quant_mode}, seed={seed}. See {output_file}")

    return parse_run_summary(output_text, quant_mode, seed, output_file, log_file)


def parse_run_summary(
    output_text: str,
    quant_mode: str,
    seed: int,
    output_file: Path,
    log_file: Path,
) -> RunSummary:
    matches: dict[str, re.Match[str]] = {}
    for field, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(output_text)
        if match is None:
            raise ValueError(f"Could not parse '{field}' from {output_file}")
        matches[field] = match

    model_bits, compute_cost = matches["efficiency"].groups()
    return RunSummary(
        seed=seed,
        quant_mode=quant_mode,
        best_genotype=matches["best_genotype"].group(1).strip(),
        architecture=matches["architecture"].group(1).strip(),
        best_fitness=float(matches["best_fitness"].group(1)),
        val_accuracy=float(matches["val_accuracy"].group(1)),
        test_accuracy=float(matches["test_accuracy"].group(1)),
        model_bits=int(model_bits),
        compute_cost=int(compute_cost),
        output_file=output_file,
        log_file=log_file,
    )


def write_recap(logs_root: Path, quant_mode: str, summaries: list[RunSummary]) -> Path:
    best_fitness = max(summaries, key=lambda item: item.best_fitness)
    best_val = max(summaries, key=lambda item: item.val_accuracy)
    best_test = max(summaries, key=lambda item: item.test_accuracy)
    smallest_model = min(summaries, key=lambda item: item.model_bits)
    smallest_compute = min(summaries, key=lambda item: item.compute_cost)

    lines = [
        f"Technique: {quant_mode}",
        f"Runs: {len(summaries)}",
        "",
        "Averages",
        f"fitness={statistics.mean(item.best_fitness for item in summaries):.4f}",
        f"val_acc={statistics.mean(item.val_accuracy for item in summaries):.4f}",
        f"test_acc={statistics.mean(item.test_accuracy for item in summaries):.4f}",
        f"model_bits={statistics.mean(item.model_bits for item in summaries):.1f}",
        f"compute_cost={statistics.mean(item.compute_cost for item in summaries):.1f}",
        "",
        "Best by fitness",
        format_summary(best_fitness),
        "",
        "Best validation accuracy",
        format_summary(best_val),
        "",
        "Best test accuracy",
        format_summary(best_test),
        "",
        "Smallest model",
        format_summary(smallest_model),
        "",
        "Smallest compute cost",
        format_summary(smallest_compute),
        "",
        "All runs",
    ]

    for summary in sorted(summaries, key=lambda item: item.seed):
        lines.append(
            (
                f"seed={summary.seed:02d} fitness={summary.best_fitness:.4f} "
                f"val_acc={summary.val_accuracy:.4f} test_acc={summary.test_accuracy:.4f} "
                f"model_bits={summary.model_bits} compute_cost={summary.compute_cost} "
                f"genotype={summary.best_genotype}"
            )
        )

    recap_path = logs_root / f"recap_{quant_mode}.txt"
    recap_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recap_path


def format_summary(summary: RunSummary) -> str:
    return (
        f"seed={summary.seed} fitness={summary.best_fitness:.4f} "
        f"val_acc={summary.val_accuracy:.4f} test_acc={summary.test_accuracy:.4f} "
        f"model_bits={summary.model_bits} compute_cost={summary.compute_cost} "
        f"genotype={summary.best_genotype} architecture={summary.architecture} "
        f"log_file={summary.log_file} output_file={summary.output_file}"
    )


def write_boxplot(
    output_path: Path,
    title: str,
    ylabel: str,
    series: list[list[float]],
    labels: list[str],
) -> None:
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.boxplot(data=series)
    else:
        plt.boxplot(series, labels=labels)
    if sns is not None:
        plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_statistics(logs_root: Path, summaries_by_mode: dict[str, list[RunSummary]]) -> Path:
    stats_path = logs_root / "statistical_tests.txt"
    ordered_modes = [mode for mode in ["binary", "mixed", "fp32"] if mode in summaries_by_mode]
    pretty_names = {
        "binary": "Binary (1-bit)",
        "mixed": "Mixed-Precision",
        "fp32": "FP32 Baseline",
    }

    lines = ["--- STATISTICAL EVALUATION ---"]
    if stats is None:
        lines.append("SciPy is not installed, so Mann-Whitney U tests were skipped.")
        stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return stats_path

    test_acc = {mode: [item.test_accuracy for item in summaries] for mode, summaries in summaries_by_mode.items()}
    fitness = {mode: [item.best_fitness for item in summaries] for mode, summaries in summaries_by_mode.items()}

    test_pairs = [
        ("mixed", "binary", "greater"),
        ("fp32", "binary", "greater"),
        ("fp32", "mixed", "two-sided"),
    ]
    lines.append("Test accuracy")
    for left, right, alternative in test_pairs:
        if left not in test_acc or right not in test_acc:
            continue
        statistic, p_value = stats.mannwhitneyu(test_acc[left], test_acc[right], alternative=alternative)
        lines.append(
            (
                f"{pretty_names[left]} vs {pretty_names[right]} "
                f"(alternative={alternative}): statistic={statistic:.4f}, p-value={p_value:.6f}"
            )
        )

    lines.append("")
    lines.append("Fitness")
    fitness_pairs = [
        ("mixed", "binary", "greater"),
        ("mixed", "fp32", "greater"),
        ("fp32", "binary", "greater"),
    ]
    for left, right, alternative in fitness_pairs:
        if left not in fitness or right not in fitness:
            continue
        statistic, p_value = stats.mannwhitneyu(fitness[left], fitness[right], alternative=alternative)
        lines.append(
            (
                f"{pretty_names[left]} vs {pretty_names[right]} "
                f"(alternative={alternative}): statistic={statistic:.4f}, p-value={p_value:.6f}"
            )
        )

    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats_path


def write_plots_and_stats(logs_root: Path, summaries_by_mode: dict[str, list[RunSummary]]) -> None:
    ordered_modes = [mode for mode in ["binary", "mixed", "fp32"] if mode in summaries_by_mode]
    if not ordered_modes:
        return

    pretty_names = {
        "binary": "Binary (1-bit)",
        "mixed": "Mixed-Precision",
        "fp32": "FP32 Baseline",
    }
    labels = [pretty_names[mode] for mode in ordered_modes]

    test_acc_series = [[item.test_accuracy for item in summaries_by_mode[mode]] for mode in ordered_modes]
    write_boxplot(
        logs_root / "test_accuracy_boxplot.png",
        "Distribution of Test Accuracy across Runs",
        "Test Accuracy",
        test_acc_series,
        labels,
    )

    fitness_series = [[item.best_fitness for item in summaries_by_mode[mode]] for mode in ordered_modes]
    write_boxplot(
        logs_root / "fitness_boxplot.png",
        "Distribution of Fitness across Runs",
        "Fitness",
        fitness_series,
        labels,
    )

    write_statistics(logs_root, summaries_by_mode)


def main() -> None:
    args = parse_args()
    summaries_by_mode: dict[str, list[RunSummary]] = {}

    for quant_mode in args.modes:
        print(f"Running {quant_mode} experiments...")
        summaries: list[RunSummary] = []
        for seed in range(1, args.runs + 1):
            print(f"  seed {seed}/{args.runs}")
            summaries.append(run_single_experiment(args, quant_mode, seed))
        summaries_by_mode[quant_mode] = summaries
        recap_path = write_recap(args.logs_root, quant_mode, summaries)
        print(f"Saved recap to {recap_path}")

    write_plots_and_stats(args.logs_root, summaries_by_mode)
    print(f"Saved plots and statistical tests to {args.logs_root}")


if __name__ == "__main__":
    main()
