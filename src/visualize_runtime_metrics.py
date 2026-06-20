"""Visualization utilities for runtime benchmark CSV logs.

This module loads one or more runtime metric CSV files and generates a standard
set of comparison plots for timing and system telemetry analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/plots")
REQUIRED_COLUMNS = {
    "timestamp",
    "loop_execution_time_ms",
    "loop_period_ms",
    "jitter_ms",
    "cpu_percent",
    "memory_percent",
    "deadline_miss",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for runtime metric visualization."""

    parser = argparse.ArgumentParser(
        description="Generate plots from one or more runtime benchmark CSV logs."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more runtime log CSV files to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PNG plots will be saved. Default: results/plots",
    )
    return parser.parse_args()


def load_runtime_logs(csv_paths: Iterable[str]) -> list[tuple[str, pd.DataFrame]]:
    """Load and validate runtime metric CSV files.

    Returns:
        A list of ``(label, dataframe)`` tuples, where the label is derived from
        the CSV filename stem.
    """

    datasets: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        dataframe = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"CSV file '{path}' is missing required columns: {missing_display}"
            )

        dataframe = dataframe.copy()
        dataframe["deadline_miss"] = dataframe["deadline_miss"].astype(bool)
        datasets.append((path.stem, dataframe))

    return datasets


def style_axes(axis: plt.Axes, title: str, x_label: str, y_label: str) -> None:
    """Apply consistent styling to a plot axis."""

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle="--", alpha=0.5)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()


def save_figure(figure: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save a Matplotlib figure as a PNG and close it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_individual_histogram(
    label: str,
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot a histogram for a single runtime log with its mean marked."""

    figure, axis = plt.subplots(figsize=(10, 6))
    series = dataframe[column].dropna()
    axis.hist(series, bins=30, alpha=0.7, label=label, edgecolor="black")

    if not series.empty:
        mean_value = series.mean()
        axis.axvline(
            mean_value,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label=f"Mean = {mean_value:.3f}",
        )

    style_axes(axis, title=title, x_label=x_label, y_label="Frequency")
    save_figure(figure, output_dir, filename)


def plot_comparison_histogram(
    datasets: list[tuple[str, pd.DataFrame]],
    column: str,
    title: str,
    x_label: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot an overlaid histogram comparing a metric across runtime logs."""

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, dataframe in datasets:
        series = dataframe[column].dropna()
        axis.hist(series, bins=30, alpha=0.5, label=label, edgecolor="black")

    style_axes(axis, title=title, x_label=x_label, y_label="Frequency")
    save_figure(figure, output_dir, filename)


def plot_deadline_miss_bar_chart(
    datasets: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Plot deadline miss counts for one or more runtime logs."""

    labels = [label for label, _ in datasets]
    miss_counts = [
        int(dataframe["deadline_miss"].sum()) for _, dataframe in datasets
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, miss_counts, label="Deadline Misses")
    axis.bar_label(bars, padding=3)
    style_axes(
        axis,
        title="Deadline Miss Comparison by Runtime Log",
        x_label="Runtime Log",
        y_label="Miss Count",
    )
    save_figure(figure, output_dir, "deadline_miss_comparison.png")


def plot_execution_time_vs_iteration(
    datasets: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Plot loop execution time against iteration index for each runtime log."""

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, dataframe in datasets:
        iterations = range(len(dataframe))
        axis.plot(
            iterations,
            dataframe["loop_execution_time_ms"],
            linewidth=1.5,
            label=label,
        )

    style_axes(
        axis,
        title="Loop Execution Time vs Iteration",
        x_label="Iteration",
        y_label="Execution Time (ms)",
    )
    save_figure(figure, output_dir, "execution_time_vs_iteration.png")


def generate_plots(
    datasets: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Generate the full set of runtime metric plots."""

    for label, dataframe in datasets:
        plot_individual_histogram(
            label=label,
            dataframe=dataframe,
            column="loop_execution_time_ms",
            title=f"Loop Execution Time Histogram: {label}",
            x_label="Loop Execution Time (ms)",
            filename=f"loop_execution_time_histogram_{label}.png",
            output_dir=output_dir,
        )
        plot_individual_histogram(
            label=label,
            dataframe=dataframe,
            column="jitter_ms",
            title=f"Jitter Histogram: {label}",
            x_label="Jitter (ms)",
            filename=f"jitter_histogram_{label}.png",
            output_dir=output_dir,
        )
        plot_individual_histogram(
            label=label,
            dataframe=dataframe,
            column="cpu_percent",
            title=f"CPU Utilization Histogram: {label}",
            x_label="CPU Utilization (%)",
            filename=f"cpu_histogram_{label}.png",
            output_dir=output_dir,
        )
        plot_individual_histogram(
            label=label,
            dataframe=dataframe,
            column="memory_percent",
            title=f"Memory Utilization Histogram: {label}",
            x_label="Memory Utilization (%)",
            filename=f"memory_histogram_{label}.png",
            output_dir=output_dir,
        )

    plot_comparison_histogram(
        datasets=datasets,
        column="loop_execution_time_ms",
        title="Loop Execution Time Comparison",
        x_label="Loop Execution Time (ms)",
        filename="execution_time_comparison.png",
        output_dir=output_dir,
    )
    plot_comparison_histogram(
        datasets=datasets,
        column="jitter_ms",
        title="Jitter Comparison",
        x_label="Jitter (ms)",
        filename="jitter_comparison.png",
        output_dir=output_dir,
    )
    plot_comparison_histogram(
        datasets=datasets,
        column="cpu_percent",
        title="CPU Utilization Comparison",
        x_label="CPU Utilization (%)",
        filename="cpu_comparison.png",
        output_dir=output_dir,
    )
    plot_comparison_histogram(
        datasets=datasets,
        column="memory_percent",
        title="Memory Utilization Comparison",
        x_label="Memory Utilization (%)",
        filename="memory_comparison.png",
        output_dir=output_dir,
    )
    plot_deadline_miss_bar_chart(datasets=datasets, output_dir=output_dir)
    plot_execution_time_vs_iteration(datasets=datasets, output_dir=output_dir)


def main() -> None:
    """Load runtime logs and generate comparison figures."""

    args = parse_args()
    datasets = load_runtime_logs(args.csv_files)
    generate_plots(datasets, Path(args.output_dir))


if __name__ == "__main__":
    main()
