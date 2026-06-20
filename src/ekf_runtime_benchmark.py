"""Baseline runtime benchmark for a simulated robotics control loop.

This module runs a simple periodic loop, records execution metrics, and exports
the results to CSV for later analysis. It intentionally uses a lightweight
NumPy computation as a stand-in for future EKF or control workload logic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

# Ensure the repository root is importable when this file is executed directly
# via ``python src/ekf_runtime_benchmark.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics_logger import RuntimeMetricsLogger
from src.stress_injector import StressInjector


DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_TARGET_PERIOD_MS = 100.0
DEFAULT_STRESS_MODE = "none"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime benchmark."""

    parser = argparse.ArgumentParser(
        description="Run a baseline simulated robotics runtime benchmark."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Benchmark duration in seconds. Default: 10.0",
    )
    parser.add_argument(
        "--target-period-ms",
        type=float,
        default=DEFAULT_TARGET_PERIOD_MS,
        help="Target control loop period in milliseconds. Default: 100.0",
    )
    parser.add_argument(
        "--stress-mode",
        type=str,
        choices=sorted(StressInjector.VALID_MODES),
        default=DEFAULT_STRESS_MODE,
        help="Stress injection mode. Default: none",
    )
    parser.add_argument(
        "--random-delay-probability",
        type=float,
        default=0.0,
        help="Probability of injecting a random delay. Default: 0.0",
    )
    parser.add_argument(
        "--max-random-delay-ms",
        type=float,
        default=0.0,
        help="Maximum injected random delay in milliseconds. Default: 0.0",
    )
    parser.add_argument(
        "--cpu-work-iterations",
        type=int,
        default=1,
        help="CPU stress workload iterations. Default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic stress behavior.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "CSV output path for logged runtime metrics. "
            "Default: data/runtime_logs/runtime_log_<stress_mode>.csv"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate benchmark configuration supplied by the user."""

    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero.")
    if args.target_period_ms <= 0.0:
        raise ValueError("--target-period-ms must be greater than zero.")
    if not 0.0 <= args.random_delay_probability <= 1.0:
        raise ValueError("--random-delay-probability must be between 0.0 and 1.0.")
    if args.max_random_delay_ms < 0.0:
        raise ValueError("--max-random-delay-ms must be non-negative.")
    if args.cpu_work_iterations < 0:
        raise ValueError("--cpu-work-iterations must be non-negative.")


def resolve_output_path(output: str | None, stress_mode: str) -> str:
    """Resolve the runtime log output path."""

    if output:
        return output
    return f"data/runtime_logs/runtime_log_{stress_mode}.csv"


def simulate_computation(stress_injector: StressInjector) -> None:
    """Run a small deterministic NumPy workload and optional stress."""

    state_vector = np.linspace(0.0, 1.0, 512, dtype=np.float64)
    transition_matrix = np.eye(512, dtype=np.float64)
    _ = transition_matrix @ state_vector
    stress_injector.apply()


def run_benchmark(
    duration_seconds: float,
    target_period_ms: float,
    output_path: str | Path,
    stress_injector: StressInjector,
) -> RuntimeMetricsLogger:
    """Execute the periodic benchmark loop and persist recorded metrics."""

    logger = RuntimeMetricsLogger(target_loop_period_ms=target_period_ms)
    target_period_seconds = target_period_ms / 1000.0
    benchmark_start_time = time.perf_counter()
    benchmark_end_time = benchmark_start_time + duration_seconds
    next_iteration_time = benchmark_start_time

    while time.perf_counter() < benchmark_end_time:
        logger.start_iteration()
        simulate_computation(stress_injector)
        logger.end_iteration()

        next_iteration_time += target_period_seconds
        current_time = time.perf_counter()
        if current_time < next_iteration_time:
            time.sleep(next_iteration_time - current_time)

    logger.save_csv(output_path)
    return logger


def print_summary(logger: RuntimeMetricsLogger, stress_mode: str) -> None:
    """Print a concise summary of benchmark results."""

    metrics = logger.metrics
    total_iterations = len(metrics)

    execution_times = [metric.loop_execution_time_ms for metric in metrics]
    loop_periods = [
        metric.loop_period_ms
        for metric in metrics
        if metric.loop_period_ms is not None
    ]
    jitters = [metric.jitter_ms for metric in metrics if metric.jitter_ms is not None]
    deadline_misses = sum(metric.deadline_miss for metric in metrics)
    deadline_miss_percentage = (
        (deadline_misses / total_iterations) * 100.0 if total_iterations else 0.0
    )

    average_execution_time = mean(execution_times) if execution_times else 0.0
    average_loop_period = mean(loop_periods) if loop_periods else 0.0
    average_jitter = mean(jitters) if jitters else 0.0

    print(f"Stress mode: {stress_mode}")
    print(f"Total iterations: {total_iterations}")
    print(f"Average loop execution time: {average_execution_time:.3f} ms")
    print(f"Average loop period: {average_loop_period:.3f} ms")
    print(f"Average jitter: {average_jitter:.3f} ms")
    print(f"Number of deadline misses: {deadline_misses}")
    print(f"Deadline miss percentage: {deadline_miss_percentage:.2f}%")


def main() -> None:
    """Run the benchmark from command-line configuration."""

    args = parse_args()
    validate_args(args)
    output_path = resolve_output_path(args.output, args.stress_mode)
    stress_injector = StressInjector(
        mode=args.stress_mode,
        random_delay_probability=args.random_delay_probability,
        max_random_delay_ms=args.max_random_delay_ms,
        cpu_work_iterations=args.cpu_work_iterations,
        random_seed=args.seed,
    )

    logger = run_benchmark(
        duration_seconds=args.duration,
        target_period_ms=args.target_period_ms,
        output_path=output_path,
        stress_injector=stress_injector,
    )
    print_summary(logger, args.stress_mode)


if __name__ == "__main__":
    main()
