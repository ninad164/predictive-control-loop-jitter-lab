"""Runtime metrics logging utilities for real-time robotics benchmark loops.

This module provides a lightweight metrics logger for control or localization
loops that need repeatable timing and resource measurements without bringing in
framework-specific dependencies such as ROS.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import psutil


@dataclass
class RuntimeMetric:
    """Single loop iteration measurement record."""

    timestamp: str
    loop_execution_time_ms: float
    loop_period_ms: Optional[float]
    jitter_ms: Optional[float]
    cpu_percent: float
    memory_percent: float
    deadline_miss: bool
    position_error_m: Optional[float] = None
    covariance_trace: Optional[float] = None


class RuntimeMetricsLogger:
    """Collect runtime telemetry for a periodic robotics software loop.

    The logger is designed for deterministic benchmarking workflows where each
    loop iteration explicitly calls :meth:`start_iteration` and
    :meth:`end_iteration`.

    Timing measurements use ``time.perf_counter()`` to provide high-resolution
    monotonic timing suitable for latency and jitter analysis.

    A deadline miss is recorded when the measured loop execution time exceeds
    the configured target loop period.
    """

    def __init__(self, target_loop_period_ms: float) -> None:
        """Initialize the metrics logger.

        Args:
            target_loop_period_ms: Desired loop period in milliseconds. This is
                used to compute jitter and to determine whether an iteration
                missed its execution deadline.

        Raises:
            ValueError: If ``target_loop_period_ms`` is not positive.
        """
        if target_loop_period_ms <= 0.0:
            raise ValueError("target_loop_period_ms must be greater than zero.")

        self.target_loop_period_ms = float(target_loop_period_ms)
        self._metrics: List[RuntimeMetric] = []
        self._active_iteration_start_perf: Optional[float] = None
        self._active_iteration_timestamp: Optional[str] = None
        self._previous_iteration_start_perf: Optional[float] = None
        # Prime psutil's non-blocking CPU sampler so subsequent readings are
        # more representative for iterative benchmark logging.
        psutil.cpu_percent(interval=None)

    @property
    def metrics(self) -> List[RuntimeMetric]:
        """Return a copy of the recorded metrics."""

        return list(self._metrics)

    def start_iteration(self) -> None:
        """Mark the start of a loop iteration.

        Raises:
            RuntimeError: If an iteration is already active and has not yet been
                closed with :meth:`end_iteration`.
        """
        if self._active_iteration_start_perf is not None:
            raise RuntimeError(
                "start_iteration() called while another iteration is active."
            )

        self._active_iteration_timestamp = datetime.now(timezone.utc).isoformat()
        self._active_iteration_start_perf = time.perf_counter()

    def end_iteration(
        self,
        position_error_m: Optional[float] = None,
        covariance_trace: Optional[float] = None,
    ) -> RuntimeMetric:
        """Mark the end of the active loop iteration and store its metrics.

        Args:
            position_error_m: Optional EKF position error for this iteration.
            covariance_trace: Optional trace of the EKF covariance matrix.

        Returns:
            The recorded metric for the completed iteration.

        Raises:
            RuntimeError: If :meth:`start_iteration` was not called first.
        """
        if self._active_iteration_start_perf is None:
            raise RuntimeError(
                "end_iteration() called without an active iteration."
            )

        end_perf = time.perf_counter()
        start_perf = self._active_iteration_start_perf
        execution_time_ms = (end_perf - start_perf) * 1000.0

        loop_period_ms: Optional[float] = None
        jitter_ms: Optional[float] = None
        if self._previous_iteration_start_perf is not None:
            loop_period_ms = (
                start_perf - self._previous_iteration_start_perf
            ) * 1000.0
            jitter_ms = abs(loop_period_ms - self.target_loop_period_ms)

        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        deadline_miss = execution_time_ms > self.target_loop_period_ms

        metric = RuntimeMetric(
            timestamp=self._active_iteration_timestamp
            or datetime.now(timezone.utc).isoformat(),
            loop_execution_time_ms=execution_time_ms,
            loop_period_ms=loop_period_ms,
            jitter_ms=jitter_ms,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            deadline_miss=deadline_miss,
            position_error_m=position_error_m,
            covariance_trace=covariance_trace,
        )
        self._metrics.append(metric)

        self._previous_iteration_start_perf = start_perf
        self._active_iteration_start_perf = None
        self._active_iteration_timestamp = None

        return metric

    def save_csv(self, output_path: str | Path) -> None:
        """Persist all recorded measurements to a CSV file.

        Args:
            output_path: Destination CSV path. Parent directories are created if
                they do not already exist.
        """
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "loop_execution_time_ms",
            "loop_period_ms",
            "jitter_ms",
            "cpu_percent",
            "memory_percent",
            "deadline_miss",
            "position_error_m",
            "covariance_trace",
        ]

        with destination.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self._metrics:
                writer.writerow(asdict(metric))
