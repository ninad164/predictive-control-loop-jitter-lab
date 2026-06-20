"""Stress injection utilities for runtime benchmarking experiments.

This module provides a configurable stress injector that can introduce timing
delays, CPU load, or both to emulate degraded runtime conditions in a robotics
software loop.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np


class StressInjector:
    """Apply configurable stress patterns to a benchmark iteration.

    Supported modes:

    - ``none``: no stress is applied
    - ``random_delay``: probabilistically inject a short sleep delay
    - ``cpu_stress``: execute a CPU-heavy NumPy workload
    - ``mixed``: combine random delay injection with CPU stress
    """

    VALID_MODES = {"none", "random_delay", "cpu_stress", "mixed"}

    def __init__(
        self,
        mode: str = "none",
        random_delay_probability: float = 0.0,
        max_random_delay_ms: float = 0.0,
        cpu_work_iterations: int = 1,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the stress injector.

        Args:
            mode: Stress mode to apply.
            random_delay_probability: Probability in the range ``[0.0, 1.0]``
                of applying a random timing delay when the mode supports it.
            max_random_delay_ms: Maximum delay duration in milliseconds for
                random delay injection.
            cpu_work_iterations: Number of repeated CPU workload iterations to
                execute when CPU stress is enabled.
            random_seed: Optional seed for deterministic random delay behavior.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported stress mode '{mode}'. "
                f"Expected one of: {sorted(self.VALID_MODES)}."
            )
        if not 0.0 <= random_delay_probability <= 1.0:
            raise ValueError(
                "random_delay_probability must be between 0.0 and 1.0."
            )
        if max_random_delay_ms < 0.0:
            raise ValueError("max_random_delay_ms must be non-negative.")
        if cpu_work_iterations < 0:
            raise ValueError("cpu_work_iterations must be non-negative.")

        self.mode = mode
        self.random_delay_probability = float(random_delay_probability)
        self.max_random_delay_ms = float(max_random_delay_ms)
        self.cpu_work_iterations = int(cpu_work_iterations)
        self._random = random.Random(random_seed)

    def apply(self) -> None:
        """Apply the configured stress pattern for one loop iteration."""

        if self.mode == "none":
            return
        if self.mode == "random_delay":
            self._apply_random_delay()
            return
        if self.mode == "cpu_stress":
            self._apply_cpu_stress()
            return

        self._apply_random_delay()
        self._apply_cpu_stress()

    def _apply_random_delay(self) -> None:
        """Inject a probabilistic sleep delay to emulate timing disturbance."""

        if self.random_delay_probability <= 0.0 or self.max_random_delay_ms <= 0.0:
            return
        if self._random.random() > self.random_delay_probability:
            return

        delay_ms = self._random.uniform(0.0, self.max_random_delay_ms)
        time.sleep(delay_ms / 1000.0)

    def _apply_cpu_stress(self) -> None:
        """Execute CPU-heavy NumPy work to emulate compute saturation."""

        if self.cpu_work_iterations <= 0:
            return

        base_matrix = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float64).reshape(
            128, 128
        )
        working_matrix = base_matrix.copy()

        # Repeated dense linear algebra provides a compact, reproducible CPU
        # workload without adding framework-specific dependencies.
        for _ in range(self.cpu_work_iterations):
            working_matrix = np.tanh(working_matrix @ base_matrix.T)
