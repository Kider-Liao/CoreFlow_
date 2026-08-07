"""Abstract base class for GPU kernel profilers.

Provides CUDA device management, warmup/repeat measurement patterns,
and timing utilities.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import numpy as np
import torch


class BaseProfiler(ABC):
    """Base class for GPU kernel profiling.

    Manages CUDA device context, warmup/repeat cycles, and metric aggregation.
    """

    def __init__(
        self,
        device: str = "cuda:2",
        dtype: torch.dtype = torch.bfloat16,
        warmup: int = 5,
        repeat: int = 10,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._warmup = warmup
        self._repeat = repeat

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _measure_latency(
        self,
        forward_fn: Callable[[], None],
        warmup: int = 0,
        repeat: int = 0,
    ) -> float:
        """Measure median GPU kernel latency.

        Runs warmup iterations, then records repeat iterations with
        CUDA synchronization before and after each.

        Args:
            forward_fn: Function that launches the GPU kernel.
            warmup: Number of warmup iterations (default: self._warmup).
            repeat: Number of measurement iterations (default: self._repeat).

        Returns:
            Median latency in seconds.
        """
        warmup = warmup or self._warmup
        repeat = repeat or self._repeat

        for _ in range(warmup):
            forward_fn()

        latencies: List[float] = []
        for _ in range(repeat):
            torch.cuda.synchronize(device=self._device)
            start = time.time()
            forward_fn()
            torch.cuda.synchronize(device=self._device)
            latencies.append(time.time() - start)

        return float(np.median(latencies))

    @abstractmethod
    def run(self) -> Any:
        """Execute profiling and return collected data."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save profiling results to a JSON file."""
