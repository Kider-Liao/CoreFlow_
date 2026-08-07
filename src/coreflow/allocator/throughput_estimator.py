"""Throughput estimator for vLLM instances using binary search."""

from typing import Dict, Optional, Tuple

import numpy as np

from coreflow.predictor.latency import LatencyPredictor
from coreflow.allocator.request_analyzer import RequestAnalyzer


class ThroughputEstimator:
    """Estimates per-instance throughput using latency models and binary search.

    Given request statistics and hardware constraints, finds the maximum
    sustainable request rate (requests/sec) via iterative binary search
    on the scheduler loop budget.
    """

    def __init__(
        self,
        latency_predictor: LatencyPredictor,
        request_analyzer: RequestAnalyzer,
        chunk_size: int = 512,
        cache_swap_latency: float = 0.1,
        context_step: int = 2000,
        max_context_length: int = 200_000,
    ) -> None:
        self._predictor = latency_predictor
        self._analyzer = request_analyzer
        self._chunk_size = chunk_size
        self._swap_latency = cache_swap_latency
        self._context_step = context_step
        self._max_context_length = max_context_length

    @property
    def max_context_idx(self) -> int:
        return self._max_context_length // self._context_step

    # ------------------------------------------------------------------
    # Single-instance throughput (binary search)
    # ------------------------------------------------------------------
    def analyze_throughput(
        self,
        num_layers: int,
        instance_mem_tokens: int,
        request_stats: dict,
        interference: bool = True,
    ) -> float:
        """Estimate max throughput for a single instance with given request pattern.

        Uses binary search to find the maximum request rate such that
        the sum of prefill, decode, and swap times fits in one second.

        Args:
            num_layers: Number of transformer layers.
            instance_mem_tokens: Max tokens the instance's KV cache can hold.
            request_stats: Dict with keys E_decode, E_prompt, E_cached, E_max.
            interference: Whether to account for decode interference.

        Returns:
            Maximum requests per second.
        """
        E_D = request_stats["E_decode"]
        E_P = request_stats["E_prompt"]
        E_C = request_stats["E_cached"]
        E_max = request_stats["E_max"]

        # Memory-limited batch size (reserve one slot for prefill)
        B_max = int(max(1, instance_mem_tokens // (E_C + E_P + E_D // 2)))

        left = 0.0
        right = 20.0

        for _ in range(20):  # Safe iteration limit
            mid = (right + left) / 2.0

            # Cache swap overhead
            time_swap = min(mid * self._swap_latency, 1.0)
            if time_swap >= 1.0:
                right = mid
                continue

            # Expected number of prefill chunks
            E_K = (E_P * mid) / self._chunk_size
            B = int(min(B_max, np.ceil(E_D * mid / max(E_K, 1e-9))))
            if B < 1:
                right = mid
                continue

            # Chunk prefill latency
            L_prefill_chunk = self._predictor.predict_latency(
                num_layers=num_layers,
                decode_batch_size=B,
                avg_decode_length=int(E_C + E_P + E_D // 2),
                longest_decode_length=E_max,
                prefill_query_len=min(E_P, self._chunk_size - B),
                prefill_seq_len=int(E_C + E_P // 2),
                interference=interference,
            )

            time_prefill = min(E_K * L_prefill_chunk, 1.0 - time_swap)
            if time_prefill >= 1.0 - time_swap:
                right = mid
                continue

            # Decode iteration latency
            L_decode = self._predictor.predict_latency(
                num_layers=num_layers,
                decode_batch_size=B,
                avg_decode_length=int(E_C + E_P + E_D // 2),
                longest_decode_length=E_max,
                prefill_query_len=0,
                prefill_seq_len=0,
                interference=interference,
            )

            num_decode_iter = max(0, int(
                (1.0 - time_prefill - time_swap) // max(L_decode, 1e-9)
            ))

            # Tokens per second: decode + prefill + swap contributions
            swap_tokens_contrib = (
                int(B - mid) * int(time_swap // max(L_decode, 1e-9))
            )
            prefill_tokens_contrib = (
                int(B - mid) * int(time_prefill // max(L_prefill_chunk, 1e-9))
            )
            tokens_per_sec = (
                swap_tokens_contrib
                + prefill_tokens_contrib
                + num_decode_iter * B
            )

            if tokens_per_sec > E_D * mid:
                left = mid
            else:
                right = mid

        return left

    # ------------------------------------------------------------------
    # Batch throughput across context ranges
    # ------------------------------------------------------------------
    def attain_instance_throughput(
        self,
        num_layers: int,
        instance_cached_tokens: int,
        agent_name: str,
        interference: bool = True,
        reuse: bool = True,
    ) -> Dict[Tuple[int, int], float]:
        """Compute instance throughput for all context range partitions.

        Iterates over all (prev_context_idx, context_idx) pairs and computes
        the throughput for requests whose context falls within that range.

        Args:
            num_layers: Number of transformer layers.
            instance_cached_tokens: KV cache capacity per instance.
            agent_name: Agent to analyze.
            interference: Account for interference.
            reuse: Enable KV cache reuse.

        Returns:
            Dict mapping (prev_idx, curr_idx) -> throughput (requests/sec).
        """
        max_idx = self.max_context_idx
        cache: Dict[Tuple[int, int], float] = {}

        for prev_idx in range(0, max_idx):
            lower_bound = prev_idx * self._context_step
            for curr_idx in range(prev_idx + 1, max_idx + 1):
                upper_bound = curr_idx * self._context_step
                key = (prev_idx, curr_idx)

                stats = self._analyzer.get_requests_in_range(
                    lower_bound, upper_bound, agent_name, reuse=reuse,
                )
                if stats["avg_decode_length"] == 0:
                    continue

                request_stats = {
                    "E_max": stats["max_length"],
                    "E_prompt": stats["avg_prefill_length"],
                    "E_decode": stats["avg_decode_length"],
                    "E_cached": stats["avg_cached_tokens"],
                }

                base_tp = self.analyze_throughput(
                    num_layers, instance_cached_tokens, request_stats,
                    interference=interference,
                )

                cache[key] = (
                    base_tp
                    * stats["out_proportion"]
                    / max(stats["query_proportion"], 1e-9)
                )

        return cache
