"""Interference profiler for decode kernel interference between sequences."""

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from vllm import _custom_ops as ops
try:
    from vllm.vllm_flash_attn import flash_attn_with_kvcache
except ImportError:
    from vllm_flash_attn import flash_attn_with_kvcache

from coreflow.profiler.base import BaseProfiler


class InterferenceProfiler(BaseProfiler):
    """Profiles decode interference between long and short sequences.

    Measures how a long sequence's decode latency is affected by
    co-scheduled short sequences in the same batch.
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_size: int = 128,
        total_num_blocks: int = 100_000,
        block_size: int = 16,
        device: str = "cuda:2",
        dtype: torch.dtype = torch.bfloat16,
        warmup: int = 5,
        repeat: int = 5,
    ) -> None:
        super().__init__(device=device, dtype=dtype, warmup=warmup, repeat=repeat)
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._total_num_blocks = total_num_blocks
        self._block_size = block_size

        self._kv_cache: tuple | None = None

    @property
    def hidden_size(self) -> int:
        return self._num_heads * self._head_size

    @property
    def kv_size(self) -> int:
        return self._num_kv_heads * self._head_size

    @property
    def scale(self) -> float:
        return self._head_size ** -0.5

    def _init_kv_cache(self) -> None:
        if self._kv_cache is None:
            self._kv_cache = (
                torch.randn(
                    (self._total_num_blocks, self._block_size,
                     self._num_kv_heads, self._head_size),
                    device=self._device, dtype=self._dtype,
                ),
                torch.randn(
                    (self._total_num_blocks, self._block_size,
                     self._num_kv_heads, self._head_size),
                    device=self._device, dtype=self._dtype,
                ),
            )

    def _forward(
        self,
        batch_size: int,
        seq_lengths: List[int],
    ) -> None:
        """Run one decode forward pass for the given batch."""
        max_seq_len = max(seq_lengths)
        num_blocks = int((max_seq_len - 1) // self._block_size + 1)

        slot_mapping = torch.randint(
            0, self._total_num_blocks, (batch_size,),
            device=self._device, dtype=torch.long,
        )
        block_tables = torch.randint(
            0, self._total_num_blocks, (batch_size, num_blocks),
            device=self._device, dtype=torch.int32,
        )
        seq_lens = torch.tensor(
            seq_lengths, device=self._device, dtype=torch.int32,
        )

        q = torch.randn(
            (batch_size, self.hidden_size),
            device=self._device, dtype=self._dtype,
        )
        k = torch.randn(
            (batch_size, self.kv_size),
            device=self._device, dtype=self._dtype,
        )
        v = torch.randn(
            (batch_size, self.kv_size),
            device=self._device, dtype=self._dtype,
        )

        q_reshaped = q.view(-1, self._num_heads, self._head_size)
        k_reshaped = k.view(-1, self._num_kv_heads, self._head_size)
        v_reshaped = v.view(-1, self._num_kv_heads, self._head_size)

        key_cache, value_cache = self._kv_cache
        ops.reshape_and_cache_flash(
            k_reshaped, v_reshaped,
            key_cache, value_cache,
            slot_mapping.flatten(),
            "auto",
            torch.tensor(1.0, device=self._device, dtype=torch.float32),
            torch.tensor(1.0, device=self._device, dtype=torch.float32),
        )

        flash_attn_with_kvcache(
            q_reshaped.unsqueeze(1),
            key_cache, value_cache,
            block_table=block_tables,
            cache_seqlens=seq_lens,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            softcap=0.0,
        )

    def profile_increment(self) -> Dict[str, dict]:
        """Profile the incremental decode cost per additional sequence.

        Returns:
            Dict mapping seq_len -> {'initial': float, 'diff': float}.
        """
        self._init_kv_cache()
        result: Dict[str, dict] = {}

        small_lengths = [1] + list(range(1000, 129000, 1000))

        for small_length in small_lengths:
            length_results: List[float] = []

            for num_small_seqs in range(4, 64, 4):
                seq_lengths = [small_length] * num_small_seqs

                def forward():
                    self._forward(num_small_seqs, seq_lengths)

                latency = self._measure_latency(forward)
                length_results.append(latency)

            model = LinearRegression()
            x = np.array(list(range(4, 64, 4))).reshape(-1, 1)
            y = np.array(length_results).reshape(-1, 1)
            model.fit(x, y)

            result[str(small_length)] = {
                "initial": float(model.intercept_[0]),
                "diff": float(model.coef_[0][0]),
            }

        return result

    def profile_interference(
        self,
        increment_data: Dict[str, dict],
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        """Profile decode interference between long and short sequences.

        Args:
            increment_data: Output from :meth:`profile_increment`.

        Returns:
            Nested dict: {batch_size: {large_seq_len: {small_seq_len: latency}}}.
        """
        self._init_kv_cache()
        interference: Dict[int, Dict[int, Dict[int, float]]] = (
            defaultdict(lambda: defaultdict(dict))
        )

        large_batch_size = 1
        for small_batch_size in range(1, 64):
            for large_seq_len in [16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000]:
                for small_seq_len in range(1000, 9000, 1000):
                    # --- Mixed batch (large + small) ---
                    seq_lengths_mixed = [large_seq_len] * large_batch_size + [small_seq_len] * small_batch_size

                    def forward_mixed():
                        self._forward(large_batch_size + small_batch_size, seq_lengths_mixed)

                    latency_mixed = self._measure_latency(forward_mixed)

                    # --- Small only ---
                    seq_lengths_small = [small_seq_len] * small_batch_size

                    def forward_small():
                        self._forward(small_batch_size, seq_lengths_small)

                    latency_small = self._measure_latency(forward_small)

                    # --- Large only ---
                    seq_lengths_large = [large_seq_len] * large_batch_size

                    def forward_large():
                        self._forward(large_batch_size, seq_lengths_large)

                    latency_large = self._measure_latency(forward_large)

                    # Interference = mixed - small_only - large_only + baseline
                    baseline = increment_data[str(large_seq_len)]["initial"]
                    inter_val = max(
                        0.0,
                        latency_mixed - latency_small - latency_large + baseline,
                    )
                    interference[small_batch_size + 1][large_seq_len][small_seq_len] = inter_val

        return {k: dict(v) for k, v in interference.items()}

    def run(self) -> dict:
        """Run full interference profiling.

        Returns:
            Nested dict of interference latency values.
        """
        increment = self.profile_increment()
        return self.profile_interference(increment)

    def save(self, path: str) -> None:
        profile = self.run()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f)
