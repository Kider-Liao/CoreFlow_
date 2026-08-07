"""Attention kernel profiler for decode and prefill latency."""

import json
from collections import defaultdict
from typing import Dict, Tuple

import torch
from vllm import _custom_ops as ops
try:
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from coreflow.profiler.base import BaseProfiler


class AttentionProfiler(BaseProfiler):
    """Profiles attention kernel latency for decode and prefill phases.

    Records per-(batch_size, total_tokens) median latencies and saves
    to a JSON profile file for later use by the predictor.
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
        repeat: int = 10,
    ) -> None:
        super().__init__(device=device, dtype=dtype, warmup=warmup, repeat=repeat)
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._total_num_blocks = total_num_blocks
        self._block_size = block_size

        self._kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None

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
        """Initialize or reuse KV cache tensor."""
        if self._kv_cache is None:
            self._kv_cache = (
                torch.randn(
                    (self._total_num_blocks, self._block_size,
                     self._num_kv_heads, self._head_size),
                    device=self._device,
                    dtype=self._dtype,
                ),
                torch.randn(
                    (self._total_num_blocks, self._block_size,
                     self._num_kv_heads, self._head_size),
                    device=self._device,
                    dtype=self._dtype,
                ),
            )

    def _decode_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
    ) -> None:
        """Run one decode forward pass."""
        num_tokens = query.shape[0]
        query_reshaped = query.view(-1, self._num_heads, self._head_size)
        key_reshaped = key.view(-1, self._num_kv_heads, self._head_size)
        value_reshaped = value.view(-1, self._num_kv_heads, self._head_size)

        key_cache, value_cache = self._kv_cache

        ops.reshape_and_cache_flash(
            key_reshaped, value_reshaped,
            key_cache, value_cache,
            slot_mapping.flatten(),
            "auto",
            torch.tensor(1.0, device=self._device, dtype=torch.float32),
            torch.tensor(1.0, device=self._device, dtype=torch.float32),
        )

        flash_attn_with_kvcache(
            query_reshaped.unsqueeze(1),
            key_cache, value_cache,
            block_table=block_tables,
            cache_seqlens=seq_lens_tensor,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            softcap=0.0,
        )

    def _prefill_forward(
        self,
        query: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_start_loc: torch.Tensor,
        max_query_len: int,
        seq_len: int,
        block_tables: torch.Tensor,
    ) -> None:
        """Run one prefill forward pass."""
        query_reshaped = query.view(-1, self._num_heads, self._head_size)
        key_cache, value_cache = self._kv_cache

        flash_attn_varlen_func(
            q=query_reshaped,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            cu_seqlens_k=None,
            seqused_k=torch.tensor(
                [seq_len], device=self._device, dtype=torch.int32,
            ),
            max_seqlen_k=seq_len,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            block_table=block_tables,
            softcap=0.0,
        )

    def profile_decode(
        self,
        batch_sizes: range,
        seq_lengths: range,
    ) -> Dict[int, Dict[int, float]]:
        """Profile decode attention latency.

        Args:
            batch_sizes: Range of batch sizes to test.
            seq_lengths: Range of sequence lengths to test.

        Returns:
            Nested dict: {batch_size: {total_tokens: latency_seconds}}.
        """
        self._init_kv_cache()
        results: Dict[int, Dict[int, float]] = defaultdict(dict)

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                num_blocks = int((seq_len - 1) // self._block_size + 1)

                def forward():
                    slot_mapping = torch.randint(
                        0, self._total_num_blocks, (batch_size,),
                        device=self._device, dtype=torch.long,
                    )
                    block_tables = torch.randint(
                        0, self._total_num_blocks, (batch_size, num_blocks),
                        device=self._device, dtype=torch.int32,
                    )
                    seq_lens = torch.full(
                        (batch_size,), seq_len,
                        device=self._device, dtype=torch.int32,
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
                    self._decode_forward(q, k, v, slot_mapping, block_tables, seq_lens)

                latency = self._measure_latency(forward)
                results[batch_size][seq_len * batch_size] = latency

        return dict(results)

    def profile_prefill(
        self,
        query_lengths: range,
        seq_lengths: range,
    ) -> Dict[int, Dict[int, float]]:
        """Profile prefill attention latency.

        Args:
            query_lengths: Range of new token counts to test.
            seq_lengths: Range of KV cache lengths to test.

        Returns:
            Nested dict: {query_len: {seq_len: latency_seconds}}.
        """
        self._init_kv_cache()
        results: Dict[int, Dict[int, float]] = defaultdict(dict)

        for query_len in query_lengths:
            for seq_len in seq_lengths:
                num_blocks = int((seq_len - 1) // self._block_size + 1)

                def forward():
                    block_tables = torch.randint(
                        0, self._total_num_blocks, (1, num_blocks),
                        device=self._device, dtype=torch.int32,
                    )
                    q = torch.randn(
                        (query_len, self.hidden_size),
                        device=self._device, dtype=self._dtype,
                    )
                    query_start = torch.tensor(
                        [0, query_len], device=self._device, dtype=torch.int32,
                    )
                    seq_start = torch.tensor(
                        [0, seq_len], device=self._device, dtype=torch.int32,
                    )
                    self._prefill_forward(
                        q, query_start, seq_start,
                        max_query_len=query_len, seq_len=seq_len,
                        block_tables=block_tables,
                    )

                latency = self._measure_latency(forward)
                results[query_len][seq_len] = latency

        return dict(results)

    def run(
        self,
        decode_batch_sizes: Tuple[int, int, int] = (16, 256, 16),
        decode_seq_lengths: Tuple[int, int, int] = (128, 10240, 128),
        prefill_query_lengths: Tuple[int, int, int] = (16, 512, 16),
        prefill_seq_lengths: Tuple[int, int, int] = (128, 10240, 128),
    ) -> dict:
        """Run full attention profiling.

        Returns:
            dict with 'decode' and 'prefill' keys containing latency tables.
        """
        decode_profile = self.profile_decode(
            batch_sizes=range(*decode_batch_sizes),
            seq_lengths=range(*decode_seq_lengths),
        )
        prefill_profile = self.profile_prefill(
            query_lengths=range(*prefill_query_lengths),
            seq_lengths=range(*prefill_seq_lengths),
        )
        return {"decode": decode_profile, "prefill": prefill_profile}

    def save(self, path: str) -> None:
        profile = self.run()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f)
