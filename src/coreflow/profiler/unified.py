"""Unified profiler orchestrating attention, interference profiling and verification."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from coreflow.profiler.attention import AttentionProfiler
from coreflow.profiler.interference import InterferenceProfiler
from coreflow.profiler.base import BaseProfiler


class UnifiedProfiler:
    """Orchestrates all GPU profiling and verifies model accuracy.

    Usage::

        profiler = UnifiedProfiler(config)
        profiler.run_all()                         # Profile everything
        profiler.save_all(output_dir="profiles/")  # Save all results
        profiler.verify_attention()                # Check model accuracy
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_size: int = 128,
        total_num_blocks: int = 100_000,
        block_size: int = 16,
        device: str = "cuda:2",
        dtype: str = "bfloat16",
        warmup: int = 5,
        repeat: int = 10,
    ) -> None:
        import torch

        _dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self._attn = AttentionProfiler(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            total_num_blocks=total_num_blocks,
            block_size=block_size,
            device=device,
            dtype=_dtype,
            warmup=warmup,
            repeat=repeat,
        )
        self._interference = InterferenceProfiler(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            total_num_blocks=total_num_blocks,
            block_size=block_size,
            device=device,
            dtype=_dtype,
            warmup=warmup,
            repeat=repeat,
        )

        # Results storage
        self._attn_data: Optional[dict] = None
        self._interference_data: Optional[dict] = None
        self._mlp_data: Optional[dict] = None

    def run_all(self) -> dict:
        """Run all profilers and return combined results."""
        self._attn_data = self._attn.run()
        self._interference_data = self._interference.run()
        return {
            "attention": self._attn_data,
            "interference": self._interference_data,
        }

    def profile_mlp(self, batch_sizes: Optional[list[int]] = None) -> dict:
        if batch_sizes is None:
            batch_sizes = list(range(1, 65)) + [96, 128, 192, 256, 384, 512]

        hidden_size = self._attn.hidden_size
        intermediate_size = hidden_size * 4
        device = self._attn.device
        dtype = self._attn.dtype

        w1 = torch.randn(
            (hidden_size, intermediate_size), device=device, dtype=dtype,
        )
        w2 = torch.randn(
            (intermediate_size, hidden_size), device=device, dtype=dtype,
        )

        result: dict[str, float] = {}
        for batch_size in batch_sizes:
            x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)

            def forward():
                y = torch.nn.functional.gelu(x @ w1)
                y @ w2

            result[str(batch_size)] = self._attn._measure_latency(forward)

        return result

    def save_all(
        self,
        attn_path: str,
        interference_path: str,
        mlp_path: Optional[str] = None,
    ) -> None:
        """Save all profiling results to JSON files."""
        if self._attn_data is None:
            self._attn_data = self._attn.run()
        if self._interference_data is None:
            self._interference_data = self._interference.run()
        if mlp_path is not None and self._mlp_data is None:
            self._mlp_data = self.profile_mlp()

        for path in [attn_path, interference_path, mlp_path]:
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(attn_path, "w", encoding="utf-8") as f:
            json.dump(self._attn_data, f)
        with open(interference_path, "w", encoding="utf-8") as f:
            json.dump(self._interference_data, f)
        if mlp_path is not None:
            with open(mlp_path, "w", encoding="utf-8") as f:
                json.dump(self._mlp_data, f)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    @staticmethod
    def verify_attention(profile_path: str) -> Tuple[float, float]:
        """Verify the accuracy of linear regression models on attention data.

        Args:
            profile_path: Path to attn_profile.json.

        Returns:
            Tuple of (decode_relative_error, prefill_relative_error).
        """
        with open(profile_path, "r") as f:
            data = json.load(f)

        def _fit_error(raw: dict) -> float:
            xs, ys = [], []
            for batch_size, seq_data in raw.items():
                for seq_len, latency in seq_data.items():
                    xs.append([int(batch_size), int(seq_len)])
                    ys.append(float(latency))
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            model = LinearRegression()
            model.fit(xs_arr, ys_arr)
            pred = model.predict(xs_arr)
            return float(np.mean(np.abs(pred - ys_arr) / ys_arr))

        decode_err = _fit_error(data["decode"])
        prefill_err = _fit_error(data["prefill"])
        return decode_err, prefill_err

    @staticmethod
    def verify_interference(profile_path: str, min_batch: int = 11) -> float:
        """Verify the accuracy of linear models on interference data.

        Args:
            profile_path: Path to interference_profile.json.
            min_batch: Minimum batch size to include (smaller sizes have sparse data).

        Returns:
            Mean relative error across all batch sizes.
        """
        with open(profile_path, "r") as f:
            data = json.load(f)

        errors: list[float] = []

        for batch_size_str, batch_data in data.items():
            if int(batch_size_str) < min_batch:
                continue

            xs, ys = [], []
            for large_seq_len, seq_data in batch_data.items():
                for small_seq_len, latency in seq_data.items():
                    val = float(latency)
                    if val == 0:
                        continue
                    xs.append([int(large_seq_len), int(small_seq_len)])
                    ys.append(val)

            if not xs:
                continue

            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            model = LinearRegression()
            model.fit(xs_arr, ys_arr)
            pred = model.predict(xs_arr)
            err = float(np.mean(np.abs(pred - ys_arr) / ys_arr))
            errors.append(err)

        return float(np.mean(errors)) if errors else 0.0

    def run_verification(
        self, attn_path: str, interference_path: str
    ) -> dict:
        """Run full verification and print results.

        Returns:
            dict with verification metrics.
        """
        decode_err, prefill_err = self.verify_attention(attn_path)
        inter_err = self.verify_interference(interference_path)

        print(f"Attention decode  relative error: {decode_err:.4f}")
        print(f"Attention prefill relative error: {prefill_err:.4f}")
        print(f"Interference mean relative error: {inter_err:.4f}")

        return {
            "decode_error": decode_err,
            "prefill_error": prefill_err,
            "interference_error": inter_err,
        }
