"""T1: measure long-context decode interference with FlashAttention.

This script is the artifact entry point derived from
``test_interference_long_context_len.py``.  It measures two latency curves:

* ``1L+16S``: one long decode request plus sixteen short decode requests run
  separately, then summed.
* ``1L&16S``: one long decode request and sixteen short decode requests run in
  the same decode batch.

The current implementation supports the Llama-3-8B shape with FlashAttention.
The CLI still accepts ``--model`` and ``--kernel`` so additional model/kernel
configs can be added without changing the user-facing command.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from statistics import median
from typing import Callable

from common import REPO_ROOT, ensure_dir, write_csv, write_json


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    dtype: str = "bfloat16"

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_size

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def scale(self) -> float:
        return self.head_size**-0.5


@dataclass(frozen=True)
class KernelConfig:
    name: str
    block_size: int = 16
    total_num_blocks: int = 100_000


@dataclass(frozen=True)
class InterferenceConfig:
    model: ModelConfig
    kernel: KernelConfig
    device: str
    warmup: int
    repeat: int
    num_large_seqs: int
    num_small_seqs: int
    small_seq_len: int
    large_seq_lens: tuple[int, ...]
    seed: int


SUPPORTED_MODELS = {
    "llama3-8b": ModelConfig(
        name="Llama-3-8B",
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
    ),
    "llama-3-8b": ModelConfig(
        name="Llama-3-8B",
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
    ),
    "llama-3-1-8b": ModelConfig(
        name="Llama-3-8B",
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama-3-8B",
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
    ),
}

SUPPORTED_KERNELS = {
    "flashattn": KernelConfig(name="FlashAttention"),
    "flash-attn": KernelConfig(name="FlashAttention"),
    "flashattention": KernelConfig(name="FlashAttention"),
    "flash_attention": KernelConfig(name="FlashAttention"),
}


def _normalize(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "")


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not lengths:
        raise argparse.ArgumentTypeError("at least one length is required")
    if any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return lengths


def resolve_model(name: str) -> ModelConfig:
    key = _normalize(name)
    if key not in SUPPORTED_MODELS:
        supported = ", ".join(sorted({cfg.name for cfg in SUPPORTED_MODELS.values()}))
        raise ValueError(f"unsupported model {name!r}; currently supported: {supported}")
    return SUPPORTED_MODELS[key]


def resolve_kernel(name: str, total_num_blocks: int) -> KernelConfig:
    key = _normalize(name)
    if key not in SUPPORTED_KERNELS:
        supported = ", ".join(sorted({cfg.name for cfg in SUPPORTED_KERNELS.values()}))
        raise ValueError(f"unsupported kernel {name!r}; currently supported: {supported}")
    base = SUPPORTED_KERNELS[key]
    return KernelConfig(
        name=base.name,
        block_size=base.block_size,
        total_num_blocks=total_num_blocks,
    )


class FlashAttentionInterferenceRunner:
    def __init__(self, config: InterferenceConfig) -> None:
        self.config = config

        import torch

        self.torch = torch
        self.ops = self._import_vllm_ops()
        self.flash_attn_with_kvcache = self._import_flash_attn_with_kvcache()
        self.dtype = getattr(torch, config.model.dtype)
        self.k_scale = None
        self.v_scale = None
        self.kv_cache: tuple | None = None

    @staticmethod
    def _drop_local_vllm_from_imports() -> None:
        local_vllm = str(REPO_ROOT / "src" / "vllm")
        sys.path[:] = [path for path in sys.path if path != local_vllm]
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                del sys.modules[name]

    @classmethod
    def _import_vllm_ops(cls):
        """Import vLLM custom ops in both coreflow and wheel-based envs."""
        try:
            from vllm import _custom_ops as ops

            return ops
        except Exception as first_error:
            cls._drop_local_vllm_from_imports()
            try:
                from vllm import _custom_ops as ops

                return ops
            except Exception as second_error:
                raise RuntimeError(
                    "failed to import vLLM custom ops from the current "
                    "environment or an installed vLLM package"
                ) from second_error

    @staticmethod
    def _import_flash_attn_with_kvcache():
        """Import FlashAttention from either bundled vLLM or top-level package."""
        try:
            from vllm.vllm_flash_attn import flash_attn_with_kvcache

            return flash_attn_with_kvcache
        except Exception:
            try:
                from vllm_flash_attn import flash_attn_with_kvcache

                return flash_attn_with_kvcache
            except Exception as exc:
                raise RuntimeError(
                    "failed to import flash_attn_with_kvcache; expected either "
                    "vllm.vllm_flash_attn or top-level vllm_flash_attn"
                ) from exc

    def _init_cuda(self) -> None:
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the FlashAttention T1 benchmark, but "
                f"torch.cuda.is_available() is False "
                f"(torch={torch.__version__}, cuda={torch.version.cuda}, "
                f"device_count={torch.cuda.device_count()})"
            )
        torch.cuda.set_device(self.config.device)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        self.k_scale = torch.tensor(1.0, device=self.config.device, dtype=torch.float32)
        self.v_scale = torch.tensor(1.0, device=self.config.device, dtype=torch.float32)

    def _init_kv_cache(self) -> None:
        if self.kv_cache is not None:
            return

        cfg = self.config
        model = cfg.model
        kernel = cfg.kernel
        torch = self.torch
        self.kv_cache = (
            torch.randn(
                (
                    kernel.total_num_blocks,
                    kernel.block_size,
                    model.num_kv_heads,
                    model.head_size,
                ),
                device=cfg.device,
                dtype=self.dtype,
            ),
            torch.randn(
                (
                    kernel.total_num_blocks,
                    kernel.block_size,
                    model.num_kv_heads,
                    model.head_size,
                ),
                device=cfg.device,
                dtype=self.dtype,
            ),
        )

    def _forward(self, seq_lengths: list[int]) -> None:
        cfg = self.config
        model = cfg.model
        kernel = cfg.kernel
        torch = self.torch

        batch_size = len(seq_lengths)
        max_seq_len = max(seq_lengths)
        num_blocks = int((max_seq_len - 1) // kernel.block_size + 1)

        slot_mapping = torch.randint(
            0,
            kernel.total_num_blocks,
            (batch_size,),
            device=cfg.device,
            dtype=torch.long,
        )
        block_tables = torch.randint(
            0,
            kernel.total_num_blocks,
            (batch_size, num_blocks),
            device=cfg.device,
            dtype=torch.int32,
        )
        seq_lens_tensor = torch.tensor(seq_lengths, device=cfg.device, dtype=torch.int32)
        query = torch.randn(
            (batch_size, model.hidden_size),
            device=cfg.device,
            dtype=self.dtype,
        )
        key = torch.randn(
            (batch_size, model.kv_size),
            device=cfg.device,
            dtype=self.dtype,
        )
        value = torch.randn(
            (batch_size, model.kv_size),
            device=cfg.device,
            dtype=self.dtype,
        )

        query = query.view(-1, model.num_heads, model.head_size)
        key = key.view(-1, model.num_kv_heads, model.head_size)
        value = value.view(-1, model.num_kv_heads, model.head_size)
        key_cache, value_cache = self.kv_cache

        self.ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            "auto",
            self.k_scale,
            self.v_scale,
        )

        self.flash_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            block_table=block_tables,
            cache_seqlens=seq_lens_tensor,
            softmax_scale=model.scale,
            causal=True,
            alibi_slopes=None,
            softcap=0.0,
        )

    def _measure_latency(self, forward: Callable[[], None]) -> float:
        torch = self.torch
        for _ in range(self.config.warmup):
            forward()

        latencies = []
        for _ in range(self.config.repeat):
            torch.cuda.synchronize(device=self.config.device)
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize(device=self.config.device)
            latencies.append(time.perf_counter() - start)

        return float(median(latencies))

    def run(self) -> list[dict]:
        self._init_cuda()
        self._init_kv_cache()

        cfg = self.config
        small_lengths = [cfg.small_seq_len] * cfg.num_small_seqs
        small_latency_s = self._measure_latency(lambda: self._forward(small_lengths))

        rows = []
        for large_seq_len in cfg.large_seq_lens:
            large_lengths = [large_seq_len] * cfg.num_large_seqs
            mixed_lengths = large_lengths + small_lengths

            large_latency_s = self._measure_latency(lambda: self._forward(large_lengths))
            mixed_latency_s = self._measure_latency(lambda: self._forward(mixed_lengths))
            separate_latency_s = large_latency_s + small_latency_s

            for label, latency_s in [
                (f"{cfg.num_large_seqs}L+{cfg.num_small_seqs}S", separate_latency_s),
                (f"{cfg.num_large_seqs}L&{cfg.num_small_seqs}S", mixed_latency_s),
            ]:
                rows.append({
                    "model": cfg.model.name,
                    "kernel": cfg.kernel.name,
                    "series": label,
                    "large_context_length": large_seq_len,
                    "large_context_length_k": large_seq_len // 1000,
                    "small_context_length": cfg.small_seq_len,
                    "num_large_seqs": cfg.num_large_seqs,
                    "num_small_seqs": cfg.num_small_seqs,
                    "latency_s": latency_s,
                    "latency_ms": latency_s * 1000.0,
                    "large_only_latency_s": large_latency_s,
                    "small_only_latency_s": small_latency_s,
                    "mixed_latency_s": mixed_latency_s,
                })
            print(
                f"L={large_seq_len}: "
                f"{cfg.num_large_seqs}L+{cfg.num_small_seqs}S={separate_latency_s * 1000.0:.3f} ms, "
                f"{cfg.num_large_seqs}L&{cfg.num_small_seqs}S={mixed_latency_s * 1000.0:.3f} ms"
            )

        return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure FlashAttention long-context decode interference."
    )
    parser.add_argument(
        "--node-list",
        default=None,
        help="Accepted for compatibility; T1 uses synthetic kernel inputs.",
    )
    parser.add_argument("--out-dir", default="results/t1_interference")
    parser.add_argument("--kernel", default="FlashAttention")
    parser.add_argument("--model", default="Llama-3-8B")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--num-large-seqs", type=int, default=1)
    parser.add_argument("--num-small-seqs", type=int, default=16)
    parser.add_argument("--small-seq-len", type=int, default=2000)
    parser.add_argument(
        "--large-seq-lens",
        type=_parse_lengths,
        default=(4000, 8000, 16000, 32000, 64000, 128000),
        help="Comma-separated long context lengths, e.g. 4000,8000,16000.",
    )
    parser.add_argument("--total-num-blocks", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        model = resolve_model(args.model)
        kernel = resolve_kernel(args.kernel, args.total_num_blocks)
    except ValueError as exc:
        parser.error(str(exc))

    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    if args.num_large_seqs <= 0 or args.num_small_seqs <= 0:
        parser.error("--num-large-seqs and --num-small-seqs must be > 0")
    if args.small_seq_len <= 0:
        parser.error("--small-seq-len must be > 0")

    max_required_blocks = max(args.large_seq_lens) // kernel.block_size + 1
    if args.total_num_blocks < max_required_blocks:
        parser.error(
            "--total-num-blocks must cover the largest sequence length "
            f"({max(args.large_seq_lens)} requires at least {max_required_blocks})"
        )

    config = InterferenceConfig(
        model=model,
        kernel=kernel,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        num_large_seqs=args.num_large_seqs,
        num_small_seqs=args.num_small_seqs,
        small_seq_len=args.small_seq_len,
        large_seq_lens=args.large_seq_lens,
        seed=args.seed,
    )

    out_dir = ensure_dir(args.out_dir)
    runner = FlashAttentionInterferenceRunner(config)
    rows = runner.run()

    write_csv(
        out_dir / "interference_latency.csv",
        rows,
        [
            "model",
            "kernel",
            "series",
            "large_context_length",
            "large_context_length_k",
            "small_context_length",
            "num_large_seqs",
            "num_small_seqs",
            "latency_s",
            "latency_ms",
            "large_only_latency_s",
            "small_only_latency_s",
            "mixed_latency_s",
        ],
    )
    write_json(out_dir / "metadata.json", {
        "model": model.name,
        "kernel": kernel.name,
        "device": args.device,
        "dtype": model.dtype,
        "num_heads": model.num_heads,
        "num_kv_heads": model.num_kv_heads,
        "head_size": model.head_size,
        "block_size": kernel.block_size,
        "total_num_blocks": kernel.total_num_blocks,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "num_large_seqs": args.num_large_seqs,
        "num_small_seqs": args.num_small_seqs,
        "small_seq_len": args.small_seq_len,
        "large_seq_lens": list(args.large_seq_lens),
    })
    print(f"Wrote T1 interference analysis to {out_dir}")


if __name__ == "__main__":
    main()
