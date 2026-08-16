"""Run CoreFlow GPU profiling and save profile JSON files.

Usage:
    python scripts/model_profile.py --device cuda:0 --output-dir Profiler
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dir

from coreflow.profiler.unified import UnifiedProfiler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile attention, interference, and MLP kernels."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--output-dir", default="Profiler")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    attn_path = output_dir / "attn_profile.json"
    interference_path = output_dir / "interference_profile.json"
    mlp_path = output_dir / "mlp_profile.json"

    profiler = UnifiedProfiler(
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        intermediate_size=args.intermediate_size,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print("Running CoreFlow GPU profiling...")
    profiler.save_all(
        attn_path=str(attn_path),
        interference_path=str(interference_path),
        mlp_path=str(mlp_path),
    )
    print(f"Attention profile:      {attn_path}")
    print(f"Interference profile:   {interference_path}")
    print(f"MLP profile:            {mlp_path}")

    if args.verify:
        result = profiler.run_verification(
            attn_path=str(attn_path),
            interference_path=str(interference_path),
        )
        print(f"Verification result: {result}")


if __name__ == "__main__":
    main()
