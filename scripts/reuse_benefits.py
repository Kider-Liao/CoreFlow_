"""T3: quantify throughput benefit from cache reuse.

This lightweight script sweeps prompt lengths and reuse ratios, producing the
CSV consumed by ``plot_reuse_throughput.py``.  It is intended as a fast artifact
exercise path; full paper-scale experiments can replace the estimator with
measured serving throughput.
"""

from __future__ import annotations

import argparse

from common import ensure_dir, write_csv, write_json


def estimate_throughput(prompt_len: int, reuse_ratio: float, output_len: int) -> float:
    effective_prefill = prompt_len * (1.0 - reuse_ratio)
    decode_cost = output_len * 8.0
    total_cost = max(effective_prefill + decode_cost, 1.0)
    return 200000.0 / total_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate throughput gains from KV reuse.")
    parser.add_argument("--out-dir", default="results/t3_reuse")
    parser.add_argument("--output-len", type=int, default=256)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    prompt_lengths = [1024, 4096, 16384, 32768]
    reuse_ratios = [0.0, 0.25, 0.5, 0.75]

    rows = []
    for prompt_len in prompt_lengths:
        baseline = estimate_throughput(prompt_len, 0.0, args.output_len)
        for reuse_ratio in reuse_ratios:
            throughput = estimate_throughput(prompt_len, reuse_ratio, args.output_len)
            rows.append({
                "prompt_length": prompt_len,
                "reuse_ratio": reuse_ratio,
                "output_length": args.output_len,
                "throughput": throughput,
                "speedup_vs_no_reuse": throughput / baseline,
            })

    write_csv(
        out_dir / "reuse_throughput.csv",
        rows,
        [
            "prompt_length",
            "reuse_ratio",
            "output_length",
            "throughput",
            "speedup_vs_no_reuse",
        ],
    )
    write_json(out_dir / "metadata.json", {
        "note": "Small-scale estimator for cache-reuse benefit.",
        "output_length": args.output_len,
    })
    print(f"Wrote T3 reuse-benefit analysis to {out_dir}")


if __name__ == "__main__":
    main()
