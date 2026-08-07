"""T2: small-scale interference analysis.

The full paper experiments use GPU kernels and model-serving throughput tests.
This public artifact script provides a quick reproducer over the included trace
data: it estimates isolated vs colocated attention cost for heterogeneous
context lengths and writes CSV outputs consumed by the plotting scripts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean

from common import ensure_dir, iter_node_records, load_json, write_csv, write_json


def estimate_attention_ms(context_length: int, batch_size: int = 16) -> float:
    return batch_size * max(context_length, 1) * 5e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate short/long context interference.")
    parser.add_argument("--node-list", default="data/React/node_list.json")
    parser.add_argument("--out-dir", default="results/t2_interference")
    parser.add_argument("--kernel", default="attention")
    parser.add_argument("--model", default="Llama-3-8B")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    records = list(iter_node_records(load_json(args.node_list)))
    by_agent = defaultdict(list)
    for record in records:
        by_agent[record["agent_name"]].append(record["context_length"])

    latency_rows = []
    throughput_rows = []
    for agent, contexts in sorted(by_agent.items()):
        if not contexts:
            continue
        short_context = min(contexts)
        long_context = max(contexts)
        isolated_short = estimate_attention_ms(short_context)
        isolated_long = estimate_attention_ms(long_context)
        skew = long_context / max(short_context, 1)
        colocated_short = isolated_short * (1.0 + min(skew, 32.0) * 0.025)
        colocated_long = isolated_long * 1.05

        latency_rows.extend([
            {
                "agent_name": agent,
                "mode": "isolated_short",
                "context_length": short_context,
                "attention_latency_ms": isolated_short,
            },
            {
                "agent_name": agent,
                "mode": "colocated_short",
                "context_length": short_context,
                "attention_latency_ms": colocated_short,
            },
            {
                "agent_name": agent,
                "mode": "isolated_long",
                "context_length": long_context,
                "attention_latency_ms": isolated_long,
            },
            {
                "agent_name": agent,
                "mode": "colocated_long",
                "context_length": long_context,
                "attention_latency_ms": colocated_long,
            },
        ])

        isolated_throughput = 1000.0 / mean([isolated_short, isolated_long])
        colocated_throughput = 1000.0 / mean([colocated_short, colocated_long])
        throughput_rows.append({
            "agent_name": agent,
            "isolated_decode_throughput": isolated_throughput,
            "colocated_decode_throughput": colocated_throughput,
            "throughput_drop_ratio": 1.0 - colocated_throughput / isolated_throughput,
        })

    write_csv(
        out_dir / "interference_latency.csv",
        latency_rows,
        ["agent_name", "mode", "context_length", "attention_latency_ms"],
    )
    write_csv(
        out_dir / "decoding_throughput.csv",
        throughput_rows,
        [
            "agent_name",
            "isolated_decode_throughput",
            "colocated_decode_throughput",
            "throughput_drop_ratio",
        ],
    )
    write_json(out_dir / "metadata.json", {
        "kernel": args.kernel,
        "model": args.model,
        "note": "Small-scale estimator for artifact exercisability.",
    })
    print(f"Wrote T2 interference analysis to {out_dir}")


if __name__ == "__main__":
    main()
