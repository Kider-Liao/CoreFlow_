"""Generate ``allocation.json`` from profiles and workflow data.

Usage:
    python scripts/allocate.py --gpus 8 --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import add_repo_to_path, ensure_dir, write_json

add_repo_to_path()

from coreflow.allocator.agent_allocator import AgentAllocator  # noqa: E402
from coreflow.allocator.instance_orchestrator import InstanceOrchestrator  # noqa: E402
from coreflow.allocator.request_analyzer import RequestAnalyzer  # noqa: E402
from coreflow.allocator.throughput_estimator import ThroughputEstimator  # noqa: E402
from coreflow.allocator.workflow_allocator import WorkflowAllocator  # noqa: E402
from coreflow.predictor.latency import LatencyPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CoreFlow resource allocation and write allocation.json."
    )
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--node-list", default=None)
    parser.add_argument("--profile-dir", default="Profiler")
    parser.add_argument("--agents", nargs="+", default=None)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--instance-cached-tokens", type=int, default=216_000)
    parser.add_argument("--gpus-per-instance", type=int, default=1)
    parser.add_argument("--max-instance-group", type=int, default=3)
    parser.add_argument("--min-instance-group", type=int, default=1)
    parser.add_argument("--context-step", type=int, default=2000)
    parser.add_argument("--max-context-length", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--cache-swap-latency", type=float, default=0.1)
    parser.add_argument("--interference", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output", default="allocation.json")
    args = parser.parse_args()

    if args.gpus_per_instance <= 0:
        parser.error("--gpus-per-instance must be positive")
    if args.gpus % args.gpus_per_instance != 0:
        parser.error(
            "--gpus must be divisible by --gpus-per-instance"
        )

    data_dir = Path(args.data_dir)
    node_list_path = args.node_list or str(data_dir / "node_list.json")
    if not Path(node_list_path).exists():
        parser.error(f"node list not found: {node_list_path}")

    profile_dir = Path(args.profile_dir)
    attn_profile = profile_dir / "attn_profile.json"
    interference_profile = profile_dir / "interference_profile.json"
    mlp_profile = profile_dir / "mlp_profile.json"

    for path in (attn_profile, interference_profile, mlp_profile):
        if not path.exists():
            parser.error(f"missing profile file: {path}")

    predictor = LatencyPredictor(
        attn_profile_path=str(attn_profile),
        interference_profile_path=str(interference_profile),
        mlp_profile_path=str(mlp_profile),
    )
    analyzer = RequestAnalyzer(str(node_list_path))
    agents = args.agents or analyzer.agents
    if not agents:
        parser.error("no agents found in node list")

    estimator = ThroughputEstimator(
        latency_predictor=predictor,
        request_analyzer=analyzer,
        chunk_size=args.chunk_size,
        cache_swap_latency=args.cache_swap_latency,
        context_step=args.context_step,
        max_context_length=args.max_context_length,
    )
    orchestrator = InstanceOrchestrator(
        context_step=args.context_step,
        max_context_length=args.max_context_length,
        dp_gpu_step=args.gpus_per_instance,
    )
    agent_allocator = AgentAllocator(
        throughput_estimator=estimator,
        instance_orchestrator=orchestrator,
    )
    workflow_allocator = WorkflowAllocator(
        agent_allocator=agent_allocator,
        request_analyzer=analyzer,
    )

    result = workflow_allocator.allocate(
        agents=agents,
        total_num_gpus=args.gpus,
        num_layers=args.num_layers,
        instance_cached_tokens=args.instance_cached_tokens,
        dp_gpu_step=args.gpus_per_instance,
        max_instance_group=args.max_instance_group,
        min_instance_group=args.min_instance_group,
        interference=args.interference,
        reuse=args.reuse,
        verbose=True,
    )

    output = {
        "config": {
            "agents": agents,
            "total_gpus": args.gpus,
            "num_layers": args.num_layers,
            "instance_cached_tokens": args.instance_cached_tokens,
            "gpus_per_instance": args.gpus_per_instance,
            "interference": args.interference,
            "reuse": args.reuse,
        },
        "allocation": {
            agent: {
                "gpus": result.gpu_allocation[agent],
                "throughput": result.agent_throughput[agent],
                "invocations": result.agent_invocations[agent],
                "instances": [
                    {
                        "gpus": gpus,
                        "gpus_per_instance": args.gpus_per_instance,
                        "num_instances": gpus // args.gpus_per_instance,
                        "context_lower": lower,
                        "context_upper": upper,
                        "throughput": throughput,
                    }
                    for gpus, lower, upper, throughput in
                    result.dp_results[agent][
                        result.gpu_allocation[agent]
                    ].instances
                ],
            }
            for agent in agents
        },
        "min_throughput": result.min_throughput,
        "iterations": result.iterations,
    }

    output_path = ensure_dir(Path(args.output).parent) / Path(args.output).name
    write_json(output_path, output)
    print(f"Wrote allocation to {output_path}")


if __name__ == "__main__":
    main()
