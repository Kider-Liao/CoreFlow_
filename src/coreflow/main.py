"""Main entry point for the resource allocation framework.

Usage:
    python -m coreflow.main --workflow reflection
    python -m coreflow.main --config custom_config.py
    python -m coreflow.main --profile  # Run profiling first
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

from coreflow.config import (
    WorkflowConfig,
    get_reflection_config,
    get_single_agent_config,
)


def build_config(args: argparse.Namespace) -> WorkflowConfig:
    """Build WorkflowConfig from CLI arguments."""
    if args.workflow == "reflection":
        cfg = get_reflection_config()
    elif args.workflow == "single_agent":
        cfg = get_single_agent_config()
    else:
        cfg = WorkflowConfig(workflow_name=args.workflow)

    # Apply CLI overrides
    if args.agents:
        cfg.agent = replace(cfg.agent, agents=args.agents)
    if args.total_gpus is not None:
        cfg.hardware = replace(cfg.hardware, total_num_gpus=args.total_gpus)
    if args.node_list:
        cfg.node_list_path = args.node_list
    if args.gpus_per_instance is not None:
        cfg.allocator = replace(
            cfg.allocator,
            gpus_per_instance=args.gpus_per_instance,
        )
    return cfg


def build_predictor(cfg: WorkflowConfig) -> "LatencyPredictor":
    """Create and return a configured LatencyPredictor."""
    from coreflow.predictor.latency import LatencyPredictor

    paths = cfg.paths
    return LatencyPredictor(
        attn_profile_path=str(paths.resolve(paths.attn_profile)),
        interference_profile_path=str(paths.resolve(paths.interference_profile)),
        mlp_profile_path=str(paths.resolve(paths.mlp_profile)),
    )


def build_allocator(
    cfg: WorkflowConfig,
    predictor: Optional["LatencyPredictor"] = None,
) -> "WorkflowAllocator":
    """Assemble the full allocation pipeline."""
    from coreflow.allocator.request_analyzer import RequestAnalyzer
    from coreflow.allocator.throughput_estimator import ThroughputEstimator
    from coreflow.allocator.instance_orchestrator import InstanceOrchestrator
    from coreflow.allocator.agent_allocator import AgentAllocator
    from coreflow.allocator.workflow_allocator import WorkflowAllocator

    paths = cfg.paths

    if predictor is None:
        predictor = build_predictor(cfg)

    node_list_path = cfg.get_node_list_path()
    analyzer = RequestAnalyzer(str(paths.resolve(node_list_path)))

    estimator = ThroughputEstimator(
        latency_predictor=predictor,
        request_analyzer=analyzer,
        chunk_size=cfg.allocator.chunk_size,
        cache_swap_latency=cfg.allocator.cache_swap_latency,
        context_step=cfg.allocator.context_step,
        max_context_length=cfg.allocator.max_context_length,
    )

    orchestrator = InstanceOrchestrator(
        context_step=cfg.allocator.context_step,
        max_context_length=cfg.allocator.max_context_length,
        dp_gpu_step=cfg.allocator.gpus_per_instance,
    )

    agent_alloc = AgentAllocator(
        throughput_estimator=estimator,
        instance_orchestrator=orchestrator,
    )

    return WorkflowAllocator(
        agent_allocator=agent_alloc,
        request_analyzer=analyzer,
        max_iterations=cfg.allocator.max_iterations,
    )


def run_allocation(cfg: WorkflowConfig, allocator: "WorkflowAllocator", verbose: bool = False) -> None:
    """Execute the allocation and print results."""
    print(f"\n{'='*60}")
    print(f"  Resource Allocation: {cfg.workflow_name}")
    print(f"{'='*60}")
    print(f"  Agents:        {cfg.agent.agents}")
    print(f"  Total GPUs:    {cfg.hardware.total_num_gpus}")
    print(f"  Model layers:  {cfg.model.num_layers}")
    print(f"  Instance KV:   {cfg.agent.instance_cached_tokens:,} tokens")
    print(f"  Interference:  {cfg.agent.interference}")
    print(f"  KV Reuse:      {cfg.agent.reuse}")
    print(f"{'='*60}\n")

    result = allocator.allocate(
        agents=cfg.agent.agents,
        total_num_gpus=cfg.hardware.total_num_gpus,
        num_layers=cfg.model.num_layers,
        instance_cached_tokens=cfg.agent.instance_cached_tokens,
        dp_gpu_step=cfg.allocator.gpus_per_instance,
        max_instance_group=cfg.agent.max_instance_group,
        min_instance_group=cfg.agent.min_instance_group,
        interference=cfg.agent.interference,
        reuse=cfg.agent.reuse,
        verbose=cfg.workflow_name == "reflection",
    )

    # --- Print results ---
    print("--- Agent Invocations (avg) ---")
    for agent, count in result.agent_invocations.items():
        print(f"  {agent}: {count:.2f}")

    print("\n--- GPU Allocation ---")
    for agent, gpus in result.gpu_allocation.items():
        tp = result.agent_throughput.get(agent, 0.0)
        print(f"  {agent}: {gpus} GPUs  (normalized throughput: {tp:.2f} req/s)")

    print(f"\n  Min (bottleneck) throughput: {result.min_throughput:.2f} req/s")
    print(f"  Converged in {result.iterations} iterations")

    # --- Print per-agent instance / context partitioning ---
    gpus_per_instance = cfg.allocator.gpus_per_instance
    print(f"\n--- Per-Agent Instance & Context Partitioning (gpus/instance={gpus_per_instance}) ---")
    for agent in cfg.agent.agents:
        assignment = result.dp_results[agent][result.gpu_allocation[agent]]
        print(f"\n  [{agent}]  {result.gpu_allocation[agent]} GPUs  |  throughput: {assignment.throughput:.2f} req/s")
        for g, lo, hi, inst_tp in assignment.instances:
            n_instances = g // gpus_per_instance
            print(f"    └─ {n_instances}× instance ({g} GPU): context [{lo:>7,}, {hi:>7,})  @ {inst_tp:.3f} req/s")

    # --- Save to JSON ---
    output = {
        "config": {
            "agents": cfg.agent.agents,
            "total_gpus": cfg.hardware.total_num_gpus,
            "num_layers": cfg.model.num_layers,
            "instance_cached_tokens": cfg.agent.instance_cached_tokens,
            "gpus_per_instance": gpus_per_instance,
            "interference": cfg.agent.interference,
            "reuse": cfg.agent.reuse,
        },
        "allocation": {
            agent: {
                "gpus": result.gpu_allocation[agent],
                "throughput": result.agent_throughput[agent],
                "invocations": result.agent_invocations[agent],
                "instances": [
                    {
                        "gpus": g,
                        "gpus_per_instance": gpus_per_instance,
                        "num_instances": g // gpus_per_instance,
                        "context_lower": lo,
                        "context_upper": hi,
                        "throughput": inst_tp,
                    }
                    for g, lo, hi, inst_tp in result.dp_results[agent][
                        result.gpu_allocation[agent]
                    ].instances
                ],
            }
            for agent in cfg.agent.agents
        },
        "min_throughput": result.min_throughput,
        "iterations": result.iterations,
    }
    output_path = cfg.paths.resolve("allocation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


def run_profiling(cfg: WorkflowConfig, verify: bool = False) -> None:
    """Run GPU profiling (requires CUDA)."""
    from coreflow.profiler.unified import UnifiedProfiler

    profiler = UnifiedProfiler(
        num_heads=cfg.model.num_heads,
        num_kv_heads=cfg.model.num_kv_heads,
        head_size=cfg.model.head_size,
        device=cfg.hardware.device,
        dtype=cfg.hardware.dtype,
        warmup=cfg.profiler.warmup,
        repeat=cfg.profiler.repeat,
    )

    print("Running GPU profiling (this may take a while)...")
    attn_path = str(cfg.paths.resolve(cfg.paths.attn_profile))
    inter_path = str(cfg.paths.resolve(cfg.paths.interference_profile))
    mlp_path = str(cfg.paths.resolve(cfg.paths.mlp_profile))

    profiler.save_all(attn_path, inter_path, mlp_path)
    print(f"  Attention profile saved to:    {attn_path}")
    print(f"  Interference profile saved to:  {inter_path}")
    print(f"  MLP profile saved to:           {mlp_path}")

    if verify:
        print("\nVerifying model accuracy...")
        profiler.run_verification(attn_path, inter_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource Allocation for Agentic Workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m coreflow.main --workflow reflection
  python -m coreflow.main --workflow single_agent
  python -m coreflow.main --agents Reflector System Search --total-gpus 32
  python -m coreflow.main --profile
  python -m coreflow.main --profile --verify
        """,
    )
    parser.add_argument(
        "--workflow", type=str, default="default",
        choices=["default", "reflection", "single_agent"],
        help="Pre-defined workflow configuration.",
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=None,
        help="List of agent names (overrides config).",
    )
    parser.add_argument(
        "--total-gpus", type=int, default=None,
        help="Total GPU count (overrides config).",
    )
    parser.add_argument(
        "--node-list", type=str, default=None,
        help="Path to node_list.json (overrides config).",
    )
    parser.add_argument(
        "--gpus-per-instance", type=int, default=None,
        help="GPUs per vLLM instance / tensor-parallel size (default: 1).",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Run GPU profiling before allocation (requires CUDA).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run model accuracy verification (only with --profile).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed allocation information.",
    )

    args = parser.parse_args()

    cfg = build_config(args)

    if args.profile:
        run_profiling(cfg, verify=args.verify)
        if not args.workflow:
            return  # Profile-only mode

    # Run allocation
    allocator = build_allocator(cfg)
    run_allocation(cfg, allocator, verbose=args.verbose)


if __name__ == "__main__":
    main()
