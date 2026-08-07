"""Workflow-level allocator: iteratively balances GPUs across multiple agents."""

from dataclasses import dataclass, field
from math import floor
from typing import Dict, List, Optional, Tuple

import numpy as np

from coreflow.allocator.agent_allocator import AgentAllocator
from coreflow.allocator.instance_orchestrator import InstanceAssignment
from coreflow.allocator.request_analyzer import RequestAnalyzer


@dataclass
class WorkflowAllocation:
    """Complete allocation result for a multi-agent workflow."""

    # Per-agent GPU allocation
    gpu_allocation: Dict[str, int] = field(default_factory=dict)

    # Per-agent normalized throughput (requests/sec adjusted for invocations)
    agent_throughput: Dict[str, float] = field(default_factory=dict)

    # Per-agent allocation results (GPU -> InstanceAssignment)
    dp_results: Dict[str, Dict[int, InstanceAssignment]] = field(default_factory=dict)

    # Min throughput across agents (the bottleneck)
    min_throughput: float = 0.0

    # Agent invocations per workflow trace
    agent_invocations: Dict[str, float] = field(default_factory=dict)

    # Number of iterations taken to converge
    iterations: int = 0


class WorkflowAllocator:
    """Top-level allocator for multi-agent workflow GPU distribution.

    Uses iterative rebalancing: starts with an invocation-proportional GPU
    distribution, then repeatedly transfers GPUs from high-throughput to
    low-throughput agents until the min-throughput across agents is maximized.
    """

    def __init__(
        self,
        agent_allocator: AgentAllocator,
        request_analyzer: RequestAnalyzer,
        max_iterations: int = 100,
    ) -> None:
        self._agent_allocator = agent_allocator
        self._analyzer = request_analyzer
        self._max_iterations = max_iterations

    def allocate(
        self,
        agents: List[str],
        total_num_gpus: int,
        num_layers: int,
        instance_cached_tokens: int,
        dp_gpu_step: int = 1,
        max_instance_group: int = 3,
        min_instance_group: int = 1,
        interference: bool = False,
        reuse: bool = False,
        initial_gpus_per_agent: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> WorkflowAllocation:
        """Run the iterative GPU rebalancing algorithm.

        Args:
            agents: Ordered list of agent names.
            total_num_gpus: Total GPU budget.
            num_layers: Transformer layers per model.
            instance_cached_tokens: KV cache capacity per instance.
            dp_gpu_step: GPU granularity for allocation (gpus_per_instance).
                All GPU adjustments are made in multiples of this value.
            max_instance_group: Max instance groups per agent.
            min_instance_group: Min instance groups per agent.
            interference: Account for interference.
            reuse: Enable KV cache reuse.
            initial_gpus_per_agent: Starting GPU allocation. If omitted, the
                allocator initializes proportionally to invocation frequency.
            verbose: Print per-iteration progress.

        Returns:
            WorkflowAllocation with final GPU distribution and throughputs.
        """
        G = dp_gpu_step  # shorthand for readability

        # Compute agent invocation counts
        agent_invocations = self._analyzer.avg_num_invocations(agents)

        # Build the per-agent throughput/allocation curve once.
        dp_results: Dict[str, Dict[int, InstanceAssignment]] = {}
        for agent in agents:
            dp_results[agent] = self._agent_allocator.allocate(
                num_layers=num_layers,
                total_num_gpus=total_num_gpus,
                instance_cached_tokens=instance_cached_tokens,
                agent_name=agent,
                max_instance_group=max_instance_group,
                min_instance_group=min_instance_group,
                interference=interference,
                reuse=reuse,
            )

        # Initialize GPU distribution respecting dp_gpu_step granularity
        if initial_gpus_per_agent is None:
            gpus = self._initial_proportional_allocation(
                agents=agents,
                total_num_gpus=total_num_gpus,
                gpu_step=G,
                min_instance_group=min_instance_group,
                agent_invocations=agent_invocations,
            )
        else:
            gpus = list(initial_gpus_per_agent)

        # Iterative rebalancing (in steps of G)
        for iteration in range(self._max_iterations):
            # Compute normalized throughput for current allocation
            throughputs = [
                dp_results[agent][gpus[i]].throughput / agent_invocations[agent]
                for i, agent in enumerate(agents)
            ]
            if verbose:
                print(f"Iter {iteration}: gpus={gpus}, throughputs={[f'{t:.2f}' for t in throughputs]}")

            min_tp = min(throughputs)
            min_idx = int(np.argmin(throughputs))

            # Try to improve by taking G GPUs from a high-throughput agent
            candidate = self._find_donor(
                agents, dp_results, gpus, min_tp, agent_invocations, G
            )
            if candidate == -1:
                break  # Converged

            gpus[min_idx] += G
            gpus[candidate] -= G

        # Build final result
        final_throughputs: Dict[str, float] = {}
        for i, agent in enumerate(agents):
            tp = dp_results[agent][gpus[i]].throughput
            final_throughputs[agent] = tp / agent_invocations[agent]

        return WorkflowAllocation(
            gpu_allocation=dict(zip(agents, gpus)),
            agent_throughput=final_throughputs,
            dp_results=dp_results,
            min_throughput=min(final_throughputs.values()),
            agent_invocations=agent_invocations,
            iterations=iteration + 1,
        )

    @staticmethod
    def _initial_proportional_allocation(
        agents: List[str],
        total_num_gpus: int,
        gpu_step: int,
        min_instance_group: int,
        agent_invocations: Dict[str, float],
    ) -> List[int]:
        """Initialize per-agent GPUs by invocation proportion, then adjust."""
        if not agents:
            return []

        min_units = min_instance_group
        total_units = total_num_gpus // gpu_step
        base_units = [min_units] * len(agents)
        remaining_units = total_units - sum(base_units)
        if remaining_units <= 0:
            gpus = [units * gpu_step for units in base_units]
            if total_num_gpus > sum(gpus):
                gpus[-1] += total_num_gpus - sum(gpus)
            return gpus

        weights = [
            max(float(agent_invocations.get(agent, 0.0)), 1e-9)
            for agent in agents
        ]
        total_weight = sum(weights)
        raw_extra = [
            remaining_units * weight / total_weight
            for weight in weights
        ]
        extra_units = [floor(value) for value in raw_extra]

        while sum(extra_units) < remaining_units:
            idx = max(
                range(len(agents)),
                key=lambda i: raw_extra[i] - extra_units[i],
            )
            extra_units[idx] += 1

        gpus = [
            (base_units[i] + extra_units[i]) * gpu_step
            for i in range(len(agents))
        ]
        remainder = total_num_gpus - sum(gpus)
        if remainder > 0:
            gpus[-1] += remainder
        return gpus

    @staticmethod
    def _find_donor(
        agents: List[str],
        dp_results: Dict[str, Dict[int, InstanceAssignment]],
        gpus: List[int],
        min_throughput: float,
        agent_invocations: Dict[str, float],
        G: int = 1,
    ) -> int:
        """Find the best agent to donate `G` GPUs.

        Returns the index of the agent whose throughput after losing `G` GPUs
        remains the highest above the current minimum (most headroom).
        """
        best_idx = -1
        max_diff = -1.0

        for i, agent in enumerate(agents):
            if gpus[i] <= G:
                continue  # Cannot donate below G GPUs

            tp_after = (
                dp_results[agent][gpus[i] - G].throughput
                / agent_invocations[agent]
            )
            diff = tp_after - min_throughput
            if diff > max_diff:
                max_diff = diff
                best_idx = i

        return best_idx
