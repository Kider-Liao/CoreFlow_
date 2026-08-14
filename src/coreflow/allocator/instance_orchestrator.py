"""Instance orchestrator for GPU-to-context-range assignment."""

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Tuple


@dataclass
class InstanceAssignment:
    """Result of a single allocation candidate."""

    throughput: float = 0.0
    instances: List[Tuple[int, int, int, float]] = field(default_factory=list)
    """Each instance: (num_gpus, context_lower_bound, context_upper_bound, throughput)."""


class InstanceOrchestrator:
    """Determines GPU-to-context-range assignment.

    The algorithm follows the optimized-mapping path from the original
    allocator:

    1. Run a GPU-count-independent DP to select the best context partitioning
       for each ``(num_groups, context_upper)``.  The DP minimizes the sum of
       inverse single-instance throughput, which is equivalent to maximizing the
       balanced throughput of a continuously divisible GPU budget.
    2. For each concrete GPU budget, reconstruct each candidate partitioning
       and greedily assign instances to the currently slowest partition.
    """

    def __init__(
        self,
        context_step: int = 2000,
        max_context_length: int = 200_000,
        dp_gpu_step: int = 1,
    ) -> None:
        self._context_step = context_step
        self._max_context_length = max_context_length
        self._dp_gpu_step = dp_gpu_step

    @property
    def max_context_idx(self) -> int:
        return self._max_context_length // self._context_step

    def solve(
        self,
        total_num_gpus: int,
        instance_throughput_cache: Dict[Tuple[int, int], float],
        max_instance_group: int,
        min_instance_group: int = 1,
    ) -> Dict[int, InstanceAssignment]:
        """Return best assignments for every GPU count up to ``total_num_gpus``.

        Args:
            total_num_gpus: Maximum GPU budget.
            instance_throughput_cache: dict mapping (prev_idx, curr_idx) -> throughput.
            max_instance_group: Maximum number of instance groups.
            min_instance_group: Minimum number of instance groups for valid solutions.

        Returns:
            Dict mapping num_gpus -> InstanceAssignment with optimal throughput
            and list of (gpus, lower_bound, upper_bound, throughput).
        """
        dn = self._dp_gpu_step
        result: Dict[int, InstanceAssignment] = {}
        partition_plans = self._build_partition_plans(
            instance_throughput_cache=instance_throughput_cache,
            max_instance_group=max_instance_group,
        )

        for num_gpus in range(0, total_num_gpus + 1):
            best = InstanceAssignment()
            total_instances = num_gpus // dn
            for num_groups in range(min_instance_group, max_instance_group + 1):
                if total_instances < num_groups:
                    continue
                scheme = partition_plans.get(num_groups)
                if not scheme:
                    continue
                assignment = self._allocate_instances_greedily(
                    scheme=scheme,
                    total_num_gpus=num_gpus,
                    gpu_step=dn,
                )
                if assignment.throughput > best.throughput:
                    best = assignment
            result[num_gpus] = best

        return result

    def _build_partition_plans(
        self,
        instance_throughput_cache: Dict[Tuple[int, int], float],
        max_instance_group: int,
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """Run optimized-mapping DP and return plans ending at max context.

        The DP state ``dp[k][c]`` stores the minimum sum of inverse throughput
        needed to cover context range ``[0, c)`` with exactly ``k`` groups.  It
        is independent of the number of GPUs available later.
        """
        max_context_idx = self.max_context_idx
        valid_transitions: Dict[int, List[Tuple[int, float, float]]] = {}
        for (lo_idx, hi_idx), throughput in instance_throughput_cache.items():
            if throughput <= 0:
                continue
            valid_transitions.setdefault(hi_idx, []).append(
                (lo_idx, 1.0 / throughput, throughput))

        inf = float("inf")
        dp = [
            [inf] * (max_context_idx + 1)
            for _ in range(max_instance_group + 1)
        ]
        parent: Dict[Tuple[int, int], Tuple[int, float]] = {}
        dp[0][0] = 0.0

        for num_groups in range(1, max_instance_group + 1):
            for context_idx in range(1, max_context_idx + 1):
                for prev_idx, cost, throughput in valid_transitions.get(
                        context_idx, []):
                    prev_cost = dp[num_groups - 1][prev_idx]
                    if prev_cost == inf:
                        continue
                    new_cost = prev_cost + cost
                    if new_cost < dp[num_groups][context_idx]:
                        dp[num_groups][context_idx] = new_cost
                        parent[(num_groups, context_idx)] = (
                            prev_idx, throughput)

        plans: Dict[int, List[Tuple[int, int, float]]] = {}
        for num_groups in range(1, max_instance_group + 1):
            if dp[num_groups][max_context_idx] == inf:
                continue
            scheme = self._reconstruct_scheme(
                parent=parent,
                num_groups=num_groups,
                context_idx=max_context_idx,
            )
            if scheme:
                plans[num_groups] = scheme
        return plans

    @staticmethod
    def _reconstruct_scheme(
        parent: Dict[Tuple[int, int], Tuple[int, float]],
        num_groups: int,
        context_idx: int,
    ) -> List[Tuple[int, int, float]]:
        scheme: List[Tuple[int, int, float]] = []
        cur_groups = num_groups
        cur_context = context_idx
        while cur_groups > 0:
            entry = parent.get((cur_groups, cur_context))
            if entry is None:
                return []
            prev_context, throughput = entry
            scheme.append((prev_context, cur_context, throughput))
            cur_context = prev_context
            cur_groups -= 1
        if cur_context != 0:
            return []
        scheme.reverse()
        return scheme

    def _allocate_instances_greedily(
        self,
        scheme: List[Tuple[int, int, float]],
        total_num_gpus: int,
        gpu_step: int,
    ) -> InstanceAssignment:
        """Assign instances by repeatedly improving the bottleneck segment."""
        num_partitions = len(scheme)
        total_instances = total_num_gpus // gpu_step
        if total_instances < num_partitions:
            return InstanceAssignment()

        counts = [1] * num_partitions
        heap = [(scheme[idx][2], idx) for idx in range(num_partitions)]
        heapq.heapify(heap)

        for _ in range(total_instances - num_partitions):
            _cur_tp, idx = heapq.heappop(heap)
            counts[idx] += 1
            heapq.heappush(heap, (counts[idx] * scheme[idx][2], idx))

        throughput = min(
            counts[idx] * scheme[idx][2] for idx in range(num_partitions))
        instances = [
            (
                counts[idx] * gpu_step,
                lo_idx * self._context_step,
                hi_idx * self._context_step,
                single_tp,
            )
            for idx, (lo_idx, hi_idx, single_tp) in enumerate(scheme)
        ]
        return InstanceAssignment(throughput=throughput, instances=instances)
