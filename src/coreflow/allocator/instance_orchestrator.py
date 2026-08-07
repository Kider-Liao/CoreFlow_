"""Instance orchestrator for GPU-to-context-range assignment."""

from dataclasses import dataclass, field
from itertools import combinations
from math import floor
from typing import Dict, Iterable, List, Tuple


@dataclass
class InstanceAssignment:
    """Result of a single allocation candidate."""

    throughput: float = 0.0
    instances: List[Tuple[int, int, int, float]] = field(default_factory=list)
    """Each instance: (num_gpus, context_lower_bound, context_upper_bound, throughput)."""


class InstanceOrchestrator:
    """Determines GPU-to-context-range assignment.

    For each candidate context partition scheme, the allocator first initializes
    instance counts proportional to each partition's service demand
    (inverse single-instance throughput), then greedily adjusts counts to match
    the exact GPU budget.  This follows the two-step allocation described in the
    paper: proportional assignment first, discrete budget correction second.
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
        C = self.max_context_idx

        # Collect best result for each reachable GPU count
        reachable: Dict[int, InstanceAssignment] = {}
        for g in range(dn, total_num_gpus + 1, dn):
            best = InstanceAssignment()
            for k in range(min_instance_group, max_instance_group + 1):
                if k * dn > g:
                    continue
                for scheme in self._iter_partition_schemes(
                    num_partitions=k,
                    max_context_idx=C,
                    throughput_cache=instance_throughput_cache,
                ):
                    assignment = self._allocate_scheme(
                        scheme=scheme,
                        total_num_gpus=g,
                        gpu_step=dn,
                    )
                    if assignment.throughput > best.throughput:
                        best = assignment
            reachable[g] = best

        # Fill result for all GPU counts (1..total_num_gpus)
        result: Dict[int, InstanceAssignment] = {}
        prev = InstanceAssignment(throughput=0.0, instances=[])
        for g in range(0, total_num_gpus + 1):
            if g in reachable:
                prev = reachable[g]
            result[g] = prev

        return result

    def _iter_partition_schemes(
        self,
        num_partitions: int,
        max_context_idx: int,
        throughput_cache: Dict[Tuple[int, int], float],
    ) -> Iterable[List[Tuple[int, int, float]]]:
        """Yield valid context partitions with per-partition throughput."""
        if num_partitions == 1:
            key = (0, max_context_idx)
            throughput = throughput_cache.get(key, 0.0)
            if throughput > 0:
                yield [(0, max_context_idx, throughput)]
            return

        for boundaries in combinations(
            range(1, max_context_idx),
            num_partitions - 1,
        ):
            points = (0, *boundaries, max_context_idx)
            scheme: List[Tuple[int, int, float]] = []
            valid = True
            for lo, hi in zip(points, points[1:]):
                throughput = throughput_cache.get((lo, hi), 0.0)
                if throughput <= 0:
                    valid = False
                    break
                scheme.append((lo, hi, throughput))
            if valid:
                yield scheme

    def _allocate_scheme(
        self,
        scheme: List[Tuple[int, int, float]],
        total_num_gpus: int,
        gpu_step: int,
    ) -> InstanceAssignment:
        """Allocate instances for one partition scheme.

        ``scheme`` entries are ``(lo_idx, hi_idx, single_instance_throughput)``.
        Slower partitions receive a larger initial share because their inverse
        throughput contributes more service demand.
        """
        total_instances = total_num_gpus // gpu_step
        num_partitions = len(scheme)
        if total_instances < num_partitions:
            return InstanceAssignment()

        inverse_tp = [1.0 / throughput for _, _, throughput in scheme]
        demand_sum = sum(inverse_tp)
        raw_counts = [
            total_instances * demand / demand_sum
            for demand in inverse_tp
        ]
        counts = [max(1, floor(count)) for count in raw_counts]

        while sum(counts) < total_instances:
            bottleneck = min(
                range(num_partitions),
                key=lambda idx: counts[idx] * scheme[idx][2],
            )
            counts[bottleneck] += 1

        while sum(counts) > total_instances:
            candidates = [
                idx for idx, count in enumerate(counts)
                if count > 1
            ]
            if not candidates:
                return InstanceAssignment()
            victim = max(
                candidates,
                key=lambda idx: (counts[idx] - 1) * scheme[idx][2],
            )
            counts[victim] -= 1

        throughput = min(
            counts[idx] * scheme[idx][2]
            for idx in range(num_partitions)
        )
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
