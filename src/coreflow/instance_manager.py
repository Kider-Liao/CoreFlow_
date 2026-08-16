"""Instance lifecycle management: registration, load tracking, failover.

Each vLLM instance registers itself with the controller on startup (POST /register).
The InstanceManager tracks all active instances and their health status.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class InstanceInfo:
    """Metadata for a single running vLLM instance."""

    instance_id: str
    agent_id: str
    host: str
    port: int
    # Context range this instance is assigned to
    context_range: Optional[Tuple[int, int]] = None
    # Number of GPUs assigned
    num_gpus: int = 1
    status: str = "starting"  # starting | ready | unavailable | draining

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class InstanceManager:
    """Central registry of all active vLLM instances.

    Responsibilities:
      - Accept instance registrations (POST /register).
      - Maintain per-instance status (ready / unavailable).
      - Support failover: mark unavailable and restart on error.
    """

    def __init__(self) -> None:
        # instance_id -> InstanceInfo
        self._instances: Dict[str, InstanceInfo] = {}
        # agent_id -> set of instance_ids
        self._agent_instances: Dict[str, Set[str]] = {}
        # (agent_id, context_range_slot) -> [instance_id, ...]
        self._context_instances: Dict[Tuple[str, int], List[str]] = {}
        # (agent_id, context_range) -> context_range_slot
        self._range_slots: Dict[Tuple[str, Tuple[int, int]], int] = {}
        # (agent_id, context_range_slot) -> context_range
        self._slot_ranges: Dict[Tuple[str, int], Tuple[int, int]] = {}
        # Next port assignment (for auto-allocating)
        self._next_port: int = 8000

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        instance_id: str,
        agent_id: str,
        host: str,
        port: Optional[int] = None,
        context_range: Optional[Tuple[int, int]] = None,
        num_gpus: int = 1,
    ) -> InstanceInfo:
        """Register a newly started instance.

        If port is None, one is auto-assigned.
        Returns the InstanceInfo with the final endpoint.
        """
        if port is None:
            port = self._next_port
            self._next_port += 1

        if context_range is not None:
            context_range = (int(context_range[0]), int(context_range[1]))

        info = InstanceInfo(
            instance_id=instance_id,
            agent_id=agent_id,
            host=host,
            port=port,
            context_range=context_range,
            num_gpus=num_gpus,
        )
        self._instances[instance_id] = info

        # Index by agent
        self._agent_instances.setdefault(agent_id, set()).add(instance_id)

        # Index by context range.  Each distinct range for an agent receives a
        # stable slot so routing can select only instances serving that range.
        if context_range is not None:
            slot = self._get_or_create_slot(agent_id, context_range)
            key = (agent_id, slot)
            context_ids = self._context_instances.setdefault(key, [])
            if instance_id not in context_ids:
                context_ids.append(instance_id)

        return info

    def set_ready(self, instance_id: str) -> None:
        """Mark an instance as ready after successful startup."""
        if instance_id in self._instances:
            self._instances[instance_id].status = "ready"

    def _get_or_create_slot(
        self, agent_id: str, context_range: Tuple[int, int]
    ) -> int:
        key = (agent_id, context_range)
        if key in self._range_slots:
            return self._range_slots[key]

        existing_slots = [
            slot for (a_id, slot) in self._slot_ranges
            if a_id == agent_id
        ]
        slot = max(existing_slots, default=-1) + 1
        self._range_slots[key] = slot
        self._slot_ranges[(agent_id, slot)] = context_range
        return slot

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, instance_id: str) -> Optional[InstanceInfo]:
        return self._instances.get(instance_id)

    def get_instances_for_agent(self, agent_id: str) -> List[InstanceInfo]:
        ids = self._agent_instances.get(agent_id, set())
        return [self._instances[i] for i in ids if self._instances[i].is_ready]

    def get_instances_for_context(
        self, agent_id: str, slot: int
    ) -> List[InstanceInfo]:
        """Return ready instances serving a specific context range slot."""
        key = (agent_id, slot)
        ids = self._context_instances.get(key, [])
        return [self._instances[i] for i in ids if self._instances[i].is_ready]

    def get_context_range_for_slot(
        self, agent_id: str, slot: int
    ) -> Optional[Tuple[int, int]]:
        """Get context range (lower, upper) for a given agent + slot."""
        context_range = self._slot_ranges.get((agent_id, slot))
        if context_range is not None:
            return context_range
        instances = self.get_instances_for_context(agent_id, slot)
        if instances:
            return instances[0].context_range
        return None

    def find_context_slot(
        self, agent_id: str, input_length: int
    ) -> Optional[int]:
        """Find which context range slot an input length falls into."""
        slots = sorted(
            (slot, cr)
            for (a_id, slot), cr in self._slot_ranges.items()
            if a_id == agent_id
        )
        for slot, (lo, hi) in slots:
            if lo <= input_length < hi:
                return slot

        for (a_id, slot), ids in self._context_instances.items():
            if a_id != agent_id:
                continue
            for inst_id in ids:
                info = self._instances[inst_id]
                if info.context_range is not None:
                    lo, hi = info.context_range
                    if lo <= input_length < hi:
                        return slot
        return None

    def all_instance_ids(self) -> List[str]:
        return list(self._instances.keys())

    # ------------------------------------------------------------------
    # Failover
    # ------------------------------------------------------------------
    def mark_unavailable(self, instance_id: str) -> Optional[InstanceInfo]:
        """Mark an instance as unavailable. Returns the info for restart."""
        info = self._instances.get(instance_id)
        if info:
            info.status = "unavailable"
        return info

    def remove(self, instance_id: str) -> None:
        """Remove a dead instance from all indexes."""
        info = self._instances.pop(instance_id, None)
        if info is None:
            return
        self._agent_instances.get(info.agent_id, set()).discard(instance_id)
        if info.context_range is not None:
            slot = self._range_slots.get((info.agent_id, info.context_range))
            if slot is not None:
                key = (info.agent_id, slot)
                lst = self._context_instances.get(key, [])
                if instance_id in lst:
                    lst.remove(instance_id)

    def find_alternative(
        self, agent_id: str, slot: int, exclude_instance_id: str
    ) -> Optional[InstanceInfo]:
        """Find another ready instance in the same (agent, slot)."""
        for info in self.get_instances_for_context(agent_id, slot):
            if info.instance_id != exclude_instance_id:
                return info
        # Try any instance for this agent
        for info in self.get_instances_for_agent(agent_id):
            if info.instance_id != exclude_instance_id:
                return info
        return None
