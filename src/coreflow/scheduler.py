"""Per-agent scheduler: routing, load balancing, KV cache migration.

Each agent has one Scheduler that manages a pool of vLLM instances assigned
to specific context ranges.  The scheduler routes requests based on a
(query_id, invocation_id) -> instance_id mapping and handles KV cache
migration when a request outgrows its current context range.
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple

import urllib.request
import urllib.error

from coreflow.instance_manager import InstanceInfo, InstanceManager

logger = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# Migration state tracker
# ---------------------------------------------------------------------------
class MigrationState:
    """Tracks ongoing KV cache migrations between instances.

    During migration a (query_id, invocation_id) is blocked — no requests
    can be dispatched until the migration completes.
    """

    def __init__(self) -> None:
        # Set of (query_id, invocation_id) currently migrating
        self._migrating: Set[Tuple[int, int]] = set()
        self._lock = threading.Lock()

    def is_blocked(self, query_id: int, invocation_id: int) -> bool:
        with self._lock:
            return (query_id, invocation_id) in self._migrating

    def start(self, query_id: int, invocation_id: int) -> None:
        with self._lock:
            self._migrating.add((query_id, invocation_id))

    def finish(self, query_id: int, invocation_id: int) -> None:
        with self._lock:
            self._migrating.discard((query_id, invocation_id))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class AgentScheduler:
    """Routes LLM requests for a single agent to the appropriate instance.

    Routing table:
        (query_id, invocation_id) -> instance_id

    Load metric:
        Number of distinct (query_id, invocation_id) currently mapped
        to an instance (in-flight invocations).
    """

    def __init__(
        self,
        agent_id: str,
        instance_manager: InstanceManager,
    ) -> None:
        self.agent_id = agent_id
        self._im = instance_manager

        # Routing: (query_id, invocation_id) -> instance_id
        self._routing: Dict[Tuple[int, int], str] = {}
        self._routing_lock = threading.Lock()

        # Per-instance load: {instance_id: set of (qid, iid)}
        self._load: Dict[str, Set[Tuple[int, int]]] = {}
        self._load_lock = threading.Lock()

        # Migration tracker
        self._migration = MigrationState()

    # ------------------------------------------------------------------
    # Load tracking
    # ------------------------------------------------------------------
    def _add_load(self, instance_id: str, query_id: int, invocation_id: int) -> None:
        with self._load_lock:
            self._load.setdefault(instance_id, set()).add(
                (query_id, invocation_id)
            )

    def _remove_load(
        self, instance_id: str, query_id: int, invocation_id: int
    ) -> None:
        with self._load_lock:
            s = self._load.get(instance_id)
            if s:
                s.discard((query_id, invocation_id))

    def get_load(self, instance_id: str) -> int:
        with self._load_lock:
            return len(self._load.get(instance_id, set()))

    def get_total_load(self) -> int:
        with self._load_lock:
            return sum(len(s) for s in self._load.values())

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route(
        self,
        query_id: int,
        invocation_id: int,
        num_input_tokens: int,
    ) -> Optional[str]:
        """Return the instance_id for a request.

        On first encounter of (query_id, invocation_id), selects the
        least-loaded instance whose context range covers num_input_tokens.
        Subsequent calls return the cached mapping.
        """
        key = (query_id, invocation_id)

        # Block while migration is in progress
        if self._migration.is_blocked(query_id, invocation_id):
            return None

        with self._routing_lock:
            if key in self._routing:
                return self._routing[key]

        # First-time mapping: find context range slot
        slot = self._im.find_context_slot(self.agent_id, num_input_tokens)
        if slot is None:
            logger.warning(
                "No context range found for agent=%s input=%d",
                self.agent_id, num_input_tokens,
            )
            return None

        # Select least-loaded instance in this slot
        candidates = self._im.get_instances_for_context(self.agent_id, slot)
        if not candidates:
            logger.warning(
                "No ready instances for agent=%s slot=%d",
                self.agent_id, slot,
            )
            return None

        best = min(candidates, key=lambda i: self.get_load(i.instance_id))

        # Record mapping (re-check under lock to avoid race)
        with self._routing_lock:
            if key in self._routing:
                return self._routing[key]
            self._routing[key] = best.instance_id
        self._add_load(best.instance_id, query_id, invocation_id)

        logger.info(
            "Route qid=%d iid=%d -> %s (slot=%d, load=%d)",
            query_id, invocation_id, best.instance_id,
            slot, self.get_load(best.instance_id),
        )
        return best.instance_id

    def get_routing(self, query_id: int, invocation_id: int) -> Optional[str]:
        with self._routing_lock:
            return self._routing.get((query_id, invocation_id))

    def release(self, query_id: int, invocation_id: int) -> None:
        """Release routing entry after a node whose cache is no longer reused."""
        key = (query_id, invocation_id)
        with self._routing_lock:
            inst_id = self._routing.pop(key, None)
        if inst_id:
            self._remove_load(inst_id, query_id, invocation_id)

    # ------------------------------------------------------------------
    # KV Cache migration
    # ------------------------------------------------------------------
    def check_and_migrate(
        self,
        query_id: int,
        invocation_id: int,
        num_input_tokens: int,
        num_output_tokens: int,
    ) -> Optional[Dict]:
        """Evaluate whether KV cache migration is needed after a node completes.

        Called when keep_cache_in_gpu=True. Compares
        (num_input_tokens + num_output_tokens) against the current context
        range's upper bound.

        Returns:
            Migration plan dict if migration is needed AND possible, else None.
            A None return means either no migration needed or migration
            not feasible (dst has no memory) — caller should keep request on
            src instance.
        """
        key = (query_id, invocation_id)
        total_length = num_input_tokens + num_output_tokens

        with self._routing_lock:
            src_id = self._routing.get(key)
        if src_id is None:
            return None

        src_info = self._im.get(src_id)
        if src_info is None:
            return None

        # Find current slot
        current_slot = self._im.find_context_slot(
            self.agent_id, num_input_tokens
        )
        if current_slot is None:
            return None

        # Check if total_length exceeds current range upper bound
        cr = self._im.get_context_range_for_slot(self.agent_id, current_slot)
        if cr is None:
            return None
        _, upper = cr
        if total_length < upper:
            return None  # No migration needed

        # Find new slot for total_length
        new_slot = self._im.find_context_slot(self.agent_id, total_length)
        if new_slot is None or new_slot == current_slot:
            return None  # No suitable new range or same range

        # Find destination instance
        candidates = self._im.get_instances_for_context(
            self.agent_id, new_slot
        )
        if not candidates:
            return None

        dst = min(candidates, key=lambda i: self.get_load(i.instance_id))

        # Check dst memory via a probe request
        has_memory = self._probe_memory(dst, total_length)
        if not has_memory:
            logger.info(
                "Migration qid=%d iid=%d: dst %s memory insufficient, staying on src %s",
                query_id, invocation_id, dst.instance_id, src_id,
            )
            return None

        # Block future requests for this invocation
        self._migration.start(query_id, invocation_id)

        plan = {
            "src_instance_id": src_id,
            "dst_instance_id": dst.instance_id,
            "src_endpoint": src_info.endpoint,
            "dst_endpoint": dst.endpoint,
            "query_id": query_id,
            "invocation_id": invocation_id,
            "token_count": total_length,
        }

        try:
            try:
                success = self._execute_transfer(
                    plan["src_endpoint"],
                    plan["dst_endpoint"],
                    query_id,
                    invocation_id,
                )
            except Exception as exc:
                logger.error("KV transfer failed unexpectedly: %s", exc)
                success = False

            if not success:
                return None

            with self._routing_lock:
                old = self._routing.pop(key, None)
                self._routing[key] = dst.instance_id
            if old:
                self._remove_load(old, query_id, invocation_id)
            self._add_load(dst.instance_id, query_id, invocation_id)
            self._notify_clear_cache(src_info.endpoint, query_id, invocation_id)
            return plan
        finally:
            self._migration.finish(query_id, invocation_id)

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------
    def _probe_memory(self, info: InstanceInfo, token_count: int) -> bool:
        """Ask an instance whether it can hold token_count KV cache tokens."""
        try:
            url = f"{info.endpoint}/memory_check"
            data = json.dumps({"token_count": token_count}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                return result.get("available", False)
        except Exception as exc:
            logger.warning("Memory probe failed for %s: %s", info.endpoint, exc)
            return False

    def _execute_transfer(
        self,
        src_endpoint: str,
        dst_endpoint: str,
        query_id: int,
        invocation_id: int,
    ) -> bool:
        """Initiate KV cache transfer from src to dst instance."""
        try:
            url = f"{src_endpoint}/transfer_kv"
            data = json.dumps({
                "dst_endpoint": dst_endpoint,
                "query_id": query_id,
                "invocation_id": invocation_id,
            }).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result.get("success", False)
        except Exception as exc:
            logger.error("KV transfer failed: %s", exc)
            return False

    def _notify_clear_cache(
        self, endpoint: str, query_id: int, invocation_id: int
    ) -> None:
        """Notify src instance to clear its cached_dict entry."""
        try:
            url = f"{endpoint}/clear_cache"
            data = json.dumps({
                "query_id": query_id,
                "invocation_id": invocation_id,
            }).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.warning("Clear cache notification failed: %s", exc)

    # ------------------------------------------------------------------
    # Dispatch to instance
    # ------------------------------------------------------------------
    def dispatch(
        self,
        instance_id: str,
        node_dict: dict,
    ) -> Optional[dict]:
        """Send a node to an instance for execution via HTTP POST.

        Args:
            instance_id: Target instance.
            node_dict: Serialised node with fields:
                query_id, invocation_id, agent_id,
                input_tokens, output_tokens, num_input_tokens, num_output_tokens,
                keep_cache_in_gpu, free_cache, repeat.

        Returns:
            Response dict from instance, or None on failure.
        """
        info = self._im.get(instance_id)
        if info is None or not info.is_ready:
            logger.warning("Instance %s not ready for dispatch", instance_id)
            return None

        try:
            url = f"{info.endpoint}/generate"
            data = json.dumps(node_dict).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            logger.error("Dispatch HTTP error %d for %s", exc.code, instance_id)
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("Dispatch failed for %s: %s", instance_id, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._load_lock:
            loads = {iid: len(keys) for iid, keys in self._load.items()}
        return {
            "agent_id": self.agent_id,
            "total_load": sum(loads.values()),
            "routing_entries": len(self._routing),
            "instance_loads": loads,
        }
