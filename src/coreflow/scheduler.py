"""Async per-agent scheduler.

Routes a DAG node to the instance that is most likely to be able to reuse its
KV cache.  The routing decision combines:

1. Exact retained GPU entries, keyed by ``(query_id, invocation_id)``.
2. Prefix-cache content hashes reported by each instance for both GPU and CPU.
3. Context-range membership when hash-hit counts are tied.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from coreflow.instance_manager import InstanceInfo, InstanceManager

logger = logging.getLogger("async_scheduler")


class AsyncAgentScheduler:
    """Async scheduler for one agent.

    Routing affinity is maintained for ``(query_id, invocation_id)`` so all
    repeats of an invocation stay on the same instance.
    """

    def __init__(
        self,
        agent_id: str,
        instance_manager: InstanceManager,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.agent_id = agent_id
        self._im = instance_manager
        self._http = http_client
        self._routing: Dict[Tuple[int, int], str] = {}
        self._load: Dict[str, Set[Tuple[int, int]]] = {}
        self._routing_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()

    async def _add_load(
        self, instance_id: str, query_id: int, invocation_id: int
    ) -> None:
        async with self._load_lock:
            self._load.setdefault(instance_id, set()).add(
                (query_id, invocation_id)
            )

    async def _remove_load(
        self, instance_id: str, query_id: int, invocation_id: int
    ) -> None:
        async with self._load_lock:
            keys = self._load.get(instance_id)
            if keys:
                keys.discard((query_id, invocation_id))

    async def get_load(self, instance_id: str) -> int:
        async with self._load_lock:
            return len(self._load.get(instance_id, set()))

    async def _commit_route(
        self, key: Tuple[int, int], instance_id: str
    ) -> str:
        """Record the selected route, preserving concurrent-safety."""
        async with self._routing_lock:
            if key in self._routing:
                return self._routing[key]
            self._routing[key] = instance_id

        await self._add_load(instance_id, key[0], key[1])
        return instance_id

    async def _fetch_cache_state(
        self, info: InstanceInfo
    ) -> Tuple[InstanceInfo, Optional[Dict[str, Any]]]:
        """Fetch one instance's routing cache state."""
        url = f"{info.endpoint}/cache_state"
        try:
            response = await self._http.get(url)
            response.raise_for_status()
            return info, response.json()
        except Exception as exc:
            logger.warning(
                "cache_state fetch failed for %s: %s", info.instance_id, exc
            )
            return info, None

    @staticmethod
    def _normalize_token_ids(input_tokens: Any) -> List[int]:
        if isinstance(input_tokens, list):
            return [int(token) for token in input_tokens]
        if isinstance(input_tokens, tuple):
            return [int(token) for token in input_tokens]
        if isinstance(input_tokens, int):
            return [input_tokens]
        return []

    @staticmethod
    def _compute_request_block_hashes(
        token_ids: List[int], block_size: int, none_hash: int
    ) -> Set[int]:
        """Compute vLLM-compatible block content hashes for the request."""
        hashes: Set[int] = set()
        prev_block_hash = none_hash

        for start in range(0, len(token_ids), block_size):
            block_token_ids = token_ids[start:start + block_size]
            if len(block_token_ids) < block_size:
                break

            is_first_block = prev_block_hash == none_hash
            block_hash = int.from_bytes(
                hashlib.sha256(
                    repr((
                        is_first_block,
                        prev_block_hash,
                        tuple(block_token_ids),
                        None,
                    )).encode("utf-8")
                ).digest()[:8],
                byteorder="big",
                signed=True,
            )
            hashes.add(block_hash)
            prev_block_hash = block_hash

        return hashes

    @staticmethod
    def _count_hash_hits(
        request_hashes: Set[int], state: Dict[str, Any]
    ) -> int:
        hash_blocks = state.get("hash_blocks") or {}
        gpu_hashes = {int(hash_value) for hash_value in hash_blocks.get("gpu", [])}
        cpu_hashes = {int(hash_value) for hash_value in hash_blocks.get("cpu", [])}
        return len(request_hashes & (gpu_hashes | cpu_hashes))

    async def _route_by_load(
        self,
        key: Tuple[int, int],
        num_input_tokens: int,
    ) -> Optional[str]:
        """Fallback used only when no cache state endpoint is available."""
        slot = self._im.find_context_slot(self.agent_id, num_input_tokens)
        if slot is None:
            return None

        candidates = self._im.get_instances_for_context(self.agent_id, slot)
        if not candidates:
            return None

        loads = [(await self.get_load(info.instance_id), info)
                 for info in candidates]
        best = min(loads, key=lambda item: item[0])[1]
        return await self._commit_route(key, best.instance_id)

    async def route(
        self,
        query_id: int,
        invocation_id: int,
        input_tokens: Any,
        num_input_tokens: int,
    ) -> Optional[str]:
        """Select the best instance for a node.

        Decision order:
        1. Existing ``(query_id, invocation_id)`` affinity.
        2. Exact retained GPU entry matching the same key.
        3. Most prefix-cache hash hits across all instances.
        4. On hash-hit ties, least GPU entries among the matching context range.
        """
        key = (query_id, invocation_id)

        async with self._routing_lock:
            if key in self._routing:
                return self._routing[key]

        infos = self._im.get_instances_for_agent(self.agent_id)
        if not infos:
            return None

        results = await asyncio.gather(
            *(self._fetch_cache_state(info) for info in infos)
        )
        cache_infos = [
            (info, state) for info, state in results if state is not None
        ]
        if not cache_infos:
            return await self._route_by_load(key, num_input_tokens)

        # Exact retained GPU entry has the strongest affinity.
        for info, state in cache_infos:
            for entry in state.get("gpu_entries") or []:
                if (
                    int(entry.get("query_id")) == query_id
                    and int(entry.get("invocation_id")) == invocation_id
                ):
                    return await self._commit_route(key, info.instance_id)

        token_ids = self._normalize_token_ids(input_tokens)

        scored = []
        for info, state in cache_infos:
            block_size = int(state.get("block_size") or 16)
            none_hash = state.get("none_hash")
            if none_hash is None:
                continue
            request_hashes = self._compute_request_block_hashes(
                token_ids, block_size, int(none_hash)
            )
            hits = self._count_hash_hits(request_hashes, state)
            scored.append((hits, info, state))

        if not scored:
            return await self._route_by_load(key, num_input_tokens)

        max_hits = max(hits for hits, _, _ in scored)
        tied = [(info, state) for hits, info, state in scored if hits == max_hits]

        if len(tied) == 1:
            return await self._commit_route(key, tied[0][0].instance_id)

        # Tie-break: among instances serving the requested context range,
        # prefer the instance with the fewest retained exact GPU entries.
        slot = self._im.find_context_slot(self.agent_id, num_input_tokens)
        if slot is None:
            return None

        context_ids = {
            info.instance_id
            for info in self._im.get_instances_for_context(self.agent_id, slot)
        }
        context_candidates = [
            (info, state)
            for info, state in cache_infos
            if info.instance_id in context_ids
        ]
        if not context_candidates:
            return None

        best = min(
            context_candidates,
            key=lambda item: len(item[1].get("gpu_entries") or []),
        )
        return await self._commit_route(key, best[0].instance_id)

    async def dispatch(
        self, instance_id: str, node_dict: dict
    ) -> Optional[dict]:
        info = self._im.get(instance_id)
        if info is None or not info.is_ready:
            return None

        url = f"{info.endpoint}/generate"
        try:
            response = await self._http.post(url, json=node_dict)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("dispatch failed for %s: %s", instance_id, exc)
            return None

    async def release(self, query_id: int, invocation_id: int) -> None:
        key = (query_id, invocation_id)
        async with self._routing_lock:
            instance_id = self._routing.pop(key, None)
        if instance_id:
            await self._remove_load(instance_id, query_id, invocation_id)
