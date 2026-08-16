"""Async per-agent scheduler.

Routing is stateless with respect to ``(query_id, invocation_id)``.  The
scheduler refreshes every instance's cache state on each route and uses:

1. Exact retained GPU entry for the request key.
2. The largest GPU+CPU prefix-cache hash hit count.
3. Context-range membership plus ``active_requests + gpu_entries`` as a
   tie-break.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from coreflow.instance_manager import InstanceInfo, InstanceManager

logger = logging.getLogger("async_scheduler")


class AsyncAgentScheduler:
    """Async scheduler for one agent."""

    def __init__(
        self,
        agent_id: str,
        instance_manager: InstanceManager,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.agent_id = agent_id
        self._im = instance_manager
        self._http = http_client

        # Route always refreshes these values.  Migration may reuse them
        # without an extra /cache_state round-trip.
        self._instance_cache_states: Dict[str, Dict[str, Any]] = {}

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
    ) -> List[int]:
        """Compute ordered vLLM-compatible block hashes for the request."""
        hashes: List[int] = []
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
            hashes.append(block_hash)
            prev_block_hash = block_hash

        return hashes

    @staticmethod
    def _count_hash_hits(
        request_hashes: List[int], state: Dict[str, Any]
    ) -> int:
        hash_blocks = state.get("hash_blocks") or {}
        gpu_hashes = {
            int(hash_value) for hash_value in hash_blocks.get("gpu", [])
        }
        cpu_hashes = {
            int(hash_value) for hash_value in hash_blocks.get("cpu", [])
        }
        cached_hashes = gpu_hashes | cpu_hashes
        hits = 0
        for block_hash in request_hashes:
            if block_hash in cached_hashes:
                hits += 1
            else:
                break
        return hits

    @staticmethod
    def _instance_score(state: Dict[str, Any]) -> int:
        active_requests = int(state.get("active_requests") or 0)
        gpu_entry_count = len(state.get("gpu_entries") or [])
        return active_requests + gpu_entry_count

    async def route(
        self,
        query_id: int,
        invocation_id: int,
        input_tokens: Any,
        num_input_tokens: int,
    ) -> Optional[str]:
        """Select the best instance for a node.

        Decision order:
        1. Exact GPU entry for ``(query_id, invocation_id)``.
        2. Largest prefix-cache hash hit count.
        3. On hit ties, matching context range and smallest
           ``active_requests + gpu_entries`` score.
        """
        infos = self._im.get_instances_for_agent(self.agent_id)
        if not infos:
            return None

        results = await asyncio.gather(
            *(self._fetch_cache_state(info) for info in infos)
        )

        for info, state in results:
            if state is not None:
                self._instance_cache_states[info.instance_id] = state

        cache_infos = [
            (info, state) for info, state in results if state is not None
        ]
        if not cache_infos:
            return None

        # Step 1: exact GPU entry is strongest.
        exact_matches: List[Tuple[InstanceInfo, Dict[str, Any]]] = []
        for info, state in cache_infos:
            for entry in state.get("gpu_entries") or []:
                if (
                    int(entry.get("query_id")) == query_id
                    and int(entry.get("invocation_id")) == invocation_id
                ):
                    exact_matches.append((info, state))
                    break

        if exact_matches:
            best = min(
                exact_matches,
                key=lambda item: self._instance_score(item[1]),
            )
            return best[0].instance_id

        # Step 2: compare prefix-cache hits across all instances.
        token_ids = self._normalize_token_ids(input_tokens)
        scored: List[Tuple[int, InstanceInfo, Dict[str, Any]]] = []

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
            return None

        max_hits = max(hits for hits, _, _ in scored)
        tied = [
            (info, state)
            for hits, info, state in scored
            if hits == max_hits
        ]

        if len(tied) == 1:
            return tied[0][0].instance_id

        # Step 3: hit-count tie.  Prefer instances in the matching context
        # range with the smallest active/gpu-entry score.
        slot = self._im.find_context_slot(
            self.agent_id, num_input_tokens
        )
        if slot is not None:
            context_ids = {
                info.instance_id
                for info in self._im.get_instances_for_context(
                    self.agent_id, slot
                )
            }
            context_candidates = [
                (info, state)
                for info, state in cache_infos
                if info.instance_id in context_ids
            ]
            if context_candidates:
                best = min(
                    context_candidates,
                    key=lambda item: self._instance_score(item[1]),
                )
                return best[0].instance_id

        # No usable context range.  Fall back to the tied instances.
        best = min(
            tied,
            key=lambda item: self._instance_score(item[1]),
        )
        return best[0].instance_id

    async def maybe_migrate_request(
        self,
        source_instance_id: Optional[str],
        query_id: int,
        invocation_id: int,
        input_token_ids: List[int],
        output_token_ids: List[int],
        keep_cache_in_gpu: bool,
        num_input_tokens: Optional[int] = None,
        num_output_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        """Migrate a completed request whose length outgrew its source range."""
        if not source_instance_id:
            return None

        source_info = self._im.get(source_instance_id)
        if source_info is None or source_info.context_range is None:
            return None

        if num_input_tokens is not None and num_output_tokens is not None:
            total_length = int(num_input_tokens) + int(num_output_tokens)
        else:
            total_length = len(input_token_ids) + len(output_token_ids)
        _, upper = source_info.context_range
        if total_length < upper:
            return None

        target_slot = self._im.find_context_slot(
            self.agent_id, total_length
        )
        if target_slot is None:
            return None

        candidates = [
            info
            for info in self._im.get_instances_for_context(
                self.agent_id, target_slot
            )
            if info.instance_id != source_instance_id
        ]
        if not candidates:
            return None

        missing = [
            info
            for info in candidates
            if info.instance_id not in self._instance_cache_states
        ]
        if missing:
            results = await asyncio.gather(
                *(self._fetch_cache_state(info) for info in missing)
            )
            for info, state in results:
                if state is not None:
                    self._instance_cache_states[info.instance_id] = state

        available = [
            info
            for info in candidates
            if self._instance_cache_states.get(info.instance_id) is not None
        ]
        if not available:
            return None

        target_info = min(
            available,
            key=lambda info: self._instance_score(
                self._instance_cache_states[info.instance_id]
            ),
        )

        payload = {
            "query_id": query_id,
            "invocation_id": invocation_id,
            "request_id": (
                f"q{query_id}_i{invocation_id}_r0_migrated"
            ),
            "input_tokens": input_token_ids,
            "output_tokens": output_token_ids,
            "keep_cache_in_gpu": keep_cache_in_gpu,
        }

        try:
            response = await self._http.post(
                f"{target_info.endpoint}/migrate_cache",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            logger.warning(
                "migrate_cache failed for %s: %s",
                target_info.instance_id,
                exc,
            )
            return None

        if not result.get("success"):
            return None

        # The target's cache changed.  Drop it so the next route refreshes.
        self._instance_cache_states.pop(target_info.instance_id, None)
        return {
            "source_instance_id": source_instance_id,
            "target_instance_id": target_info.instance_id,
            "device": result.get("device"),
        }

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
