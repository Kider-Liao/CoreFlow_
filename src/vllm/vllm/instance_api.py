"""Instance-level API endpoints for the CoreFlow controller.

Provides HTTP endpoints consumed by the controller's scheduler:
  POST /generate       – Execute an LLM request (forced decoding).
  POST /transfer_kv    – Transfer KV cache blocks to another instance.
  POST /memory_check   – Check if this instance can hold N more tokens.
  POST /clear_cache    – Remove a cached_dict entry.

KV Cache Transfer Protocol
--------------------------
When a request outgrows its current context range, its KV cache must be
moved to another instance.  The protocol works as follows:

1. Scheduler sends POST /transfer_kv to src instance with:
     {"dst_endpoint": "...", "query_id": N, "invocation_id": N}
2. src instance:
   a. Identifies the blocks belonging to (query_id, invocation_id) via
      ``cached_dict`` (a dict mapping (query_id, invocation_id) -> request_id).
   b. Serialises block metadata (block IDs, token positions) and block data
      (the actual KV cache tensors) into a binary payload.
   c. Sends POST /receive_kv to dst instance with the payload.
3. dst instance:
   a. Allocates new blocks for the incoming KV cache.
   b. Copies the tensor data into the allocated blocks.
   c. Records the new mapping in its own ``cached_dict``.
   d. Returns success/failure.
4. src instance clears its ``cached_dict`` entry on success.

This module DOES NOT modify vLLM's core engine.  Instead, it provides
the API layer + protocol logic; the actual block manipulation should be
integrated with vLLM's ``BlockSpaceManager`` and ``LLMEngine``.
"""

from __future__ import annotations

import json
import os
import traceback
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# KV cache management (per-instance state)
# ---------------------------------------------------------------------------
@dataclass
class CachedRequest:
    """Metadata for a finished request whose KV cache is held in GPU memory."""

    request_id: str
    query_id: int
    invocation_id: int
    # List of (block_id, token_start, token_end) for each block
    blocks: List[Tuple[int, int, int]] = field(default_factory=list)
    # Total tokens cached
    token_count: int = 0


class InstanceKVState:
    """Per-instance KV cache state: cached_dict and memory tracking.

    This tracks which completed requests have their KV cache pinned in
    GPU memory (keep_cache_in_gpu=True) and provides the bookkeeping needed
    for cross-instance KV cache transfer.
    """

    def __init__(self, max_gpu_blocks: int = 100_000, block_size: int = 16) -> None:
        # (query_id, invocation_id) -> CachedRequest
        self.cached_dict: Dict[Tuple[int, int], CachedRequest] = {}
        # Currently allocated GPU blocks (occupied + free)
        self.max_gpu_blocks = max_gpu_blocks
        self.block_size = block_size
        # Free block count (simplified — real impl queries vLLM block manager)
        self._free_blocks = max_gpu_blocks

    def has_memory(self, token_count: int) -> bool:
        """Check if there is enough free GPU memory for token_count tokens."""
        blocks_needed = (token_count + self.block_size - 1) // self.block_size
        return blocks_needed <= self._free_blocks

    def add_cached(
        self,
        request_id: str,
        query_id: int,
        invocation_id: int,
        blocks: List[Tuple[int, int, int]],
        token_count: int,
    ) -> None:
        """Record a completed request's KV cache in cached_dict."""
        key = (query_id, invocation_id)
        stale = self.cached_dict.pop(key, None)
        if stale is not None:
            self._free_blocks += len(stale.blocks)
        self.cached_dict[key] = CachedRequest(
            request_id=request_id,
            query_id=query_id,
            invocation_id=invocation_id,
            blocks=blocks,
            token_count=token_count,
        )
        blocks_used = len(blocks)
        self._free_blocks -= blocks_used

    def remove_cached(self, query_id: int, invocation_id: int) -> Optional[CachedRequest]:
        """Remove a cached request and free its blocks."""
        key = (query_id, invocation_id)
        entry = self.cached_dict.pop(key, None)
        if entry:
            self._free_blocks += len(entry.blocks)
        return entry

    def get_cached(self, query_id: int, invocation_id: int) -> Optional[CachedRequest]:
        return self.cached_dict.get((query_id, invocation_id))


# ---------------------------------------------------------------------------
# KV transfer protocol (HTTP-based, between instances)
# ---------------------------------------------------------------------------
def transfer_kv_to(
    src_state: InstanceKVState,
    dst_endpoint: str,
    query_id: int,
    invocation_id: int,
) -> Dict[str, Any]:
    """Execute the KV cache transfer from this (src) instance to dst.

    Steps:
      1. Look up the cached request in src's cached_dict.
      2. Serialise block metadata + data.
      3. POST to dst's /receive_kv.
      4. On success, clear src's cached_dict entry.

    Args:
        src_state: This instance's KV state.
        dst_endpoint: HTTP endpoint of the destination instance.
        query_id, invocation_id: Identifies the cached request.

    Returns:
        {"success": True/False, "error": ...}
    """
    # Step 1: Look up cached request
    entry = src_state.get_cached(query_id, invocation_id)
    if entry is None:
        return {"success": False, "error": f"No cached entry for ({query_id}, {invocation_id})"}

    # Step 2: Build transfer payload
    # In a real implementation, this would contain the actual KV cache
    # tensors (serialised).  For now we send metadata only — the actual
    # tensor transfer requires vLLM block manager integration.
    payload = {
        "query_id": query_id,
        "invocation_id": invocation_id,
        "request_id": entry.request_id,
        "blocks": entry.blocks,       # [(block_id, token_start, token_end), ...]
        "token_count": entry.token_count,
        # block_data would be binary tensor data in a real implementation
    }

    # Step 3: Send to dst
    try:
        url = f"{dst_endpoint}/receive_kv"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "dst rejected")}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    # Step 4: Clear src cached_dict
    src_state.remove_cached(query_id, invocation_id)

    return {"success": True}


def receive_kv_from(
    dst_state: InstanceKVState,
    query_id: int,
    invocation_id: int,
    request_id: str,
    blocks: List[Tuple[int, int, int]],
    token_count: int,
) -> Dict[str, Any]:
    """Receive KV cache blocks from another instance.

    Called by the dst instance's /receive_kv endpoint handler.

    Steps:
      1. Check memory availability.
      2. Allocate blocks (allocate new block IDs).
      3. Copy tensor data (in a real impl).
      4. Record in dst's cached_dict.

    Returns:
        {"success": True/False, "error": ...}
    """
    # Step 1: Memory check
    if not dst_state.has_memory(token_count):
        return {
            "success": False,
            "error": f"Insufficient memory: need {token_count} tokens",
        }

    # Step 2: Allocate blocks (map old block IDs to new ones)
    # In a real implementation, this queries vLLM's BlockSpaceManager.
    new_blocks: List[Tuple[int, int, int]] = []
    next_block = dst_state.max_gpu_blocks - dst_state._free_blocks
    for old_bid, token_start, token_end in blocks:
        new_bid = next_block
        next_block += 1
        new_blocks.append((new_bid, token_start, token_end))

    # Step 3: Copy tensor data (placeholder — real impl uses vLLM internals)
    # copy_block_data(src_blocks, new_blocks)

    # Step 4: Record in dst cached_dict
    dst_state.add_cached(
        request_id=request_id,
        query_id=query_id,
        invocation_id=invocation_id,
        blocks=new_blocks,
        token_count=token_count,
    )

    return {"success": True, "new_blocks": new_blocks}


# ---------------------------------------------------------------------------
# HTTP endpoint handlers (for use with any Python HTTP framework)
# ---------------------------------------------------------------------------
class InstanceAPIHandler:
    """Handlers for the instance-specific API endpoints.

    Usage: attach these to your HTTP server routes.

        handler = InstanceAPIHandler(kv_state)
        server.route("/generate", handler.generate)
        server.route("/transfer_kv", handler.transfer_kv)
        server.route("/memory_check", handler.memory_check)
        server.route("/clear_cache", handler.clear_cache)
    """

    def __init__(
        self,
        kv_state: Optional[InstanceKVState] = None,
        llm: Optional[Any] = None,
        controller_url: Optional[str] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.kv_state = kv_state or InstanceKVState()
        self.llm = llm
        self.controller_url = controller_url.rstrip("/") if controller_url else None
        self.eos_token_id = eos_token_id
        self._generate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # POST /generate
    # ------------------------------------------------------------------
    def generate(self, body: dict) -> dict:
        """Execute an LLM generation request with forced decoding.

        Expected body fields:
            query_id, invocation_id, agent_id,
            input_tokens, output_tokens, num_input_tokens, num_output_tokens,
            keep_cache_in_gpu, free_cache, repeat.

        This delegates to the actual vLLM engine.  In a real deployment
        this would interact with vLLM's AsyncLLMEngine.
        """
        query_id = body.get("query_id")
        invocation_id = body.get("invocation_id")
        agent_id = body.get("agent_id")
        input_tokens = body.get("input_tokens", [])
        output_tokens = body.get("output_tokens", [])
        keep_cache_in_gpu = body.get("keep_cache_in_gpu", False)
        free_cache = body.get("free_cache", True)
        repeat = body.get("repeat", 0)
        num_input_tokens = body.get("num_input_tokens", len(input_tokens))
        num_output_tokens = body.get("num_output_tokens", len(output_tokens))

        if self.llm is None:
            generated = list(output_tokens)
        else:
            from vllm import SamplingParams
            from vllm.model_executor.layers.forced_decode import (
                ForceDecodeLogitsProcessor)

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max(len(output_tokens), 1),
                ignore_eos=True,
                logits_processors=[
                    ForceDecodeLogitsProcessor(
                        list(output_tokens),
                        eos_token_id=self.eos_token_id or -1,
                    )
                ] if output_tokens else None,
                extra_args={
                    "query_id": query_id,
                    "invocation_id": invocation_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "keep_cache_in_gpu": keep_cache_in_gpu,
                    "free_cache": free_cache,
                    "eos_token_id": self.eos_token_id,
                },
            )
            prompt = {"prompt_token_ids": input_tokens or [self.eos_token_id or 0]}
            request_id = f"q{query_id}_i{invocation_id}_r{repeat}"
            with self._generate_lock:
                outputs = self.llm.generate(
                    [prompt],
                    sampling_params,
                    use_tqdm=False,
                )
            if not outputs or not outputs[0].outputs:
                generated = []
            else:
                generated = list(outputs[0].outputs[0].token_ids)

        if free_cache:
            self.kv_state.remove_cached(query_id, invocation_id)
        elif keep_cache_in_gpu:
            token_count = len(input_tokens) + len(output_tokens)
            block_size = self.kv_state.block_size
            blocks = [
                (idx, start, min(start + block_size, token_count))
                for idx, start in enumerate(range(0, token_count, block_size))
            ]
            self.kv_state.add_cached(
                request_id=f"req_{query_id}_{invocation_id}",
                query_id=query_id,
                invocation_id=invocation_id,
                blocks=blocks,
                token_count=token_count,
            )

        success = (
            True if not output_tokens
            else generated[:len(output_tokens)] == list(output_tokens)
        )
        error = None if success else (
            f"forced decode mismatch: expected={output_tokens} got={generated}"
        )

        self._notify_controller({
            "query_id": query_id,
            "invocation_id": invocation_id,
            "repeat": repeat,
            "agent_id": agent_id,
            "keep_cache_in_gpu": keep_cache_in_gpu,
            "free_cache": free_cache,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "success": success,
            "error": error,
        })

        return {
            "success": success,
            "query_id": query_id,
            "invocation_id": invocation_id,
            "generated_ids": generated,
            "keep_cache_in_gpu": keep_cache_in_gpu,
            "free_cache": free_cache,
            "error": error,
        }

    def _notify_controller(self, payload: dict) -> None:
        if not self.controller_url:
            return
        url = f"{self.controller_url}/node_complete"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()

    # ------------------------------------------------------------------
    # POST /transfer_kv
    # ------------------------------------------------------------------
    def transfer_kv(self, body: dict) -> dict:
        """Initiate KV cache transfer to another instance.

        Expected body: {"dst_endpoint": str, "query_id": int, "invocation_id": int}
        """
        return transfer_kv_to(
            src_state=self.kv_state,
            dst_endpoint=body["dst_endpoint"],
            query_id=body["query_id"],
            invocation_id=body["invocation_id"],
        )

    # ------------------------------------------------------------------
    # POST /receive_kv
    # ------------------------------------------------------------------
    def receive_kv(self, body: dict) -> dict:
        """Receive KV cache blocks from a source instance.

        Expected body: {"query_id", "invocation_id", "request_id",
                        "blocks", "token_count"}
        """
        return receive_kv_from(
            dst_state=self.kv_state,
            query_id=body["query_id"],
            invocation_id=body["invocation_id"],
            request_id=body["request_id"],
            blocks=body.get("blocks", []),
            token_count=body.get("token_count", 0),
        )

    # ------------------------------------------------------------------
    # POST /memory_check
    # ------------------------------------------------------------------
    def memory_check(self, body: dict) -> dict:
        """Check if this instance has enough free KV cache memory."""
        token_count = body.get("token_count", 0)
        available = self.kv_state.has_memory(token_count)
        return {"available": available, "free_blocks": self.kv_state._free_blocks}

    # ------------------------------------------------------------------
    # POST /clear_cache
    # ------------------------------------------------------------------
    def clear_cache(self, body: dict) -> dict:
        """Remove an entry from cached_dict (called after migration)."""
        query_id = body["query_id"]
        invocation_id = body["invocation_id"]
        removed = self.kv_state.remove_cached(query_id, invocation_id)
        return {"success": True, "was_cached": removed is not None}


# ---------------------------------------------------------------------------
# Standalone HTTP server (for testing / lightweight deployment)
# ---------------------------------------------------------------------------
def run_instance_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    kv_state: Optional[InstanceKVState] = None,
    llm: Optional[Any] = None,
    controller_url: Optional[str] = None,
    eos_token_id: Optional[int] = None,
) -> None:
    """Run a standalone HTTP server exposing the instance API.

    This is for testing; in production the endpoints are integrated into
    the vLLM server.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    handler_api = InstanceAPIHandler(
        kv_state=kv_state,
        llm=llm,
        controller_url=controller_url,
        eos_token_id=eos_token_id,
    )

    class Handler(BaseHTTPRequestHandler):
        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode())

        def _send_json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                body = self._read_body()
            except Exception:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            routes = {
                "/generate":     handler_api.generate,
                "/transfer_kv":  handler_api.transfer_kv,
                "/receive_kv":   handler_api.receive_kv,
                "/memory_check": handler_api.memory_check,
                "/clear_cache":  handler_api.clear_cache,
            }
            handler = routes.get(self.path)
            if handler:
                try:
                    result = handler(body)
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({
                        "success": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }, 500)
            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": "Not found"}, 404)

        def log_message(self, format, *args):
            pass  # suppress default logging

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Instance API server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a CoreFlow vLLM instance API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--controller-url", default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--context-lower", type=int, default=0)
    parser.add_argument("--context-upper", type=int, default=200000)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    llm = None
    eos_token_id = None
    if args.model:
        os.environ["VLLM_USE_V1"] = "0"
        from vllm import LLM

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            trust_remote_code=args.trust_remote_code,
        )
        eos_token_id = llm.get_tokenizer().eos_token_id

    if args.controller_url and args.instance_id and args.agent_id:
        register_payload = {
            "instance_id": args.instance_id,
            "agent_id": args.agent_id,
            "host": args.advertise_host,
            "port": args.port,
            "context_range": [args.context_lower, args.context_upper],
            "num_gpus": args.num_gpus,
        }
        req = urllib.request.Request(
            f"{args.controller_url.rstrip('/')}/register",
            data=json.dumps(register_payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(resp.read().decode())

    run_instance_server(
        host=args.host,
        port=args.port,
        llm=llm,
        controller_url=args.controller_url,
        eos_token_id=eos_token_id,
    )
