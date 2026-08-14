"""Async instance API for CoreFlow instances."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Optional

import httpx
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vllm import SamplingParams
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Device


app = FastAPI()
engine: Optional[AsyncLLMEngine] = None
eos_token_id: Optional[int] = None
controller_url: Optional[str] = None


def _forced_decoding_logits_processor(output_tokens: List[int]):
    def _force_next_token(_past_tokens, logits):
        next_index = min(len(_past_tokens), len(output_tokens) - 1)
        desired_token = output_tokens[next_index]
        forced_logits = torch.full_like(logits, float("-inf"))
        forced_logits[desired_token] = 0.0
        return forced_logits

    return _force_next_token


def _tokens_from_value(value: Any, token_id: int) -> list[int]:
    if isinstance(value, int):
        return [token_id] * max(value, 0)
    if isinstance(value, list):
        return [int(token) for token in value]
    if value is None:
        return []
    raise TypeError(f"Expected token list or token count, got {type(value).__name__}")


class GenerateRequest(BaseModel):
    query_id: Optional[int] = None
    invocation_id: Optional[int] = None
    agent_id: Optional[str] = None
    input_tokens: Any
    output_tokens: Any
    cached_tokens: int = 0
    keep_cache_in_gpu: bool = False
    free_cache: bool = True
    repeat: int = 0
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None


@app.post("/generate")
async def generate(req: GenerateRequest):
    global engine, eos_token_id, controller_url
    if engine is None:
        return {"success": False, "error": "engine not initialized"}

    input_tokens = _tokens_from_value(req.input_tokens, 1)
    output_tokens = _tokens_from_value(req.output_tokens, 2)
    requested_cached_tokens = int(req.cached_tokens or 0)
    if requested_cached_tokens < 0 or requested_cached_tokens > len(input_tokens):
        return {
            "success": False,
            "error": "cached_tokens out of range",
        }
    cached_tokens = min(requested_cached_tokens, max(len(input_tokens) - 1, 0))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max(len(output_tokens), 1),
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        logits_processors=[
            _forced_decoding_logits_processor(output_tokens)
        ] if output_tokens else None,
        extra_args={
            "query_id": req.query_id,
            "invocation_id": req.invocation_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "requested_input_tokens": len(input_tokens),
            "keep_cache_in_gpu": req.keep_cache_in_gpu,
            "free_cache": req.free_cache,
            "eos_token_id": eos_token_id if eos_token_id is not None else -1,
        },
    )

    prompt = {
        "prompt_token_ids": input_tokens or [eos_token_id or 0]
    }
    request_id = f"q{req.query_id}_i{req.invocation_id}_r{req.repeat}"

    generated: List[int] = []
    try:
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id,
        ):
            if output.finished and output.outputs:
                generated = list(output.outputs[0].token_ids)
    except Exception as exc:
        return {
            "success": False,
            "query_id": req.query_id,
            "invocation_id": req.invocation_id,
            "generated_ids": [],
            "input_tokens": len(input_tokens),
            "model_input_tokens": len(input_tokens),
            "output_tokens": len(output_tokens),
            "cached_tokens": cached_tokens,
            "keep_cache_in_gpu": req.keep_cache_in_gpu,
            "free_cache": req.free_cache,
            "error": str(exc),
        }

    token_match = (
        True if not output_tokens
        else generated[:len(output_tokens)] == output_tokens
    )
    error = None if token_match else (
        f"forced decode mismatch: expected={output_tokens} got={generated}"
    )

    if controller_url:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(f"{controller_url}/node_complete", json={
                    "query_id": req.query_id,
                    "invocation_id": req.invocation_id,
                    "repeat": req.repeat,
                    "agent_id": req.agent_id,
                    "keep_cache_in_gpu": req.keep_cache_in_gpu,
                    "free_cache": req.free_cache,
                    "num_input_tokens": req.num_input_tokens or len(input_tokens),
                    "num_output_tokens": req.num_output_tokens or len(output_tokens),
                    "success": token_match,
                    "error": error,
                })
        except Exception:
            pass

    return {
        "success": token_match,
        "query_id": req.query_id,
        "invocation_id": req.invocation_id,
        "generated_ids": generated,
        "input_tokens": len(input_tokens),
        "model_input_tokens": len(input_tokens),
        "output_tokens": len(output_tokens),
        "cached_tokens": cached_tokens,
        "keep_cache_in_gpu": req.keep_cache_in_gpu,
        "free_cache": req.free_cache,
        "token_match": token_match,
        "error": error,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cache_state")
async def cache_state():
    global engine
    if engine is None:
        return {"error": "engine not initialized"}

    scheduler = engine.engine.scheduler[0]
    block_manager = scheduler.block_manager
    return {
        "block_size": block_manager.block_size,
        "none_hash": PrefixCachingBlock._none_hash,
        "gpu_entries": [
            {"query_id": query_id, "invocation_id": invocation_id}
            for query_id, invocation_id
            in block_manager.get_gpu_cached_entry_keys()
        ],
        "hash_blocks": {
            "gpu": block_manager.get_cached_block_hashes(Device.GPU),
            "cpu": block_manager.get_cached_block_hashes(Device.CPU),
        },
    }


def run_instance_server(
    host: str,
    port: int,
    async_engine: AsyncLLMEngine,
    tokenizer_eos: Optional[int],
    controller: Optional[str],
) -> None:
    global engine, eos_token_id, controller_url
    engine = async_engine
    eos_token_id = tokenizer_eos
    controller_url = controller
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--controller-url", default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--context-lower", type=int, default=0)
    parser.add_argument("--context-upper", type=int, default=200000)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-queue-stats", action="store_true")
    parser.add_argument("--queue-log-interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    os.environ["VLLM_USE_V1"] = "0"
    engine_kwargs = dict(
        model=args.model,
        load_format=args.load_format,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        preemption_mode="swap",
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        skip_tokenizer_init=args.skip_tokenizer_init,
    )
    if args.swap_space is not None:
        engine_kwargs["swap_space"] = args.swap_space
    engine_args = AsyncEngineArgs(**engine_kwargs)
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer_eos = None
    if not args.skip_tokenizer_init:
        # Tokenizer retrieval is async in vLLM, but this startup path is sync.
        # We keep the previous behavior by leaving eos unset when skipped;
        # otherwise tokenizer loading is left to vLLM's async path.
        tokenizer_eos = None

    if args.controller_url and args.instance_id and args.agent_id:
        try:
            import urllib.request
            payload = {
                "instance_id": args.instance_id,
                "agent_id": args.agent_id,
                "host": args.advertise_host,
                "port": args.port,
                "context_range": [args.context_lower, args.context_upper],
                "num_gpus": args.num_gpus,
            }
            req = urllib.request.Request(
                f"{args.controller_url.rstrip('/')}/register",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
        except Exception:
            pass

    run_instance_server(
        host=args.host,
        port=args.port,
        async_engine=async_engine,
        tokenizer_eos=tokenizer_eos,
        controller=args.controller_url,
    )


if __name__ == "__main__":
    main()
