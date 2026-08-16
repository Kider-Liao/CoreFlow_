"""Generate and submit complete CoreFlow workload JSON.

``workload`` expands raw node lists into concrete token id lists and cache
flags.  ``submit`` only posts a previously generated workload file.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request

from common import load_json, write_json


def _count_tokens(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], int):
            return value[0]
        return len(value)
    return 0


def _random_tokens(rng: random.Random, length: int) -> List[int]:
    return [rng.randint(0, 1023) for _ in range(max(length, 0))]


def _parse_node_id(node_id: str) -> Tuple[int, int]:
    invocation_id, repeat = node_id.split(":")
    return int(invocation_id), int(repeat)


def _build_workload(
    node_list: Any,
    node_dependency: Any,
    query_id: int,
    work_dir: str,
) -> Dict[str, Any]:
    if isinstance(node_list, dict):
        trace_id = sorted(node_list)[0]
        raw_nodes = node_list[trace_id]
    else:
        trace_id = "trace_0"
        raw_nodes = node_list

    if isinstance(node_dependency, dict) and trace_id in node_dependency:
        raw_dependencies = node_dependency[trace_id]
    else:
        raw_dependencies = node_dependency

    raw_by_key = {
        _parse_node_id(record["node_id"]): record
        for record in raw_nodes
    }
    rng = random.Random(query_id)
    token_states: Dict[int, List[int]] = {}
    expanded_nodes: List[Dict[str, Any]] = []

    ordered_records = sorted(
        raw_nodes,
        key=lambda record: _parse_node_id(record["node_id"]),
    )

    for record in ordered_records:
        node_id = record["node_id"]
        invocation_id, repeat = _parse_node_id(node_id)
        agent_name = record["agent_name"]

        num_input_tokens = _count_tokens(record.get("input_tokens", 0))
        num_output_tokens = _count_tokens(record.get("output_tokens", 0))
        cached_tokens = _count_tokens(record.get("cached_tokens", 0))

        input_length = num_input_tokens + cached_tokens
        output_length = num_output_tokens

        input_token_ids: List[int] = []
        if repeat > 0:
            previous_tokens = token_states.get(invocation_id, [])
            input_token_ids.extend(previous_tokens[:input_length])

        remaining_input = input_length - len(input_token_ids)
        if remaining_input > 0:
            input_token_ids.extend(_random_tokens(rng, remaining_input))

        output_token_ids = _random_tokens(rng, output_length)
        token_states[invocation_id] = input_token_ids + output_token_ids

        next_key = (invocation_id, repeat + 1)
        current_id = f"{invocation_id}:{repeat}"
        next_id = f"{invocation_id}:{repeat + 1}"
        direct_repeat = (
            next_key in raw_by_key
            and raw_dependencies.get(next_id) == [current_id]
        )
        current_total_tokens = num_input_tokens + num_output_tokens

        keep_cache_in_gpu = False
        free_cache = True
        if direct_repeat:
            next_record = raw_by_key[next_key]
            next_cached_tokens = _count_tokens(
                next_record.get("cached_tokens", 0)
            )
            if next_cached_tokens <= current_total_tokens:
                keep_cache_in_gpu = True
                free_cache = False

        expanded_nodes.append({
            "node_id": node_id,
            "agent_name": agent_name,
            "invocation_id": invocation_id,
            "repeat": repeat,
            "input_token_ids": input_token_ids,
            "output_token_ids": output_token_ids,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "cached_tokens": cached_tokens,
            "keep_cache_in_gpu": keep_cache_in_gpu,
            "free_cache": free_cache,
        })

    return {
        "query_id": query_id,
        "work_dir": work_dir,
        "node_dependency": raw_dependencies,
        "node_list": expanded_nodes,
    }


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or submit CoreFlow workloads."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    workload_parser = subparsers.add_parser(
        "workload", help="Expand node lists into a complete workload JSON."
    )
    workload_parser.add_argument("--node-list", default="data/node_list.json")
    workload_parser.add_argument(
        "--node-dependency", default="data/node_dependency.json"
    )
    workload_parser.add_argument("--query-id", type=int, default=0)
    workload_parser.add_argument("--work-dir", default="/tmp/coreflow_workloads")
    workload_parser.add_argument(
        "-o", "--output", default="results/t3_e2e/workload.json"
    )

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a generated workload file to a controller."
    )
    submit_parser.add_argument(
        "--workload", default="results/t3_e2e/workload.json"
    )
    submit_parser.add_argument(
        "--controller-url", default="http://127.0.0.1:5000"
    )
    submit_parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args()

    if args.command == "workload":
        node_list = load_json(args.node_list)
        node_dependency = load_json(args.node_dependency)
        query_id = args.query_id or int(time.time() * 1000)
        workload = _build_workload(
            node_list=node_list,
            node_dependency=node_dependency,
            query_id=query_id,
            work_dir=args.work_dir,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, workload)
        print(f"Wrote workload to {output} (query_id={query_id})")
        return

    workload = load_json(args.workload)
    url = args.controller_url.rstrip("/") + "/query"
    status, text = _post_json(url, workload, args.timeout)
    print(f"POST {url} -> {status}")
    print(text)


if __name__ == "__main__":
    main()
