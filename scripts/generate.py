"""T4 step: generate or submit CoreFlow online workloads."""

from __future__ import annotations

import argparse
import json
from urllib import request
from pathlib import Path
from typing import Any

from common import load_json, write_json


def _build_query(node_list: Any, node_dependency: Any, query_id: int, work_dir: str) -> dict:
    if isinstance(node_list, dict):
        trace_id = sorted(node_list)[0]
        nodes = node_list[trace_id]
    else:
        trace_id = "trace_0"
        nodes = node_list

    if isinstance(node_dependency, dict) and trace_id in node_dependency:
        deps = node_dependency[trace_id]
    else:
        deps = node_dependency

    return {
        "query_id": query_id,
        "work_dir": work_dir,
        "node_list": nodes,
        "node_dependency": deps,
    }


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, str]:
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
    parser = argparse.ArgumentParser(description="Generate or submit CoreFlow workloads.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("workload", help="Export a controller query JSON.")
    export_parser.add_argument("--node-list", default="data/node_list.json")
    export_parser.add_argument("--node-dependency", default="data/node_dependency.json")
    export_parser.add_argument("--query-id", type=int, default=1)
    export_parser.add_argument("--work-dir", default="/tmp/coreflow_workloads")
    export_parser.add_argument("-o", "--output", default="results/t4_e2e/query.json")

    submit_parser = subparsers.add_parser("submit", help="Submit the included workload to a controller.")
    submit_parser.add_argument("--controller-url", default="http://127.0.0.1:5000")
    submit_parser.add_argument("--node-list", default="data/node_list.json")
    submit_parser.add_argument("--node-dependency", default="data/node_dependency.json")
    submit_parser.add_argument("--query-id", type=int, default=1)
    submit_parser.add_argument("--work-dir", default="/tmp/coreflow_workloads")
    submit_parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args()

    node_list = load_json(args.node_list)
    node_dependency = load_json(args.node_dependency)
    query = _build_query(node_list, node_dependency, args.query_id, args.work_dir)

    if args.command == "workload":
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, query)
        print(f"Wrote workload query to {out}")
    else:
        url = args.controller_url.rstrip("/") + "/query"
        status, text = _post_json(url, query, args.timeout)
        print(f"POST {url} -> {status}")
        print(text)


if __name__ == "__main__":
    main()
