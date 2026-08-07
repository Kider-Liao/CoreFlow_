"""Shared helpers for CoreFlow artifact scripts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


def add_repo_to_path() -> None:
    root = str(REPO_ROOT / "src")
    vllm_root = str(REPO_ROOT / "src/vllm")
    for path in [root, vllm_root]:
        if path not in sys.path:
            sys.path.insert(0, path)


add_repo_to_path()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def token_sum(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return int(sum(value))
    return 0


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_node_records(node_list: dict):
    for trace_id, nodes in node_list.items():
        for node in nodes:
            input_tokens = token_sum(node.get("input_tokens", 0))
            output_tokens = token_sum(node.get("output_tokens", 0))
            cached_tokens = token_sum(node.get("cached_tokens", 0))
            context_length = input_tokens + cached_tokens
            yield {
                "trace_id": trace_id,
                "node_id": node.get("node_id", ""),
                "agent_name": node.get("agent_name", ""),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "context_length": context_length,
                "total_tokens": context_length + output_tokens,
                "cache_reuse_ratio": (
                    cached_tokens / context_length if context_length > 0 else 0.0
                ),
            }


def try_make_line_plot(
    csv_path: Path,
    x_col: str,
    y_col: str,
    out_pdf: Path,
    title: str,
    group_col: str | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return False

    plt.figure(figsize=(6, 4))
    if group_col is None:
        xs = [float(r[x_col]) for r in rows]
        ys = [float(r[y_col]) for r in rows]
        plt.plot(xs, ys, marker="o")
    else:
        groups = sorted({r[group_col] for r in rows})
        for group in groups:
            subset = [r for r in rows if r[group_col] == group]
            xs = [float(r[x_col]) for r in subset]
            ys = [float(r[y_col]) for r in subset]
            plt.plot(xs, ys, marker="o", label=group)
        plt.legend()
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    return True
