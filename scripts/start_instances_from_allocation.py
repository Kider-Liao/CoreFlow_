"""Start or stop vLLM instances according to ``allocation.json``.

GPU devices are declared with the standard ``CUDA_VISIBLE_DEVICES``
environment variable, e.g. ``CUDA_VISIBLE_DEVICES=1,2``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _parse_gpu_devices() -> List[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES must list the GPUs available to instances"
        )
    return devices


def _wait_health(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for instance health: {url}")


def _start_instance(
    *,
    agent_id: str,
    instance_id: str,
    gpu_devices: List[str],
    port: int,
    context_lower: int,
    context_upper: int,
    controller_url: str,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    log_file: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    env["PYTHONPATH"] = (
        f"{ROOT / 'src'}{os.pathsep}{ROOT / 'src' / 'vllm'}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm.instance_api",
        "--host",
        "0.0.0.0",
        "--advertise-host",
        "127.0.0.1",
        "--port",
        str(port),
        "--controller-url",
        controller_url,
        "--instance-id",
        instance_id,
        "--agent-id",
        agent_id,
        "--context-lower",
        str(context_lower),
        "--context-upper",
        str(context_upper),
        "--num-gpus",
        str(len(gpu_devices)),
        "--tensor-parallel-size",
        str(len(gpu_devices)),
        "--model",
        model,
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enforce-eager",
    ]

    log_handle = open(log_file, "w", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start(args: argparse.Namespace) -> None:
    allocation_path = Path(args.allocation)
    if not allocation_path.exists():
        raise SystemExit(f"allocation file not found: {allocation_path}")

    with allocation_path.open() as handle:
        data = json.load(handle)

    allocation = data.get("allocation", {})
    config = data.get("config", {})
    default_gpus_per_instance = int(config.get("gpus_per_instance", 1))

    gpu_pool = _parse_gpu_devices()
    available_gpus = gpu_pool.copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "instances.json"

    controller_url = args.controller_url.rstrip("/")
    model = args.model
    next_port = args.base_port
    instances: List[Dict[str, Any]] = []

    for agent_id, agent_alloc in allocation.items():
        for group in agent_alloc.get("instances", []):
            gpus_per_instance = int(
                group.get("gpus_per_instance", default_gpus_per_instance)
            )
            num_instances = int(
                group.get("num_instances", 1)
            )
            context_lower = int(group["context_lower"])
            context_upper = int(group["context_upper"])

            for index in range(num_instances):
                if len(available_gpus) < gpus_per_instance:
                    raise SystemExit(
                        "not enough CUDA_VISIBLE_DEVICES for allocation; "
                        f"need {gpus_per_instance}, have {len(available_gpus)}"
                    )

                assigned = available_gpus[:gpus_per_instance]
                available_gpus = available_gpus[gpus_per_instance:]
                port = next_port
                next_port += 1
                instance_id = (
                    f"{agent_id}-{context_lower}-{context_upper}-{index}"
                )
                log_file = out_dir / f"{instance_id}.log"

                proc = _start_instance(
                    agent_id=agent_id,
                    instance_id=instance_id,
                    gpu_devices=assigned,
                    port=port,
                    context_lower=context_lower,
                    context_upper=context_upper,
                    controller_url=controller_url,
                    model=model,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    log_file=log_file,
                )

                url = f"http://127.0.0.1:{port}"
                _wait_health(url)

                instances.append({
                    "pid": proc.pid,
                    "port": port,
                    "instance_id": instance_id,
                    "agent_id": agent_id,
                    "gpu_devices": assigned,
                    "context_lower": context_lower,
                    "context_upper": context_upper,
                    "endpoint": url,
                })

    state_path.write_text(json.dumps(instances, indent=2))
    print(f"Started {len(instances)} instances; state written to {state_path}")


def _stop(args: argparse.Namespace) -> None:
    state_path = Path(args.state_file)
    if not state_path.exists():
        return
    instances = json.loads(state_path.read_text())

    for item in instances:
        pid = int(item["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        alive = []
        for item in instances:
            pid = int(item["pid"])
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                continue
        if not alive:
            break
        time.sleep(0.2)

    for item in instances:
        pid = int(item["pid"])
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start")
    start_parser.add_argument("--allocation", default="allocation.json")
    start_parser.add_argument("--controller-url", default="http://127.0.0.1:5000")
    start_parser.add_argument("--model", required=True)
    start_parser.add_argument("--max-model-len", type=int, default=200000)
    start_parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    start_parser.add_argument("--base-port", type=int, default=18001)
    start_parser.add_argument("--out-dir", default="results/t3_e2e")
    start_parser.set_defaults(func=_start)

    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--state-file", default="results/t3_e2e/instances.json")
    stop_parser.set_defaults(func=_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
