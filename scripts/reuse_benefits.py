"""T2: benchmark cache-reuse throughput on a single CoreFlow instance."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib import error, request

from common import REPO_ROOT, ensure_dir, write_csv, write_json


def _parse_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all integers must be positive")
    return result


def _parse_floats(value: str) -> list[float]:
    result = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(item < 0.0 or item > 1.0 for item in result):
        raise argparse.ArgumentTypeError("ratios must be in [0, 1]")
    return result


def _random_token_ids(length: int) -> list[int]:
    """Return random token ids suitable for dummy-model throughput tests."""
    return [random.randint(0, 1023) for _ in range(max(length, 0))]


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> tuple[int, dict]:
    with request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def wait_for_health(instance_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            status, body = _get_json(instance_url.rstrip("/") + "/health", 2.0)
            if status == 200 and body.get("status") == "ok":
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"instance did not become healthy: {last_error}")


def start_instance(args: argparse.Namespace, out_dir: Path) -> tuple[subprocess.Popen, str]:
    host = args.host
    port = args.port if args.port > 0 else _find_free_port(host)
    instance_url = f"http://{host}:{port}"
    max_model_len = args.max_model_len or (max(args.input_tokens) + args.output_tokens)

    env = os.environ.copy()
    python_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src/vllm")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        sys.executable,
        "-m",
        "vllm.instance_api",
        "--host",
        "0.0.0.0",
        "--advertise-host",
        host,
        "--port",
        str(port),
        "--model",
        args.model,
        "--load-format",
        args.load_format,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.skip_tokenizer_init:
        cmd.append("--skip-tokenizer-init")
    if args.log_queue_stats:
        cmd.append("--log-queue-stats")
        cmd.extend(["--queue-log-interval", str(args.queue_log_interval)])

    log_path = out_dir / "instance.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._coreflow_log_file = log_file  # type: ignore[attr-defined]
    return proc, instance_url


def stop_instance(proc: subprocess.Popen | None, keep_instance: bool) -> None:
    if proc is None:
        return
    log_file = getattr(proc, "_coreflow_log_file", None)
    if keep_instance:
        if log_file is not None:
            log_file.flush()
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    if log_file is not None:
        log_file.close()


def _send_one(
    instance_url: str,
    payload: dict,
    timeout_s: float,
    send_time: float,
) -> dict:
    started = time.monotonic()
    record = {
        "request_id": payload["request_id"],
        "input_tokens": len(payload["input_tokens"]),
        "output_tokens": len(payload["output_tokens"]),
        "cached_tokens": payload["cached_tokens"],
        "ratio": payload["ratio"],
        "send_time": send_time,
        "start_time": started,
        "finish_time": None,
        "latency_s": None,
        "success": False,
        "error": None,
    }
    try:
        status, body = _post_json(
            instance_url.rstrip("/") + "/generate",
            payload,
            timeout_s,
        )
        finished = time.monotonic()
        record.update({
            "finish_time": finished,
            "latency_s": finished - started,
            "success": status == 200 and bool(body.get("success")),
            "error": body.get("error"),
            "model_input_tokens": body.get("model_input_tokens"),
        })
    except error.HTTPError as exc:
        finished = time.monotonic()
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        record.update({
            "finish_time": finished,
            "latency_s": finished - started,
            "success": False,
            "error": f"HTTP Error {exc.code}: {body}",
        })
    except Exception as exc:
        finished = time.monotonic()
        record.update({
            "finish_time": finished,
            "latency_s": finished - started,
            "success": False,
            "error": str(exc),
        })
    return record


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1,
              int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def run_group(
    instance_url: str,
    input_tokens: int,
    output_tokens: int,
    ratio: float,
    rate: float,
    duration: float,
    timeout: float,
    drain_timeout: float,
    max_workers: int,
    query_start: int,
    warmup_requests: int = 0,
    throughput_window: float = 60.0,
) -> tuple[dict, list[dict], int]:
    cached_tokens = int(ratio * input_tokens)
    query_cursor = query_start
    for _ in range(warmup_requests):
        payload = {
            "request_id": f"warmup_q{query_cursor}",
            "query_id": query_cursor,
            "invocation_id": 0,
            "input_tokens": _random_token_ids(input_tokens),
            "output_tokens": _random_token_ids(output_tokens),
            "cached_tokens": cached_tokens,
            "ratio": ratio,
            "keep_cache_in_gpu": False,
            "free_cache": True,
        }
        _send_one(instance_url, payload, timeout, time.monotonic())
        query_cursor += 1

    interval = 1.0 / rate
    group_start = time.monotonic()
    deadline = group_start + duration
    futures = []
    request_count = 0

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        next_send = group_start
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.01))
                continue

            query_id = query_cursor + request_count
            payload = {
                "request_id": f"q{query_id}",
                "query_id": query_id,
                "invocation_id": 0,
                "input_tokens": _random_token_ids(input_tokens),
                "output_tokens": _random_token_ids(output_tokens),
                "cached_tokens": cached_tokens,
                "ratio": ratio,
                "keep_cache_in_gpu": False,
                "free_cache": True,
            }
            futures.append(
                executor.submit(
                    _send_one,
                    instance_url,
                    payload,
                    timeout,
                    next_send,
                ))
            request_count += 1
            next_send += interval

        records = []
        try:
            for future in as_completed(futures, timeout=drain_timeout):
                records.append(future.result())
        except FuturesTimeoutError:
            now = time.monotonic()
            for index, future in enumerate(futures):
                if future.done():
                    continue
                future.cancel()
                records.append({
                    "request_id": f"timeout_{query_cursor + index}",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "ratio": ratio,
                    "send_time": None,
                    "start_time": None,
                    "finish_time": now,
                    "latency_s": None,
                    "success": False,
                    "error": "drain timeout",
                })
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    experiment_end = group_start + duration
    throughput_window_s = min(throughput_window, duration)
    throughput_window_start = experiment_end - throughput_window_s
    completed_in_throughput_window = [
        record for record in records
        if (record["success"]
            and throughput_window_start <= record["finish_time"] <= experiment_end)
    ]
    completed_in_experiment = [
        record for record in records
        if record["success"] and record["finish_time"] <= experiment_end
    ]
    completed_after_drain = [record for record in records if record["success"]]
    failed = [record for record in records if not record["success"]]
    latencies = [
        float(record["latency_s"]) for record in completed_after_drain
        if record["latency_s"] is not None
    ]
    row = {
        "model": "",
        "load_format": "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ratio": ratio,
        "cached_tokens": cached_tokens,
        "rate_rps": rate,
        "duration_s": duration,
        "throughput_window_s": throughput_window_s,
        "sent_requests": request_count,
        "completed_requests": len(completed_in_throughput_window),
        "total_completed_requests": len(completed_in_experiment),
        "completed_after_drain": len(completed_after_drain),
        "failed_requests": len(failed),
        "throughput_rps": (
            len(completed_in_throughput_window) / throughput_window_s
        ),
        "overall_throughput_rps": len(completed_in_experiment) / duration,
        "mean_latency_s": mean(latencies) if latencies else 0.0,
        "p50_latency_s": median(latencies) if latencies else 0.0,
        "p95_latency_s": _percentile(latencies, 95),
    }
    return row, records, query_cursor + request_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cache-reuse throughput on a single vLLM instance.")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instance-url", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=106)
    parser.add_argument("--input-tokens", type=_parse_ints, default=[2000, 4000, 8000, 16000])
    parser.add_argument("--ratios", type=_parse_floats, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--drain-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--throughput-window", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--log-queue-stats", action="store_true")
    parser.add_argument("--queue-log-interval", type=float, default=1.0)
    parser.add_argument("--out-dir", default="results/t2_throughput")
    parser.add_argument("--keep-instance", action="store_true")
    args = parser.parse_args()

    if args.output_tokens <= 0:
        parser.error("--output-tokens must be positive")
    if args.rate <= 0 or args.duration <= 0:
        parser.error("--rate and --duration must be positive")
    if args.throughput_window <= 0:
        parser.error("--throughput-window must be positive")
    if args.queue_log_interval <= 0:
        parser.error("--queue-log-interval must be positive")

    out_dir = ensure_dir(args.out_dir)
    proc: subprocess.Popen | None = None
    instance_url = args.instance_url
    try:
        if instance_url is None:
            proc, instance_url = start_instance(args, out_dir)
            print(f"Started T2 instance at {instance_url}")
        wait_for_health(instance_url, args.startup_timeout)
        print(f"Instance is healthy: {instance_url}")

        rows = []
        raw_records = []
        next_query_id = 1
        for input_len in args.input_tokens:
            for ratio in args.ratios:
                print(
                    f"Running input_tokens={input_len}, ratio={ratio}, "
                    f"cached_tokens={int(ratio * input_len)}")
                row, records, next_query_id = run_group(
                    instance_url=instance_url,
                    input_tokens=input_len,
                    output_tokens=args.output_tokens,
                    ratio=ratio,
                    rate=args.rate,
                    duration=args.duration,
                    timeout=args.request_timeout,
                    drain_timeout=args.drain_timeout,
                    max_workers=args.max_workers,
                    query_start=next_query_id,
                    warmup_requests=args.warmup_requests,
                    throughput_window=args.throughput_window,
                )
                row["model"] = args.model
                row["load_format"] = args.load_format
                rows.append(row)
                raw_records.extend(records)

        csv_path = out_dir / "reuse_throughput.csv"
        write_csv(
            csv_path,
            rows,
            [
                "model",
                "load_format",
                "input_tokens",
                "output_tokens",
                "ratio",
                "cached_tokens",
                "rate_rps",
                "duration_s",
                "throughput_window_s",
                "sent_requests",
                "completed_requests",
                "total_completed_requests",
                "completed_after_drain",
                "failed_requests",
                "throughput_rps",
                "overall_throughput_rps",
                "mean_latency_s",
                "p50_latency_s",
                "p95_latency_s",
            ],
        )
        with open(out_dir / "raw_requests.jsonl", "w", encoding="utf-8") as f:
            for record in raw_records:
                f.write(json.dumps(record) + "\n")
        write_json(out_dir / "metadata.json", {
            "model": args.model,
            "load_format": args.load_format,
            "instance_url": instance_url,
            "input_tokens": args.input_tokens,
            "ratios": args.ratios,
            "output_tokens": args.output_tokens,
            "rate": args.rate,
            "duration": args.duration,
            "drain_timeout": args.drain_timeout,
            "warmup_requests": args.warmup_requests,
            "throughput_window": args.throughput_window,
        })
        print(f"Wrote T2 reuse throughput results to {out_dir}")
    finally:
        stop_instance(proc, args.keep_instance)


if __name__ == "__main__":
    main()
