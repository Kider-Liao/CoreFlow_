#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALLOCATION="allocation.json"
OUT_DIR="results/t3_e2e"
CONTROLLER_HOST="127.0.0.1"
CONTROLLER_PORT="5000"
MODEL="${T3_MODEL:-}"
MAX_MODEL_LEN="200000"
GPU_MEMORY_UTILIZATION="0.45"
NODE_LIST="data/node_list.json"
NODE_DEPENDENCY="data/node_dependency.json"
WORK_DIR="/tmp/coreflow_t3"
BASE_PORT="18001"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allocation)
      ALLOCATION="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --controller-host)
      CONTROLLER_HOST="$2"; shift 2 ;;
    --controller-port)
      CONTROLLER_PORT="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --node-list)
      NODE_LIST="$2"; shift 2 ;;
    --node-dependency)
      NODE_DEPENDENCY="$2"; shift 2 ;;
    --work-dir)
      WORK_DIR="$2"; shift 2 ;;
    --base-port)
      BASE_PORT="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi

CONTROLLER_URL="http://${CONTROLLER_HOST}:${CONTROLLER_PORT}"
mkdir -p "$OUT_DIR"
CONTROLLER_LOG="${OUT_DIR}/controller.log"
STATE_FILE="${OUT_DIR}/instances.json"
WORKLOAD_FILE="${OUT_DIR}/workload.json"
CONTROLLER_PID=""

cleanup() {
  if [[ -f "$STATE_FILE" ]]; then
    python scripts/start_instances_from_allocation.py stop \
      --state-file "$STATE_FILE" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CONTROLLER_PID" ]] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    kill "$CONTROLLER_PID" 2>/dev/null || true
    wait "$CONTROLLER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_health() {
  local url="$1"
  for _ in $(seq 1 240); do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/vllm:${PYTHONPATH:-}"

echo "[T3] Generating workload"
python scripts/generate.py workload \
  --node-list "$NODE_LIST" \
  --node-dependency "$NODE_DEPENDENCY" \
  --output "$WORKLOAD_FILE" \
  --work-dir "$WORK_DIR"

echo "[T3] Starting controller on ${CONTROLLER_URL}"
python scripts/system_start.py \
  --host "$CONTROLLER_HOST" \
  --port "$CONTROLLER_PORT" \
  --allocation "$ALLOCATION" \
  >"$CONTROLLER_LOG" 2>&1 &
CONTROLLER_PID=$!
wait_health "$CONTROLLER_URL"

echo "[T3] Starting instances from allocation"
python scripts/start_instances_from_allocation.py start \
  --allocation "$ALLOCATION" \
  --controller-url "$CONTROLLER_URL" \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --base-port "$BASE_PORT" \
  --out-dir "$OUT_DIR"

echo "[T3] Submitting workload to controller"
python scripts/generate.py submit \
  --workload "$WORKLOAD_FILE" \
  --controller-url "$CONTROLLER_URL" \
  --timeout 30

echo "[T3] Waiting for controller query completion"
for _ in $(seq 1 240); do
  if grep -q "Query .* completed" "$CONTROLLER_LOG"; then
    echo "[T3] Workload completed"
    exit 0
  fi
  sleep 1
done

echo "[T3] Workload did not complete within timeout" >&2
exit 1
