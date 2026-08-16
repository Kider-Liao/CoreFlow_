#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-coreflow}"

conda run -n "${CONDA_ENV}" python scripts/reuse_benefits.py \
  --model /tmp/llama-3-8b \
  --load-format dummy \
  --skip-tokenizer-init \
  --input-tokens 2000,4000,8000,16000 \
  --ratios 0,0.2,0.4,0.6,0.8,1.0 \
  --rate 5 \
  --duration 300 \
  --throughput-window 60 \
  --output-tokens 106 \
  --warmup-requests 0 \
  --out-dir results/t2_throughput \
  --startup-timeout 900 \
  --request-timeout 1800 \
  --drain-timeout 1800 \
  --max-model-len 32768 \
  --port 8916 \
  --cuda-visible-devices 0 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --max-workers 2048 \
  --log-queue-stats \
  --queue-log-interval 10 \
  "$@"
