# CoreFlow

CoreFlow is a context-centric resource management prototype for LLM-based
agent workflows. 

## Requirements

The artifact was prepared for Ubuntu 20.04 with Python 3.10, CUDA-capable
NVIDIA GPUs, and a conda environment. The main software dependencies are:

- vLLM 0.9.2
- FastAPI 0.121.0
- SciPy 1.8 or newer
- Ray 2.53.0
- Redis 7.4.0
- seaborn 0.13.2

Install CUDA-compatible PyTorch before installing vLLM if your environment does
not already provide it.

## Installation

Create and activate a clean conda environment:

```bash
conda create -n coreflow python=3.10 -y
conda activate coreflow
```

Install CoreFlow and the bundled vLLM source in editable mode:

```bash
cd /path/to/CoreFlow_
bash requirement.sh
```

The installation script runs:

```bash
python -m pip install -e .
python -m pip install -e ./src/vllm
```

If Ray is used across multiple hosts, start the Ray cluster separately on each
host before running the end-to-end serving experiment.

## Artifact Tasks

Run the following commands from the repository root.

T1: heterogeneous request interference analysis:

```bash
python scripts/interference_analyze.py \
  --kernel FlashAttention \
  --model Llama-3-8B \
  --device cuda:0 \
  --out-dir results/t1_interference
```

T2: cache reuse throughput benefit analysis:

```bash
bash scripts/reuse_benefits.sh
```

The T2 wrapper defaults to the full CUDA graph run with
`max_model_len=32768`, `gpu_memory_utilization=0.9`, `rate=5`,
`duration=300`, `output_tokens=106`, and queue statistics enabled in
`results/t2_throughput/instance.log`. Extra arguments are passed through to
`scripts/reuse_benefits.py`, for example:

```bash
bash scripts/reuse_benefits.sh \
  --out-dir results/t2_throughput_debug \
  --input-tokens 2000 \
  --ratios 0.5 \
  --duration 60
```

T3: offline profiling, resource allocation, controller startup, and workload
submission:

```bash
# 1. offline profiling
python scripts/model_profile.py --device cuda:0 --output-dir Profiler

# 2. resource allocation
python scripts/allocate.py \
  --gpus 32 \
  --data-dir data \
  --profile-dir Profiler \
  --output allocation.json

# 3. start controller and instances described by allocation.json, then
#    generate/submit workload.  GPU devices are declared with the standard
#    CUDA_VISIBLE_DEVICES environment variable.
CUDA_VISIBLE_DEVICES=1,2 \
T3_MODEL=/data/liaojianxiong/hf_config/Qwen2.5-7B \
bash scripts/t3_e2e.sh
```
