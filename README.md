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
  --node-list data/React/node_list.json \
  --kernel FlashAttention \
  --model Llama-3-8B
```

T2: cache reuse throughput benefit analysis:

```bash
python scripts/reuse_benefits.py
```

T3: offline profiling, controller startup, and workload generation:

```bash
bash scripts/model_profile.sh
python scripts/system_start.py --allocation allocation.json --port 5000
python scripts/generate.py workload \
  --node-list data/React/node_list.json \
  --node-dependency data/React/node_dependency.json
python scripts/generate.py submit \
  --controller-url http://127.0.0.1:5000 \
  --node-list data/React/node_list.json \
  --node-dependency data/React/node_dependency.json
```