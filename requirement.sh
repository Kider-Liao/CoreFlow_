#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e .

pushd ./src/vllm >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}" \
  python -m pip install -e .
popd >/dev/null

echo "CoreFlow Python dependencies installed."
