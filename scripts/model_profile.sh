#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  if [[ "$#" -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    echo "Usage: scripts/model_profile.sh"
    exit 0
  fi
  echo "Usage: scripts/model_profile.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Running GPU profiling via coreflow.main. This requires CUDA."
python -m coreflow.main --profile --verify
