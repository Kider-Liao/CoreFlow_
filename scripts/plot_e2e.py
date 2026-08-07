"""Summarize T4 end-to-end artifact outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/t4_e2e/README.txt")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "T4 end-to-end execution uses:\n"
        "  python scripts/system_start.py --port 5000\n"
        "  python scripts/generate.py submit --controller-url http://127.0.0.1:5000\n\n"
        "Full-scale experiment logs can be summarized from this directory.\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
