"""Plot T3 reuse-throughput output."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import try_make_line_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/t3_reuse/reuse_throughput.csv")
    parser.add_argument("--out", default="results/t3_reuse/reuse_throughput.pdf")
    args = parser.parse_args()
    ok = try_make_line_plot(
        Path(args.csv),
        "reuse_ratio",
        "speedup_vs_no_reuse",
        Path(args.out),
        "Estimated cache reuse speedup",
        group_col="prompt_length",
    )
    print(f"Wrote {args.out}" if ok else "matplotlib unavailable or CSV empty; skipped PDF")


if __name__ == "__main__":
    main()
