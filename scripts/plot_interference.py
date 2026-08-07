"""Plot T2 interference latency output."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import try_make_line_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/t2_interference/interference_latency.csv")
    parser.add_argument("--out", default="results/t2_interference/interference_latency.pdf")
    args = parser.parse_args()
    ok = try_make_line_plot(
        Path(args.csv),
        "context_length",
        "attention_latency_ms",
        Path(args.out),
        "Estimated attention interference",
        group_col="mode",
    )
    print(f"Wrote {args.out}" if ok else "matplotlib unavailable or CSV empty; skipped PDF")


if __name__ == "__main__":
    main()
