"""Plot T2 decoding throughput drop output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/t2_interference/decoding_throughput.csv")
    parser.add_argument("--out", default="results/t2_interference/decoding_throughput.pdf")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped PDF")
        return
    rows = list(csv.DictReader(open(args.csv, "r", encoding="utf-8")))
    if not rows:
        print("CSV empty; skipped PDF")
        return
    agents = [r["agent_name"] for r in rows]
    drops = [float(r["throughput_drop_ratio"]) for r in rows]
    plt.figure(figsize=(6, 4))
    plt.bar(agents, drops)
    plt.ylabel("throughput_drop_ratio")
    plt.title("Estimated colocated decoding overhead")
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
