"""T4 step: start the CoreFlow online controller."""

from __future__ import annotations

import argparse

from common import add_repo_to_path

add_repo_to_path()

from coreflow.controller import run_controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the CoreFlow controller.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--allocation", default="allocation.json")
    args = parser.parse_args()

    ctrl = run_controller(
        host=args.host,
        port=args.port,
        allocation_path=args.allocation,
    )
    print(f"CoreFlow controller is running at http://{args.host}:{args.port}")
    ctrl.wait()


if __name__ == "__main__":
    main()
