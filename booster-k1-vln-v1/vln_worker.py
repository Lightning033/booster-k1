"""Run one instruction in the navigation process and report its final outcome."""

import argparse
import json
import signal
import traceback
from pathlib import Path
from typing import Callable

from pipeline.pipeline import main
from navigation_interrupt import NavigationInterrupt


def run_navigation(instruction: str, result_path: Path, navigate: Callable[[str], None]) -> int:
    """Write the result only after the pipeline returns or unwinds its cleanup."""
    result = {"status": "completed", "message": ""}
    exit_code = 0
    interrupt = NavigationInterrupt()
    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        try:
            # The parent must not signal us during imports or handler setup.
            result_path.with_suffix(".ready").touch()
            navigate(instruction)
            if interrupt.requested:
                result = {"status": "cancelled", "message": "Navigation stopped by operator."}
        except KeyboardInterrupt:
            result = {"status": "cancelled", "message": "Navigation stopped by operator."}
        except BaseException as exc:
            result = {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
            traceback.print_exc()
        interrupt.cleaning_up = True
        result_path.write_text(json.dumps(result), encoding="utf-8")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return exit_code


def cli(navigate: Callable[[str], None] = main) -> int:
    parser = argparse.ArgumentParser(description="Run one VLN server command.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("instruction")
    args = parser.parse_args()
    return run_navigation(args.instruction, args.result, navigate)


if __name__ == "__main__":
    raise SystemExit(cli())
