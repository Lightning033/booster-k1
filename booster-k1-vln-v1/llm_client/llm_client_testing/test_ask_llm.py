"""
Independent component test for ask_llm() (Booster K1 VLN).

Goal: confirm the LLM call works end to end, on its own, before wiring ask_llm()
into the VLN pipeline. It loads the 3 grid views saved in
checkpoints/checkpoint-1/ and asks the model for ONE navigation action.

Prerequisites:
  * pip install openai
  * LM Studio running with a VISION (VLM) model loaded, server started on :1234.
    (Or point elsewhere via VLN_LLM_BASE_URL / VLN_LLM_MODEL / VLN_LLM_API_KEY.)

Run from anywhere:
    python booster-k1-vln/llm_client/llm_client_testing/test_ask_llm.py
    python booster-k1-vln/llm_client/llm_client_testing/test_ask_llm.py "go to the kitchen"
"""

import json
import sys
import time
from pathlib import Path

import cv2

# Put the project root (booster-k1-vln) on sys.path so we can import the package.
#   this file: <root>/llm_client/llm_client_testing/test_ask_llm.py
#   parents[2] = <root> (booster-k1-vln)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from llm_client.llm_client import ask_llm
    from llm_client.llm_backend import DEFAULT_BASE_URL, DEFAULT_MODEL
except ImportError as exc:
    print(f"Import failed: {exc}")
    print("Is the 'openai' package installed?  ->  pip install openai")
    sys.exit(1)


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "checkpoint-1"
# ask_llm() expects the GRID-overlaid views (same as overlay_views() output).
VIEW_FILES = {
    "front": "front_grid.png",
    "left": "left_grid.png",
    "right": "right_grid.png",
}
DEFAULT_INSTRUCTION = "Move straight to leave the room, then turn left and stand near the human"
EXPECTED_KEYS = {"thought", "action", "view", "u", "v", "new_planning"}


def load_grid_images() -> dict:
    """Load the 3 grid views from checkpoint-1 as {name: BGR image}."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {CHECKPOINT_DIR}")

    images = {}
    for name, filename in VIEW_FILES.items():
        path = CHECKPOINT_DIR / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        print(f"  loaded {name:<5s} <- {filename}  ({w}x{h})")
        images[name] = image
    return images


def main() -> int:
    instruction = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INSTRUCTION

    print("=" * 70)
    print("ask_llm() component test")
    print("=" * 70)
    print(f"Backend base_url : {DEFAULT_BASE_URL}")
    print(f"Backend model    : {DEFAULT_MODEL}")
    print(f"Checkpoint dir   : {CHECKPOINT_DIR}")
    print(f"Instruction      : {instruction}")
    print("-" * 70)

    print("Loading grid images:")
    try:
        grid_images = load_grid_images()
    except FileNotFoundError as exc:
        print(f"\nFAILED to load images: {exc}")
        return 1

    # First-step test: no prior planning, no prior thoughts.
    planning: list = []
    thought_history: list = []

    print("-" * 70)
    print("Calling ask_llm() ...")
    start = time.time()
    try:
        result = ask_llm(grid_images, instruction, planning, thought_history)
    except Exception as exc:
        print(f"\nLLM CALL FAILED: {type(exc).__name__}: {exc}")
        print("\nHints:")
        print("  - Is LM Studio running with its server started on port 1234?")
        print("  - Is a VISION (VLM) model loaded?")
        print("  - Is VLN_LLM_MODEL set to the exact model id shown in LM Studio?")
        return 1
    elapsed = time.time() - start

    print(f"\nLLM CALL OK  ({elapsed:.1f}s)")
    print("-" * 70)
    print("Parsed result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 70)

    # Light sanity checks on the returned dict.
    missing = EXPECTED_KEYS - set(result)
    if missing:
        print(f"WARNING: result is missing keys: {missing}")
    if result.get("action") not in {"move", "look_behind", "stop"}:
        print(f"WARNING: unexpected action: {result.get('action')!r}")

    print(f"action = {result.get('action')!r} | view = {result.get('view')!r} | "
          f"(u, v) = ({result.get('u')}, {result.get('v')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
