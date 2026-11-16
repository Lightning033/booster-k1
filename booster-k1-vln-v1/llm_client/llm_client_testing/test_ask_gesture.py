"""
Manual check for ask_gesture() (Booster K1 VLN). Needs the LM Studio server; no robot.

It sends a few sample instructions to the model and prints the gesture chosen
for each one beside the gesture we expect. With an argument, it asks about that
one instruction only.

Run from anywhere:
    python booster-k1-vln-v1/llm_client/llm_client_testing/test_ask_gesture.py
    python booster-k1-vln-v1/llm_client/llm_client_testing/test_ask_gesture.py "walk to the door and dab"
"""

import sys
import time
from pathlib import Path

# Put the project root on sys.path so we can import the package.
#   this file: <root>/llm_client/llm_client_testing/test_ask_gesture.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_client.llm_backend import DEFAULT_BASE_URL, DEFAULT_MODEL
from llm_client.llm_client import ask_gesture

# (instruction, expected gesture)
SAMPLES = [
    ("Move and stand besides the orange ladder.", "nothing"),
    ("Go to the person and greet them.", "nothing"),
    ("Go to the person near the sofa and shake hands with them.", "handshake"),
    ("Walk to the door, then do a dab.", "dab"),
    ("Go to the human and cheer with both arms.", "wave"),
    ("Walk to the table and wave at everyone.", "wave"),
    ("Walk to the chair and bow.", "nothing"),
]


def run_sample(instruction, expected):
    """Ask the model about one instruction; print the result and return True if it matched."""
    start = time.time()
    gesture = ask_gesture(instruction)["gesture"]
    status = "OK  " if gesture == expected else "DIFF"
    print(f"{status} {time.time() - start:5.1f}s  got={gesture:<9} "
          f"expected={expected:<9} | {instruction}")
    return gesture == expected


def main():
    print(f"Backend: {DEFAULT_BASE_URL}   model: {DEFAULT_MODEL}")
    if len(sys.argv) > 1:
        print(ask_gesture(sys.argv[1]))
        return 0
    matches = sum(run_sample(instruction, expected) for instruction, expected in SAMPLES)
    print(f"{matches}/{len(SAMPLES)} matched the expected gesture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
