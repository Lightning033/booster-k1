from __future__ import annotations

import json
import math
import re
from typing import Optional, Sequence

try:
    from .llm_backend import get_backend
except ImportError:
    from llm_backend import get_backend


VIEW_ORDER = ["front", "left", "right"]
VALID_ACTIONS = {"move", "look_behind", "stop"}
VALID_VIEWS = {"front", "left", "right"}
VALID_GESTURES = {"nothing", "dab", "wave", "handshake"}


SYSTEM_PROMPT = """Guide a Booster K1 robot to follow the navigation instruction in order.
Use the previous plan, recent thoughts, and front/left/right images to track progress.

Each step:
1. Review progress toward the goal.
2. Choose one action: "move" toward clear floor away from obstacles,
   "look_behind" if the target is likely behind, or "stop" when near the goal.
3. Give a brief thought and update the remaining plan.

For "move", select a view and coordinates: u runs left to right, v top to bottom,
both from 0 to 1000. For other actions, use view="front", u=500, v=500.

Return only JSON:
{"thought": "...", "action": "move|look_behind|stop", "view": "front|left|right", "u": 500, "v": 500, "new_planning": ["..."]}"""


GESTURE_SYSTEM_PROMPT = """You choose the gesture a Booster K1 humanoid robot performs after it reaches
the goal of a navigation instruction.

GESTURES
- "handshake": hold out a hand to shake hands with a person.
- "dab": do a dab gesture.
- "wave": cheer by waving both arms.
- "nothing": no gesture.

RULES
- Choose a gesture ONLY if the instruction asks the robot to do it, in any wording
  (for example "shake hands", "greet him with a handshake", "do a dab", "wave",
  "cheer", "wave your arms").
- If the instruction does not ask for a gesture, choose "nothing". Do not guess a
  gesture from the situation: "go to the person and greet them" is "nothing".
- If the instruction asks for a gesture that is not in the list (for example
  "bow"), choose "nothing".
- If the instruction asks for more than one gesture, choose the first one.
- Reply with ONLY a JSON object. No extra text, no markdown code fences.

JSON FORMAT
{"gesture": "nothing|dab|wave|handshake"}"""


def _format_list(items: Sequence[str]) -> str:
    if not items:
        return "None yet."
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def build_user_text(instruction: str, planning: Sequence[str],
                    thought_history: Sequence[str]) -> str:
    return (
        "[Instruction]\n"
        f"{instruction}\n\n"
        "[Previous Planning]\n"
        f"{_format_list(planning)}\n\n"
        "[Recent Thoughts]\n"
        f"{_format_list(thought_history)}\n\n"
        "[Current Views]\n"
        "Three images follow - FRONT, LEFT, RIGHT - each with a 0-1000 grid.\n"
        "Reply with ONLY the JSON object."
    )


def ask_llm(grid_images: dict, instruction: str,
            planning: Optional[Sequence[str]] = None,
            thought_history: Optional[Sequence[str]] = None) -> dict:
    user_text = build_user_text(instruction, planning or [], thought_history or [])
    images = [(f"{name.upper()} VIEW", grid_images[name])
              for name in VIEW_ORDER if name in grid_images]
    raw_text = get_backend().complete(SYSTEM_PROMPT, user_text, images)
    return parse_response(raw_text)


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 1000 or not math.isfinite(value):
        return None
    return int(round(value))


def _extract_json(raw_text: str) -> Optional[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    return data if isinstance(data, dict) else None


def _safe_stop(reason: str) -> dict:
    return {"thought": reason, "action": "stop", "view": "front",
            "u": 500, "v": 500, "new_planning": [], "invalid_reply": True}


def parse_response(raw_text: Optional[str]) -> dict:
    if not raw_text or not raw_text.strip():
        return _safe_stop("Empty response from model; stopping for safety.")

    data = _extract_json(raw_text)
    if data is None:
        return _safe_stop("Could not parse JSON from model; stopping for safety.")

    action = str(data.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        return _safe_stop("Unknown action from model; stopping.")

    view = str(data.get("view", "")).strip().lower()
    u = _coordinate(data.get("u"))
    v = _coordinate(data.get("v"))
    if action == "move" and view not in VALID_VIEWS:
        return _safe_stop("Move needs a valid view; stopping.")
    if action == "move" and (u is None or v is None):
        return {"thought": "Move rejected: u/v must be numbers from 0 to 1000. "
                           "No movement occurred; capture again and choose a new point.",
                "action": "retry", "view": view,
                "u": data.get("u"), "v": data.get("v"),
                "new_planning": [], "rejected_response": data}
    if view not in VALID_VIEWS:
        view = "front"

    planning = data.get("new_planning", [])
    if isinstance(planning, list):
        planning = [str(step) for step in planning]
    else:
        planning = [str(planning)]

    return {
        "thought": str(data.get("thought", "")),
        "action": action,
        "view": view,
        "u": u if u is not None else 500,
        "v": v if v is not None else 500,
        "new_planning": planning,
    }


def ask_gesture(instruction: str) -> dict:
    user_text = (
        "[Instruction]\n"
        f"{instruction}\n\n"
        "Reply with ONLY the JSON object."
    )
    raw_text = get_backend().complete(GESTURE_SYSTEM_PROMPT, user_text, [])
    return parse_gesture_response(raw_text)


def parse_gesture_response(raw_text: Optional[str]) -> dict:
    if not raw_text or not raw_text.strip():
        return {"gesture": "nothing"}
    data = _extract_json(raw_text)
    if data is None:
        return {"gesture": "nothing"}
    gesture = str(data.get("gesture", "")).strip().lower()
    if gesture not in VALID_GESTURES:
        return {"gesture": "nothing"}
    return {"gesture": gesture}


if __name__ == "__main__":
    import glob
    from pathlib import Path

    import cv2
    import numpy as np

    HERE = Path(__file__).resolve().parent
    SAMPLES = HERE.parent / "capture_3_views_and_3_depth_images"

    def load_or_dummy(view: str):
        matches = sorted(glob.glob(str(SAMPLES / f"k1_{view}_*.png")))
        return cv2.imread(matches[-1]) if matches else np.full((360, 640, 3), 90, dtype=np.uint8)

    grid_images = {v: load_or_dummy(v) for v in VIEW_ORDER}
    instruction = "Move straight to leave the room, then turn left and stand near the human"
    planning = ["Go through the doorway ahead", "After leaving, turn left to find the human"]
    thoughts = ["I'm in a bedroom. The door is ahead in FRONT, so I'm moving toward it."]

    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    print("\n=== USER TEXT ===")
    print(build_user_text(instruction, planning, thoughts))
    print(f"\n[{len(grid_images)} images attached: {', '.join(VIEW_ORDER)}]")

    print("\n=== PARSE SELF-TEST (fences + out-of-range u/v) ===")
    sample = '```json\n{"thought":"t","action":"move","view":"left","u":1200,"v":-5,"new_planning":["a","b"]}\n```'
    print("parsed:", parse_response(sample))

    print("\n=== LIVE CALL (needs LM Studio running + a VLM loaded) ===")
    try:
        print(ask_llm(grid_images, instruction, planning, thoughts))
    except Exception as exc:
        print(f"Skipped live call: {type(exc).__name__}: {exc}")
