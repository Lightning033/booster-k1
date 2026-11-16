"""
LLM step: ask the model where to go (Booster K1 VLN).

This file owns the VLN LLM logic:
  1. the SYSTEM_PROMPT (role + all standing rules: coordinate system, the 3
     actions, and the JSON output schema),
  2. build the per-step user text (instruction + memory),
  3. order the 3 grid views,
  4. hand system prompt + user text + images to a swappable backend
     (llm_backend.py) that talks to the model and returns raw text,
  5. parse that text into the action dict the pipeline expects.

The system prompt lives HERE (not in the pipeline) so pipeline.py stays light,
and the standing rules sit in the system message - the recommended place for
stable behavior rules and output schemas.

Swapping LM Studio -> OpenAI is a config change in llm_backend.py; nothing in
this file changes.

ask_llm(...) returns:
    {
      "thought": str,
      "action":  "move" | "look_behind" | "stop" | "retry",
      "view":    "front" | "left" | "right",
      "u": int (0..1000),
      "v": int (0..1000),
      "new_planning": [str, ...],
    }
A stop made because the reply was unusable also has "invalid_reply": True.

ask_gesture(instruction) is a second, text-only call, made once after a real
arrival. It returns {"gesture": "nothing" | "dab" | "wave" | "handshake"}.
"""

from __future__ import annotations

import json
import math
import re
from typing import Optional, Sequence

try:                     # imported as a package by the pipeline
    from .llm_backend import get_backend
except ImportError:      # run directly: python llm_client.py
    from llm_backend import get_backend


VIEW_ORDER = ["front", "left", "right"]
VALID_ACTIONS = {"move", "look_behind", "stop"}
VALID_VIEWS = {"front", "left", "right"}
VALID_GESTURES = {"nothing", "dab", "wave", "handshake"}


# The system prompt: role + all the standing rules. Stable across every step,
# so it lives in the system message (best practice for behavior rules + output
# schemas). Per-step content (instruction, memory, images) goes in the user text.
SYSTEM_PROMPT = """You are the navigation brain of a Booster K1 humanoid robot.
Each step you look at the scene and choose where the robot should step next so
it follows the given instruction.

WHAT YOU RECEIVE EACH STEP
- The navigation instruction (the goal).
- Your previous planning (your own to-do notes) and your recent thoughts.
  Use them to track progress: you are CONTINUING a task, not starting over.
- Three camera images: FRONT, LEFT, RIGHT. Each has a coordinate grid drawn on
  it. There is NO back view.
- FRONT view is robot head pointing straight at 0 radian (0°). LEFT view is robot head pointing 1 radian (≈57.3°) left. Similarly RIGHT view is robot head pointing 1 radian (≈57.3°) right. 

COORDINATE SYSTEM (same on every image)
- u = horizontal: 0 = left edge, 1000 = right edge.
- v = vertical:   0 = top (far away), 1000 = bottom (close to the robot).
- A point with a large v (about 700-900) is floor a short distance ahead.

CHOOSE ONE ACTION
- "move": walk toward a point. Put "view" (front/left/right) and "u","v" on
  WALKABLE FLOOR that leads toward the goal. Never pick walls, people, objects,
  or the ceiling/sky.
- "look_behind": turn the body 180 degrees to see behind. Use ONLY when the
  target is likely behind you and is not visible in the three views.
- "stop": the robot has reached the final destination in the instruction
  (roughly within 2-3 meters).

RULES
- Select a point which is far from all obstacles. The robot navigation may have some error so leave some buffer space around so that robot does not move close to obstacles.
- Do multi-step instructions IN ORDER.
- If the next turn is conditional or not reached yet, keep moving forward.
- Keep "thought" to a few short sentences: which area you are in, which view has
  the target, what your plan says to do next, and whether you are progressing.
- Put 1-3 short steps in "new_planning" for what to do after this move.
- Reply with ONLY a JSON object. No extra text, no markdown code fences.

ALWAYS REMEMBER
- If you find yourself almost near the target, then you can end VLN. No need to be perfectly near the target.
- If you were asked to walk to a landmark (example : a ladder) and if you find that you have just crossed that landmark then just end the VLN. This means that you are already near the landmark and it is fine to end the VLN there. No need to move/turn around in such a situation.

JSON FORMAT
{"thought": "...", "action": "move|look_behind|stop", "view": "front|left|right", "u": 0-1000, "v": 0-1000, "new_planning": ["...", "..."]}

For "look_behind" and "stop", still include view/u/v (any valid numbers) - they
are ignored for those actions."""


# The gesture prompt: used once, after the robot reaches the goal. Text only.
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
    """Assemble the per-step user text (dynamic content only; rules are in the
    system prompt)."""
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
    """Ask the model for the next action.

    Args:
        grid_images: {view_name: BGR grid image}, keys among front/left/right.
        instruction: the natural-language navigation goal.
        planning: the model's planning notes from the previous step.
        thought_history: the model's recent thoughts (verbatim, oldest first).
    Returns:
        The action dict documented at the top of this file.
    """
    user_text = build_user_text(instruction, planning or [], thought_history or [])
    images = [(f"{name.upper()} VIEW", grid_images[name])
              for name in VIEW_ORDER if name in grid_images]
    raw_text = get_backend().complete(SYSTEM_PROMPT, user_text, images)
    return parse_response(raw_text)


# ----------------------------- response parsing ----------------------------

def _coordinate(value):
    # Check the original number BEFORE rounding: -0.1 and 1000.1 are invalid.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 1000 or not math.isfinite(value):
        return None
    return int(round(value))


def _extract_json(raw_text: str) -> Optional[dict]:
    """Return the JSON object in a non-empty model reply, or None.

    Strips markdown code fences and tolerates prose around the object.
    """
    text = raw_text.strip()
    if text.startswith("```"):                        # strip ```json ... ``` fences
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # grab the outermost {...}
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    return data if isinstance(data, dict) else None


def _safe_stop(reason: str) -> dict:
    """Fallback when we cannot trust the model output: stop instead of moving.

    "invalid_reply" marks that the robot did NOT reach the goal, so the pipeline
    must not treat this stop as arrival. A stop chosen by the model has no such key.
    """
    return {"thought": reason, "action": "stop", "view": "front",
            "u": 500, "v": 500, "new_planning": [], "invalid_reply": True}


def parse_response(raw_text: Optional[str]) -> dict:
    """Turn the model's raw reply into the validated action dict.

    Defensive on purpose: strips markdown fences, tolerates surrounding prose,
    rejects invalid move coordinates with an internal 'retry' action.
    Malformed JSON, unknown actions, and invalid views still stop navigation.
    A move must explicitly name a valid view and finite coordinates; never
    invent a center/front target when the model omits or corrupts those fields.
    """
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


# ------------------------------ gesture choice ------------------------------

def ask_gesture(instruction: str) -> dict:
    """Ask the model which gesture the instruction requests (text only, no images).

    Returns {"gesture": "nothing" | "dab" | "wave" | "handshake"}.
    """
    user_text = (
        "[Instruction]\n"
        f"{instruction}\n\n"
        "Reply with ONLY the JSON object."
    )
    raw_text = get_backend().complete(GESTURE_SYSTEM_PROMPT, user_text, [])
    return parse_gesture_response(raw_text)


def parse_gesture_response(raw_text: Optional[str]) -> dict:
    """Turn the model's raw reply into {"gesture": name}.

    An empty reply, a reply without a JSON object, or an unknown gesture gives
    "nothing": when we cannot trust the output, the arms do not move.
    """
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
    # Component test:
    #   1) Offline - print exactly what we would send (no server needed).
    #   2) Parse self-test - prove fence stripping and coordinate rejection work.
    #   3) Live - if LM Studio is running with a VLM loaded, do a real call.
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
