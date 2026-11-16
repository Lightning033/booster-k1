"""
Head control — the only module that moves the K1's head.

Confirmed on the robot (30 September 2026):
    RotateHead(pitch, yaw) takes RADIANS and sets an ABSOLUTE target.
    + yaw turns the head to the robot's LEFT.
    + pitch tilts the head DOWN.
    It returns code 400 / 501 unless the robot is in kWalking mode.
    About 1.0 rad is full travel each way in yaw.

A single large jump snaps the head, so every move is stepped in MAX_STEP
increments with a short pause between them. Limits are clamped before anything
is sent.

Standalone, needs the robot in WALK mode and a spotter beside it:
    python head_control/head_control.py yawtest      # + yaw should turn LEFT
    python head_control/head_control.py pitchtest    # + pitch should tilt DOWN
    python head_control/head_control.py center
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk_client.sdk_client import get_client, retry

# Direction signs. Both were confirmed on the robot; change them only after
# re-running yawtest / pitchtest and seeing the head go the wrong way.
YAW_SIGN = 1.0              # +1 means a positive command turns the head LEFT
PITCH_SIGN = 1.0            # +1 means a positive command tilts the head DOWN

YAW_LIMIT = 0.90            # rad; just inside the roughly 1.0 rad of travel
PITCH_LIMIT = 0.35          # rad; deliberately conservative. The published K1
                            # range is wider (about -40 to +30 degrees), but
                            # this has not been measured on this robot. Raise it
                            # only after measuring.
MAX_STEP = 0.10             # rad per command; larger snaps the head
STEP_PAUSE_S = 0.15         # between stepped commands


def clamp(value, limit):
    return max(-limit, min(limit, value))


class HeadController:
    """Remembers where the head was last commanded to go.

    It cannot read the head's true position, so position() is the last target
    that was successfully sent. sync() exists for the case where that belief
    must be abandoned.
    """

    def __init__(self, client=None, yaw_limit=YAW_LIMIT, pitch_limit=PITCH_LIMIT,
                 max_step=MAX_STEP, step_pause_s=STEP_PAUSE_S):
        self.client = client if client is not None else get_client()
        self.yaw_limit = yaw_limit
        self.pitch_limit = pitch_limit
        self.max_step = max_step
        self.step_pause_s = step_pause_s
        self.pitch = 0.0
        self.yaw = 0.0
        self.commands_sent = 0

    def position(self):
        """(pitch, yaw) last successfully commanded."""
        return self.pitch, self.yaw

    def _send(self, pitch, yaw):
        retry(self.client.RotateHead, PITCH_SIGN * pitch, YAW_SIGN * yaw)
        # Update only AFTER the command is accepted, so a failed send does not
        # leave this object believing the head is somewhere it never went.
        self.pitch, self.yaw = pitch, yaw
        self.commands_sent += 1

    def step_towards(self, pitch, yaw):
        """Move ONE step towards (pitch, yaw). Returns True when already there.

        One step per call is what the follow loop wants: it gets a fresh frame
        between steps, so the next step uses a current picture rather than
        finishing a move planned from an old one.
        """
        target_pitch = clamp(pitch, self.pitch_limit)
        target_yaw = clamp(yaw, self.yaw_limit)
        delta_pitch = target_pitch - self.pitch
        delta_yaw = target_yaw - self.yaw
        if abs(delta_pitch) < 1e-4 and abs(delta_yaw) < 1e-4:
            return True
        next_pitch = self.pitch + clamp(delta_pitch, self.max_step)
        next_yaw = self.yaw + clamp(delta_yaw, self.max_step)
        self._send(next_pitch, next_yaw)
        return False

    def look_at(self, pitch, yaw, timeout_s=6.0):
        """Step all the way to (pitch, yaw), pausing between steps."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.step_towards(pitch, yaw):
                return True
            time.sleep(self.step_pause_s)
        return False

    def center(self):
        """Back to straight ahead. Called on exit so the robot is left tidy."""
        return self.look_at(0.0, 0.0)

    def sync(self, pitch, yaw):
        """Adopt (pitch, yaw) as the believed position without commanding anything.

        Used when a frame is too stale to act on: the in-flight move is abandoned
        rather than completed, because finishing it would mean turning the head on
        the strength of a photograph of the past.
        """
        self.pitch = clamp(pitch, self.pitch_limit)
        self.yaw = clamp(yaw, self.yaw_limit)


if __name__ == "__main__":
    from sdk_client.sdk_client import ensure_walking

    command = sys.argv[1] if len(sys.argv) > 1 else "center"
    ensure_walking()
    head = HeadController()

    if command == "yawtest":
        print("Sending yaw = +0.6 rad. WATCH THE HEAD.")
        print("Stand BEHIND the robot: it should turn to the robot's LEFT.")
        head.look_at(0.0, 0.6)
        time.sleep(1.5)
        print("Back to centre.")
        head.center()
    elif command == "pitchtest":
        print("Sending pitch = +0.3 rad. WATCH THE HEAD.")
        print("It should tilt DOWN, towards the floor.")
        head.look_at(0.3, 0.0)
        time.sleep(1.5)
        print("Back to centre.")
        head.center()
    elif command == "center":
        print("Centring the head.")
        head.center()
    else:
        print("usage: python head_control/head_control.py [yawtest|pitchtest|center]")
        raise SystemExit(2)

    print("Done. %d command(s) sent." % head.commands_sent)
