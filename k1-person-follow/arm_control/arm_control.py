"""
Arm control — points a hand at the person the head is tracking.

UNVERIFIED ON THIS ROBOT. Everything below is read off the SDK headers, not
measured. The head numbers in head_control.py were confirmed by watching the
robot; none of these have been. Specifically unknown:

  * the torso frame's axis directions (assumed X forward, Y left, Z up, which is
    the usual body frame, but Booster's headers only say "base frame (torso
    frame)" without naming the axes),
  * the reachable workspace, so REACH_M and SHOULDER_Z_M are guesses on the
    conservative side,
  * whether an arm move disturbs balance while the robot is standing in kWalking.

Because of that: arm pointing is OFF unless asked for, and `armtest` exists to
check the frame before the follow loop ever uses it. Run armtest first, with
clear space around the robot and somebody beside it.

The SDK call:
    MoveHandEndEffectorV2(Posture, time_millis, HandIndex)
        Posture  = Position(x, y, z) in metres + Orientation(roll, pitch, yaw) in rad,
                   in the torso frame
        HandIndex = kLeftHand (0) or kRightHand (1)
    Supported on K1. Returns kRpcStatusCodeStateTransitionFailed if the planner
    cannot start, which is the error to expect if the pose is unreachable.

This breaks the head-only guarantee the earlier booster-k1-person-recognition
project was built around. That was deliberate there and is deliberately dropped
here, but it means the robot now moves a limb: keep people clear of the arm.

Standalone, robot in WALK mode, clear space, spotter present:
    python arm_control/arm_control.py armtest     # slow, three poses, confirms the frame
    python arm_control/arm_control.py rest        # put the arm back down
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk_client.sdk_client import get_client, retry

# --- Geometry assumptions. All unverified; see the module docstring. ----------
# Torso frame assumed: X forward, Y left, Z up.
FORWARD_SIGN = 1.0          # +X is forward
LEFT_SIGN = 1.0             # +Y is left, matching + head yaw turning left
UP_SIGN = 1.0               # +Z is up

REACH_M = 0.32              # how far from the torso to place the hand
SHOULDER_Z_M = 0.18         # hand height above the torso origin when pointing
REST_POSITION = (0.08, 0.18, -0.22)   # arm down by the side, in torso frame

YAW_LIMIT = 1.20            # rad; do not ask for a bearing beyond this
PITCH_LIMIT = 0.50          # rad

MOVE_TIME_MS = 900          # the planner's duration for one move; slow is safer
MIN_MOVE_INTERVAL_S = 1.2   # never issue arm moves faster than this
MIN_BEARING_CHANGE = 0.15   # rad; ignore smaller changes so the planner is not
                            # spammed with near-identical targets


def clamp(value, limit):
    return max(-limit, min(limit, value))


class ArmPointer:
    """Points one hand along a bearing. Picks the hand on the person's side.

    Rate-limited and deadbanded on purpose: an end-effector move takes most of a
    second to plan and execute, so commanding it at camera frame rate would queue
    up stale targets and fight itself.
    """

    def __init__(self, client=None, reach_m=REACH_M, shoulder_z_m=SHOULDER_Z_M,
                 move_time_ms=MOVE_TIME_MS,
                 min_interval_s=MIN_MOVE_INTERVAL_S,
                 min_bearing_change=MIN_BEARING_CHANGE):
        self.client = client if client is not None else get_client()
        self.reach_m = reach_m
        self.shoulder_z_m = shoulder_z_m
        self.move_time_ms = int(move_time_ms)
        self.min_interval_s = min_interval_s
        self.min_bearing_change = min_bearing_change
        self.last_bearing = None
        self.last_move_at = 0.0
        self.last_hand = None
        self.moves_sent = 0
        self.at_rest = True

    # -- the SDK objects are built here so importing this module needs no SDK --
    def _posture(self, position, orientation):
        import booster_robotics_sdk_python as sdk

        x, y, z = position
        roll, pitch, yaw = orientation
        return sdk.Posture(sdk.Position(float(x), float(y), float(z)),
                           sdk.Orientation(float(roll), float(pitch), float(yaw)))

    def _hand(self, use_left):
        import booster_robotics_sdk_python as sdk

        return sdk.HandIndex.kLeftHand if use_left else sdk.HandIndex.kRightHand

    def _send(self, position, orientation, use_left):
        posture = self._posture(position, orientation)
        retry(self.client.MoveHandEndEffectorV2, posture, self.move_time_ms,
              self._hand(use_left))
        self.moves_sent += 1
        self.last_hand = "left" if use_left else "right"
        self.last_move_at = time.monotonic()

    def target_for(self, bearing_rad, pitch_rad=0.0):
        """(position, orientation, use_left) for pointing along bearing_rad.

        bearing_rad is the same convention as head yaw: positive is the robot's
        left. The hand on that side is used, so the arm never crosses the body.
        """
        bearing = clamp(bearing_rad, YAW_LIMIT)
        pitch = clamp(pitch_rad, PITCH_LIMIT)
        use_left = bearing >= 0.0
        # Place the hand out along the bearing at a fixed reach, raised to roughly
        # shoulder height, and aim the wrist the same way.
        x = FORWARD_SIGN * self.reach_m * math.cos(bearing)
        y = LEFT_SIGN * self.reach_m * math.sin(bearing)
        z = UP_SIGN * self.shoulder_z_m - self.reach_m * math.sin(pitch)
        return (x, y, z), (0.0, pitch, bearing), use_left

    def point_at(self, bearing_rad, pitch_rad=0.0, force=False):
        """Point along bearing_rad if enough has changed and enough time passed.

        Returns True when a command was actually sent.
        """
        now = time.monotonic()
        if not force:
            if now - self.last_move_at < self.min_interval_s:
                return False
            if (self.last_bearing is not None
                    and abs(bearing_rad - self.last_bearing) < self.min_bearing_change):
                return False
        position, orientation, use_left = self.target_for(bearing_rad, pitch_rad)
        self._send(position, orientation, use_left)
        self.last_bearing = bearing_rad
        self.at_rest = False
        return True

    def rest(self):
        """Lower the arm back to its side. Called on exit, and when nobody is locked."""
        if self.at_rest:
            return False
        x, y, z = REST_POSITION
        for use_left in (True, False):
            side = y if use_left else -y
            try:
                self._send((x, side, z), (0.0, 0.0, 0.0), use_left)
            except RuntimeError as exc:
                # Leaving one arm up is bad, so try the other regardless, and
                # report rather than raise: this runs during shutdown.
                print("arm: could not rest %s hand: %s"
                      % ("left" if use_left else "right", exc))
        self.last_bearing = None
        self.at_rest = True
        return True


if __name__ == "__main__":
    from sdk_client.sdk_client import ensure_walking

    command = sys.argv[1] if len(sys.argv) > 1 else "armtest"
    ensure_walking()
    arm = ArmPointer()

    if command == "rest":
        print("Lowering both arms.")
        arm.rest()
    elif command == "armtest":
        print("ARM TEST — the robot will move an arm.")
        print("Clear the space around it and keep a hand on it.")
        print("Three poses, slowly: straight ahead, 45 deg left, 45 deg right.\n")
        try:
            input("Press Enter when the area is clear, or Ctrl+C to stop: ")
        except KeyboardInterrupt:
            print("\ncancelled")
            raise SystemExit(0)

        for label, bearing in (("straight ahead", 0.0),
                               ("45 degrees LEFT", math.radians(45)),
                               ("45 degrees RIGHT", math.radians(-45))):
            position, orientation, use_left = arm.target_for(bearing)
            print("%-18s -> %s hand at x=%+.3f y=%+.3f z=%+.3f, yaw=%+.3f rad"
                  % (label, "left" if use_left else "right",
                     position[0], position[1], position[2], orientation[2]))
            try:
                arm.point_at(bearing, force=True)
            except RuntimeError as exc:
                print("   FAILED: %s" % exc)
                print("   An unreachable pose or a refused planner means REACH_M /"
                      " SHOULDER_Z_M need adjusting, or the frame axes differ"
                      " from the assumption in this module.")
                break
            time.sleep(2.5)

        print("\nDid the hand point where each label said?")
        print("  yes            -> the frame assumptions hold; --arm is safe to use")
        print("  mirrored       -> flip LEFT_SIGN in this file")
        print("  up/down wrong  -> flip UP_SIGN, or adjust SHOULDER_Z_M")
        print("\nLowering arms.")
        arm.rest()
    else:
        print("usage: python arm_control/arm_control.py [armtest|rest]")
        raise SystemExit(2)

    print("Done. %d arm command(s) sent." % arm.moves_sent)
