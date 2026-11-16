"""
Ask the ROBOT what its torso frame actually is. Reads only; moves nothing.

arm_control.py assumes the torso frame is X forward, Y left, Z up, because the
SDK headers say only "base frame (torso frame)" and never name the axes. That
assumption decides which way the arm points, and until now the only way to check
it was to move the arm and watch.

There is a better way. The SDK has:

    GetFrameTransform(Frame src, Frame dst, Transform &transform)
    Frame: kBody, kHead, kLeftHand, kRightHand, kLeftFoot, kRightFoot

So the robot will report where its own hands and feet are relative to its body.
That is enough to read the convention straight off the hardware:

  * the LEFT hand's Y tells us which sign of Y means left,
  * the FEET are below the body, so their Z tells us which sign means up,
  * the HEAD is above the body, which confirms it the other way,
  * the hand's distance from the body origin gives a real figure for REACH_M
    instead of the 0.32 m guess.

Nothing is commanded. This sends only queries, so it is safe in any mode and
needs no spotter — though the robot does have to be on and reachable.

    python arm_control/frame_probe.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk_client.sdk_client import get_client, retry


def read_transform(client, source, destination):
    """(x, y, z) of destination in source's frame, or None if refused."""
    import booster_robotics_sdk_python as sdk

    transform = sdk.Transform()
    try:
        retry(client.GetFrameTransform, source, destination, transform)
    except RuntimeError as exc:
        print("    GetFrameTransform failed: %s" % exc)
        return None
    position = transform.position_
    return (position.x_, position.y_, position.z_)


def main():
    import booster_robotics_sdk_python as sdk

    client = get_client()
    body = sdk.Frame.kBody

    targets = [
        ("left hand", sdk.Frame.kLeftHand),
        ("right hand", sdk.Frame.kRightHand),
        ("head", sdk.Frame.kHead),
        ("left foot", sdk.Frame.kLeftFoot),
        ("right foot", sdk.Frame.kRightFoot),
    ]

    print("Querying frame transforms from the BODY frame. Nothing moves.\n")
    print("%-12s %9s %9s %9s" % ("part", "x", "y", "z"))
    readings = {}
    for label, frame in targets:
        position = read_transform(client, body, frame)
        if position is None:
            print("%-12s   (unavailable)" % label)
            continue
        readings[label] = position
        print("%-12s %+9.3f %+9.3f %+9.3f" % (label, position[0], position[1],
                                              position[2]))

    print("\n--- what this says about the torso frame ---")
    conclusions = []

    left = readings.get("left hand")
    right = readings.get("right hand")
    if left and right:
        if left[1] > right[1]:
            conclusions.append("+Y is the robot's LEFT  (LEFT_SIGN = 1.0 is correct)")
        else:
            conclusions.append("+Y is the robot's RIGHT (LEFT_SIGN must be -1.0)")

    feet = [readings[name] for name in ("left foot", "right foot")
            if name in readings]
    head = readings.get("head")
    if feet and head:
        foot_z = sum(position[2] for position in feet) / len(feet)
        if head[2] > foot_z:
            conclusions.append("+Z is UP (UP_SIGN = 1.0 is correct); "
                               "head above feet by %.3f m" % (head[2] - foot_z))
        else:
            conclusions.append("+Z is DOWN (UP_SIGN must be -1.0); "
                               "head below feet by %.3f m" % (foot_z - head[2]))
    elif feet:
        foot_z = sum(position[2] for position in feet) / len(feet)
        conclusions.append("feet are at z = %+.3f, so +Z is %s"
                           % (foot_z, "UP" if foot_z < 0 else "DOWN"))

    if left:
        reach = math.hypot(left[0], left[1])
        conclusions.append("left hand is %.3f m from the torso axis, %.3f m "
                           "along z — compare with REACH_M and SHOULDER_Z_M"
                           % (reach, left[2]))

    if not conclusions:
        print("  Nothing conclusive: no transform was returned.")
        print("  GetFrameTransform may be unsupported on this build, in which")
        print("  case `python arm_control/arm_control.py armtest` is the fallback.")
    else:
        for line in conclusions:
            print("  %s" % line)
        print("\nIf any sign disagrees with arm_control.py, change it there and")
        print("re-run this probe. Only then use --arm.")


if __name__ == "__main__":
    main()
