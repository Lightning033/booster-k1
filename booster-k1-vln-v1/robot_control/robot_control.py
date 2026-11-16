"""
Robot movement (Booster K1).

We do NOT stand the robot up. We assume it is already standing in PREP mode.
Our job is to switch it into WALK mode to start VLN, and stop in WALK at the end.

Built:
    enter_walk_mode()  -> ChangeMode(kWalking)   (start of VLN)
    enter_prep_mode()  -> ChangeMode(kPrepare)   (explicit manual command only)
    turn_body(a)       -> turn in place a radians (+ = left, - = right)
    walk_forward(d)    -> walk straight d meters
    look_behind()      -> turn the body 180 degrees
    move_to_target()   -> face the target point + take one bounded step toward it
    perform_gesture(g) -> play dab / wave / handshake for GESTURE_SECONDS, then stop it

The SDK is imported only when a client or mode is needed. Importing this module
on the development machine does not connect to a robot. One client is reused.

Standalone (on the robot):
    python robot_control.py         # enter WALK mode
    python robot_control.py prep    # enter PREP mode
    python robot_control.py dab     # play one gesture: dab | wave | handshake
                                    # (robot must already be in WALK; it stays in WALK)
"""

import math
import sys
import time

_INTERFACE = "wlp5s0"      # network interface for the DDS domain
_client = None             # shared B1LocoClient, created on first use
_channel_initialized = False

# Open-loop motion tuning (from reference-code/move_ahead.py + turn_body.py):
# we command a fixed velocity for a computed time (velocity x time = distance/angle).
WALK_SPEED = 0.2           # m/s, forward walking speed
MAX_STEP_DISTANCE_M = 1.5  # maximum requested walk per VLN decision; then look again
YAW_RATE = 0.5             # rad/s, in-place turn rate
SETTLE_S = 0.5             # short pause after a motion so the robot fully stops
WALK_MODE_SETTLE_S = 5.0    # same mode-entry wait as both tested reference scripts

# Body turn (radians) to face each view, matching the head yaw used at capture:
# the left view was taken with the head at +1.0 rad, the right view at -1.0 rad.
VIEW_TURN = {"front": 0.0, "left": 1.0, "right": -1.0}

# End-of-run gestures. "wave" is the cheering gesture (both arms), not WaveHand().
# "nothing" is also accepted by perform_gesture() and sends no command.
GESTURES = ("dab", "wave", "handshake")
GESTURE_SECONDS = 10.0     # how long a gesture plays before it is stopped


def _get_client():
    """Join the DDS domain and create the client once; reuse it after."""
    global _client, _channel_initialized
    if _client is None:
        from booster_robotics_sdk_python import ChannelFactory, B1LocoClient

        if not _channel_initialized:
            ChannelFactory.Instance().Init(0, _INTERFACE)
            _channel_initialized = True
        client = B1LocoClient()
        check_result(client.Init(), "Init")
        _client = client
    return _client


def get_client():
    """Return the ONE shared B1LocoClient (created once).

    Other steps that need the Booster SDK (e.g. capture_3_views_and_3_depth_images, which drives the
    head) must reuse THIS client. Creating a second client / calling
    ChannelFactory.Init twice in one process initializes the DDS layer twice,
    which crashes (segfault). So there is exactly one client per process.
    """
    return _get_client()


def check_result(result, operation):
    """SDK calls may raise or return an error code; accept only None / zero."""
    if result is not None and result != 0:
        raise RuntimeError(f"{operation} failed: code = {result}")
    return result


def _change_mode(mode, label):
    res = check_result(_get_client().ChangeMode(mode), f"ChangeMode({label})")
    print(f"{label} command sent successfully.")
    return res


def enter_walk_mode():
    """Switch to WALK and wait before sending movement or head commands."""
    from booster_robotics_sdk_python import RobotMode

    res = _change_mode(RobotMode.kWalking, "WALK")
    time.sleep(WALK_MODE_SETTLE_S)
    return res


def enter_prep_mode():
    """Switch the robot into PREP mode when explicitly requested."""
    from booster_robotics_sdk_python import RobotMode

    return _change_mode(RobotMode.kPrepare, "PREP")


def stop():
    """Send a zero-velocity command (stop moving)."""
    return check_result(_get_client().Move(0.0, 0.0, 0.0), "Move(stop)")


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _timed_move(vx, yaw_rate, duration):
    """Use the reference scripts' velocity/time control, always attempting stop.

    Put Move inside try too: an RPC timeout may mean the command was received
    but its acknowledgement was lost. Never retry a nonzero movement blindly.
    Count the Move reply wait in the duration; a late reply gets no extra sleep.
    This cannot interrupt a blocked RPC or remove stop-command delivery delay.
    """
    try:
        client = _get_client()
        deadline = time.monotonic() + duration
        check_result(client.Move(vx, 0.0, yaw_rate), "Move")
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    finally:
        stop()
    time.sleep(SETTLE_S)


def turn_body(angle_rad, yaw_rate=YAW_RATE):
    """Turn in place by angle_rad (>0 = left, <0 = right). Open-loop: velocity x time."""
    angle_rad = _finite(angle_rad, "angle_rad")
    yaw_rate = _finite(yaw_rate, "yaw_rate")
    if yaw_rate <= 0:
        raise ValueError("yaw_rate must be positive")
    if abs(angle_rad) < 1e-3:
        return
    direction = 1.0 if angle_rad > 0 else -1.0
    duration = abs(angle_rad) / yaw_rate
    _finite(duration, "turn duration")
    print(f"turn: {angle_rad:+.3f} rad at {direction * yaw_rate:+.3f} rad/s for {duration:.3f}s")
    _timed_move(0.0, direction * yaw_rate, duration)


def walk_forward(distance_m, speed=WALK_SPEED):
    """Walk straight forward distance_m meters. Open-loop: velocity x time."""
    distance_m = _finite(distance_m, "distance_m")
    speed = _finite(speed, "speed")
    if distance_m < 0 or speed <= 0:
        raise ValueError("distance_m must be nonnegative and speed must be positive")
    if distance_m == 0.0:
        return
    duration = distance_m / speed
    _finite(duration, "walk duration")
    print(f"walk: {distance_m:.3f} m at {speed:.3f} m/s for {duration:.3f}s")
    _timed_move(speed, 0.0, duration)


def look_behind():
    """Turn the body 180 degrees (pi rad) to look behind."""
    turn_body(math.pi)


def move_to_target(point_cam, view):
    """Turn toward the target and take one bounded step (v1, flat-ground).

    v1 ignores the camera mount offset/pitch by request. X = right, Z = forward;
    Y (down) is ignored. The selected view's head yaw is handled explicitly.

    Steps:
      1. turn to face the view direction (head yaw used to capture it:
         front 0, left +1 rad, right -1 rad),
      2. stop, then turn to aim at the point within that view,
      3. walk forward at most MAX_STEP_DISTANCE_M toward the point, then stop.

    A closer point uses its full ground distance. A farther point uses only
    one capped step; the pipeline captures new views and asks the LLM again.
    There is no automatic continuation toward the old target. This bounds the
    requested distance, not measured physical travel (motion is open-loop).

    Args:
        point_cam: np.array([X, Y, Z]) meters in the camera optical frame, or
                   None when there was no usable depth (then we skip the move).
        view:      "front" | "left" | "right".
    """
    if view not in VIEW_TURN:
        raise ValueError(f"Unknown view: {view!r}")
    if point_cam is None:
        print("move_to_target: no depth at the target pixel; skipping this move.")
        return False

    # Validate the whole target before even turning toward the selected view.
    x, _, z = (_finite(value, "camera coordinate") for value in point_cam)
    if z <= 0:
        raise ValueError("Target depth must be positive")

    point_turn = math.atan2(-x, z)       # aim at the point; + turn = left
    distance = _finite(math.hypot(x, z), "target distance")
    step_limit = _finite(MAX_STEP_DISTANCE_M, "MAX_STEP_DISTANCE_M")
    if step_limit <= 0:
        raise ValueError("MAX_STEP_DISTANCE_M must be positive")
    step_distance = min(distance, step_limit)
    _finite(step_distance / WALK_SPEED, "walk duration")
    print(f"target motion: view={view}, view_turn={VIEW_TURN[view]:+.3f} rad, "
          f"point_turn={point_turn:+.3f} rad, target_distance={distance:.3f} m, "
          f"step_distance={step_distance:.3f} m")

    turn_body(VIEW_TURN[view])           # first reproduce the capture head yaw with the body
    turn_body(point_turn)               # then face the selected point in that view
    walk_forward(step_distance)         # one step only; reassess on the next checkpoint
    return True


def _start_gesture(client, gesture):
    """Send the SDK command that starts one gesture; return the SDK reply."""
    from booster_robotics_sdk_python import DanceId, HandAction

    if gesture == "dab":
        return client.Dance(DanceId.kDabbingGesture)
    if gesture == "wave":
        return client.Dance(DanceId.kCheeringGesture)
    if gesture == "handshake":
        return client.Handshake(HandAction.kHandOpen)
    raise ValueError(f"Unknown gesture: {gesture!r}")


def _end_gesture(client, gesture):
    """Send the SDK command that stops one gesture; return the SDK reply."""
    from booster_robotics_sdk_python import DanceId, HandAction

    if gesture == "handshake":
        return client.Handshake(HandAction.kHandClose)
    return client.Dance(DanceId.kStop)    # stops the current upper-body dance or gesture


def perform_gesture(gesture):
    """Play one gesture for GESTURE_SECONDS, then stop it.

    gesture: "nothing" | "dab" | "wave" | "handshake".
    "nothing" sends no command. An unknown name raises before any command.
    The stop command is attempted even on Ctrl+C or an SDK error, as in _timed_move().
    """
    if gesture == "nothing":
        return
    if gesture not in GESTURES:
        raise ValueError(f"Unknown gesture: {gesture!r}")
    client = _get_client()
    try:
        check_result(_start_gesture(client, gesture), f"start gesture {gesture}")
        print(f"gesture: {gesture} for {GESTURE_SECONDS:.1f}s")
        time.sleep(GESTURE_SECONDS)
    finally:
        check_result(_end_gesture(client, gesture), f"stop gesture {gesture}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "prep":
        enter_prep_mode()
    elif command in GESTURES:
        perform_gesture(command)
    else:
        enter_walk_mode()
