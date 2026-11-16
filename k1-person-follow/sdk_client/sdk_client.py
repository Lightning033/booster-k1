"""
Booster K1 SDK connection — one client, one DDS init, per process.

Everything that talks to the robot goes through get_client(). Calling
ChannelFactory.Init twice in one process initializes DDS twice and segfaults,
so there is exactly ONE client per process and it is created on first use.

This module NEVER imports ROS2. ROS2 lives in camera_process/, in its own
process: on this lab PC depth frames go missing when ROS2 and the Booster SDK
share a process.

Mode matters:
    RotateHead and MoveHandEndEffectorV2 are refused unless the robot is in
    kWalking. In kPrepare you get code 400 / 501. ensure_walking() reports the
    mode; it does not change it unless you ask.

Standalone:
    python sdk_client/sdk_client.py             # list interfaces, connect, report mode
    python sdk_client/sdk_client.py walk        # ChangeMode(kWalking)
    python sdk_client/sdk_client.py prep        # ChangeMode(kPrepare)
"""

import os
import time

# The DDS network interface. wlp5s0 is the lab workstation's WiFi card.
# On the robot itself it is wlP5p1s0 (capital P). Override without editing:
#     export K1_NET_INTERFACE=enp0s31f6
INTERFACE = os.environ.get("K1_NET_INTERFACE", "wlp5s0")
DDS_DOMAIN = int(os.environ.get("K1_DDS_DOMAIN", "0"))

WALK_MODE_SETTLE_S = 5.0    # wait after entering WALK before commanding joints

_client = None
_channel_initialized = False


def list_interfaces():
    """Network interface names on this machine, for when the default is wrong."""
    try:
        return sorted(os.listdir("/sys/class/net"))
    except OSError:
        return []


def check_result(result, operation):
    """SDK calls either raise or return a code. Accept only None or zero.

    This build returns None on success for some calls and 0 for others, so both
    are treated as success. Codes seen in practice (from the SDK's rpc/error.hpp):
        100 timeout, 400 bad request, 409 conflict with current state,
        429 too frequent, 500 service error, 501 refused, 502 mode transition
    """
    if result is not None and result != 0:
        raise RuntimeError("%s failed: code = %s" % (operation, result))
    return result


def retry(func, *args, tries=5, wait=1.0):
    """Call func, retrying only on code 100 (RPC timeout over WiFi).

    Any other error is raised immediately: a bad request or a refusal will not
    fix itself, and blindly repeating a motion command is unsafe.
    """
    last = None
    for attempt in range(tries):
        try:
            return check_result(func(*args), getattr(func, "__name__", "sdk call"))
        except RuntimeError as exc:
            last = exc
            if "code = 100" not in str(exc) or attempt == tries - 1:
                raise
            print("  timeout, retry %d/%d ..." % (attempt + 1, tries))
            time.sleep(wait)
    raise last


def get_client():
    """The ONE shared B1LocoClient for this process, created on first call."""
    global _client, _channel_initialized
    if _client is None:
        from booster_robotics_sdk_python import ChannelFactory, B1LocoClient

        if not _channel_initialized:
            ChannelFactory.Instance().Init(DDS_DOMAIN, INTERFACE)
            _channel_initialized = True
        client = B1LocoClient()
        check_result(client.Init(), "Init")
        _client = client
    return _client


def get_mode():
    """The robot's current mode as a string, or None if it cannot be read.

    GetMode's signature differs between SDK builds — some take a response
    object, some return one. Both shapes are tried rather than guessed at.

    The mode field name differs too. This SDK uses a trailing underscore on
    struct members (position_, x_), so the field is mode_ and an earlier version
    of this function, looking only for .mode, silently fell back to the response
    OBJECT. str() of that is '<GetModeResponse object at 0x...>', which contains
    no mode name, so is_walking() read False and ensure_walking() refused to
    start while the robot was in fact walking. Returning None for "could not
    read it" rather than a useless repr is what stops that: an unknown mode is
    not evidence of the wrong mode, and the first head command reports a bad
    mode clearly enough on its own.
    """
    import booster_robotics_sdk_python as sdk

    client = get_client()
    try:
        response = sdk.GetModeResponse()
        client.GetMode(response)
        mode = _mode_field(response)
    except TypeError:
        mode = _mode_field(client.GetMode())
    return None if mode is None else str(mode)


def _mode_field(response):
    """Pull the mode out of whatever shape GetMode handed back.

    Returns None when nothing mode-like is found, so callers can tell "not
    walking" apart from "could not tell".
    """
    if response is None:
        return None
    if isinstance(response, str):
        return response
    for attribute in ("mode_", "mode", "robot_mode_", "robot_mode"):
        if hasattr(response, attribute):
            value = getattr(response, attribute)
            if value is not None and not callable(value):
                return value
    # Last resort: an enum-like attribute whose name mentions a known mode.
    for name in dir(response):
        if name.startswith("_"):
            continue
        try:
            value = getattr(response, name)
        except Exception:
            continue
        if callable(value):
            continue
        text = str(value).lower()
        if any(word in text for word in ("walking", "prepare", "damping",
                                         "custom")):
            return value
    return None


def is_walking(mode=None):
    """True when the reported mode looks like kWalking.

    False for an unreadable mode too, so callers must check for None themselves
    rather than treating "cannot tell" as "not walking".
    """
    if mode is None:
        mode = get_mode()
    if mode is None:
        return False
    return "walking" in str(mode).lower()


def change_mode(name):
    """ChangeMode to 'walk' or 'prep'. Separate from the follow loop on purpose.

    The follow loop never calls this: a mode change can make the K1 shift its
    feet to rebalance, and the person following the robot should be the one who
    decides when the legs become active.
    """
    import booster_robotics_sdk_python as sdk

    modes = {"walk": sdk.RobotMode.kWalking, "prep": sdk.RobotMode.kPrepare}
    if name not in modes:
        raise ValueError("mode must be 'walk' or 'prep', got %r" % (name,))
    retry(get_client().ChangeMode, modes[name])
    print("ChangeMode(%s) sent." % name)
    if name == "walk":
        time.sleep(WALK_MODE_SETTLE_S)


def ensure_walking():
    """Raise a readable error if the robot is not in WALK mode.

    Called before the follow loop so the failure is one clear sentence instead
    of 'code = 400' from the first head command.
    """
    mode = get_mode()
    if mode is not None and not is_walking(mode):
        raise RuntimeError(
            "Robot is in %s, but head and arm commands need kWalking.\n"
            "    Put it in walking mode first (remote, app, or:\n"
            "      python sdk_client/sdk_client.py walk)" % mode)
    return mode


if __name__ == "__main__":
    import sys

    argument = sys.argv[1] if len(sys.argv) > 1 else ""

    if argument in ("walk", "prep"):
        change_mode(argument)
        print("Robot mode is now: %s" % get_mode())
    else:
        print("Interfaces on this machine:")
        for name in list_interfaces():
            marker = "   <== will be used" if name == INTERFACE else ""
            print("  %-10s%s" % (name, marker))
        print("\nConnecting on %r, DDS domain %d ..." % (INTERFACE, DDS_DOMAIN))
        mode = get_mode()
        print("\nConnected. Robot mode: %s" % mode)
        if not is_walking(mode):
            print("\nHead and arm commands need kWalking. To switch:")
            print("  python sdk_client/sdk_client.py walk")
