# =============================================================================
# capture_3_views_and_3_depth_images.py
# Take 3 RGB + depth captures from the Booster K1 camera: LEFT, RIGHT, FRONT.
#
# The camera subscriptions run in a SEPARATE process (camera_node.py) that
# never loads the Booster SDK: on the lab PC, depth frames went missing when
# the SDK and ROS2 shared one process. This file runs in the pipeline process.
# It moves the head, then asks the camera process for one fresh RGB + depth
# pair and waits for the reply, so only one side acts at a time.
#
# Two ways to use it:
#   1. Imported (in the pipeline):
#          camera = initialize_camera()  # BEFORE any Booster SDK calls
#          enter_walk_mode()
#          views, depth_images = get_3_views_and_3_depth_images(camera)
#          # At the end of the run: close_camera(camera)
#      -> returns two dicts keyed by "front"/"left"/"right":
#           views        = BGR numpy arrays (OpenCV convention)
#           depth_images = float32 depth maps in METERS, aligned to each view
#         Nothing is saved; the caller decides where to store them.
#
#   2. Standalone (independent testing):
#          python capture_3_views_and_3_depth_images.py
#      -> captures the 3 views (+ depth) and SAVES the RGB as PNGs here.
#
# Note on channel order: the camera streams rgb8; we return BGR so the images
# drop straight into the rest of the pipeline (grid overlay + cv2.imwrite),
# which all use OpenCV's BGR convention. Depth is returned as float32 meters.
# =============================================================================

import multiprocessing
import signal
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

# Reuse the ONE shared Booster client from robot_control so the SDK / DDS is
# initialized exactly once per process. Creating a second client (or calling
# ChannelFactory.Init twice) initializes DDS twice and crashes with a segfault.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot_control.robot_control import get_client, check_result, VIEW_TURN

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
TIMEOUT_S = 15.0          # camera process: wait for one RGB + depth pair
REPLY_MARGIN_S = 5.0      # extra time for a reply to cross the pipe
STARTUP_TIMEOUT_S = 30.0  # new interpreter + ROS2 discovery + first frames
CLOSE_TIMEOUT_S = 3.0     # wait for the camera process to exit, then kill it
BODY_SETTLE_S = 3.0       # wait at each checkpoint before any head movement
SETTLE_S = 1.0            # wait after head move, so the photo is not blurry

YAW_LEFT = VIEW_TURN["left"]      # one source for capture yaw and body alignment
YAW_RIGHT = VIEW_TURN["right"]
YAW_FRONT = VIEW_TURN["front"]


# -----------------------------------------------------------------------------
# RETRY: robot calls sometimes time out over WiFi (code 100).
# -----------------------------------------------------------------------------
def retry(func, *args, tries=5, wait=1.0):
    for i in range(tries):
        try:
            result = func(*args)
            return check_result(result, func.__name__)
        except RuntimeError as e:
            if "code = 100" not in str(e) or i == tries - 1:
                raise                      # other error, or last try: stop
            print(f"Timeout, retry {i + 1}/{tries} ...")
            time.sleep(wait)


# -----------------------------------------------------------------------------
# HEAD: remembers where the head is, and moves it in small steps.
# -----------------------------------------------------------------------------
class Head:
    def __init__(self, client, pitch=0.0, step=0.10, dt=0.15):
        self.client = client
        self.pitch = pitch
        self.step = step
        self.dt = dt
        self.yaw = YAW_FRONT       # assume head starts facing front

    def move_to(self, target_yaw):
        """Walk the head from its current yaw to target_yaw."""
        while abs(target_yaw - self.yaw) > 1e-6:
            delta = target_yaw - self.yaw
            next_yaw = self.yaw + max(-self.step, min(self.step, delta))
            retry(self.client.RotateHead, self.pitch, next_yaw)
            self.yaw = next_yaw    # update only after the command is sent
            time.sleep(self.dt)

    def front(self):
        # A new Head object cannot know the physical head's starting yaw.
        # Send center even when the locally remembered yaw is already zero.
        if abs(self.yaw - YAW_FRONT) <= 1e-6:
            retry(self.client.RotateHead, self.pitch, YAW_FRONT)
        self.move_to(YAW_FRONT)

    def left(self):
        self.move_to(YAW_LEFT)

    def right(self):
        self.move_to(YAW_RIGHT)


# -----------------------------------------------------------------------------
# IMAGE SAVE (standalone test only)
# -----------------------------------------------------------------------------
def write_png(path, width, height, rgb_bytes):
    """Minimal, dependency-free PNG writer (expects RGB bytes)."""
    row_len = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += rgb_bytes[y * row_len:(y + 1) * row_len]

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def save_png(path, bgr):
    """Save a BGR numpy array to a PNG file."""
    height, width = bgr.shape[:2]
    rgb_bytes = np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()   # BGR -> RGB
    write_png(path, width, height, rgb_bytes)


# -----------------------------------------------------------------------------
# CAMERA PROCESS: ROS2 runs in its own process (camera_node.py).
# -----------------------------------------------------------------------------
def _camera_main(conn, timeout_s):
    """Entry point of the camera process. Loads ROS2, never the Booster SDK."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl+C belongs to the pipeline process
    from capture_3_views_and_3_depth_images.camera_node import serve   # ROS2 loads here only
    serve(conn, timeout_s)


class CameraProcess:
    """Pipeline-side handle to the camera process.

    One request at a time: capture() sends a label and blocks until the reply
    to that request arrives, so the head never moves during a capture. Every
    wait is bounded; any failure raises and the caller runs its cleanup.
    """

    def __init__(self):
        # "spawn" starts a fresh interpreter, so nothing from this process
        # (SDK state, DDS threads) is copied. Linux defaults to "fork".
        context = multiprocessing.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(target=_camera_main, args=(child_conn, TIMEOUT_S),
                                        name="k1_camera", daemon=True)
        self._request_id = 0
        self._process.start()
        child_conn.close()          # keep only our end, so an exited process reads as EOF
        try:
            self._receive(STARTUP_TIMEOUT_S)    # ready once RGB and depth both arrived
        except BaseException:
            self.close()
            raise

    def capture(self, label):
        """Return one fresh (BGR image, depth in meters) pair for the current head yaw."""
        self._request_id += 1
        try:
            self._conn.send((self._request_id, label))
        except OSError as exc:
            raise RuntimeError("Camera process is not running") from exc
        return self._receive(TIMEOUT_S + REPLY_MARGIN_S)

    def _receive(self, timeout_s):
        """Return the payload of the reply to the current request, or raise."""
        if not self._conn.poll(timeout_s):
            raise RuntimeError(f"Camera process did not reply within {timeout_s}s")
        try:
            reply_id, status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError("Camera process exited unexpectedly") from None
        if reply_id != self._request_id:        # never accept a reply meant for another view
            raise RuntimeError(f"Camera reply {reply_id} does not match request {self._request_id}")
        if status != "ok":
            raise RuntimeError(f"Camera process: {payload}")
        return payload

    def close(self):
        """Ask the camera process to exit; kill it if it does not."""
        if self._conn.closed:
            return
        try:
            self._conn.send(None)
        except OSError:                          # it already exited
            pass
        self._process.join(CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CLOSE_TIMEOUT_S)
        self._conn.close()


def initialize_camera():
    """Start the camera process; returns once RGB and depth both arrive.

    Call BEFORE any Booster SDK call, so a missing stream stops the run
    before the robot receives any command.
    """
    return CameraProcess()


def capture(camera, label=""):
    """Let the head settle, then get one fresh RGB + depth pair."""
    time.sleep(SETTLE_S)
    image, depth = camera.capture(label)
    print(f"{label}: captured {image.shape[1]}x{image.shape[0]} rgb + depth")
    return image, depth


# -----------------------------------------------------------------------------
# MAIN FUNCTION  (returns images in memory; does NOT save)
# -----------------------------------------------------------------------------
def get_3_views_and_3_depth_images(camera):
    """Capture left/right/front RGB + depth. Returns (views, depth_images).

        views        = {"front": BGR, "left": BGR, "right": BGR}
        depth_images = {"front": depth_m, "left": depth_m, "right": depth_m}

    views[name] and depth_images[name] are an aligned RGB + depth pair captured
    at the same head yaw with the head stationary. The two topic subscriptions
    are not timestamp-synchronized. The caller must start the camera with
    initialize_camera() BEFORE entering WALK mode or making any SDK calls.

    Uses the shared Booster client from robot_control (ChannelFactory is
    initialized once per process). The same camera process is reused for
    every view and navigation step, then closed at the end of the run.
    """
    client = get_client()             # shared client; ChannelFactory.Init happens once

    head = Head(client)
    views, depth_images = {}, {}

    try:
        time.sleep(BODY_SETTLE_S)          # let the body settle before moving the head
        head.front()                      # make sure head faces front

        head.left()
        views["left"], depth_images["left"] = capture(camera, "left")

        head.right()                      # goes left -> front -> right
        views["right"], depth_images["right"] = capture(camera, "right")

        head.front()
        views["front"], depth_images["front"] = capture(camera, "front")
    finally:
        head.front()                      # head back to front (do NOT close the camera here)
        time.sleep(SETTLE_S)               # let the head settle before body movement

    return views, depth_images


def close_camera(camera=None):
    """Stop the camera process at the end of a run (None is allowed)."""
    if camera is not None:
        camera.close()


# -----------------------------------------------------------------------------
# STANDALONE: capture and SAVE to the current folder (for independent testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from robot_control.robot_control import enter_walk_mode, stop

    camera = None
    robot_started = False
    try:
        camera = initialize_camera()
        robot_started = True
        enter_walk_mode()
        stop()
        views, depth_images = get_3_views_and_3_depth_images(camera)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for label, image in views.items():
            out_path = Path.cwd() / f"k1_{label}_{stamp}.png"
            save_png(out_path, image)
            d = depth_images[label]
            print(f"saved {out_path}  (+ depth {d.shape[1]}x{d.shape[0]} m)")
    finally:
        try:
            if robot_started:
                stop()                    # remain in WALK mode
        finally:
            close_camera(camera)
