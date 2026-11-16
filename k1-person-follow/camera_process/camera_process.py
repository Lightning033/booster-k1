"""
Parent-side handle to the camera process.

Start this BEFORE any Booster SDK call, so a dead camera stops the run before
the robot is sent a single command. The child process loads ROS2 and never the
SDK (see camera_node.py for why).

Use:
    camera = CameraProcess()                  # before any SDK call
    bgr, age_s, depth = camera.latest()       # (None, inf, None) if nothing yet
    ...
    camera.close()                            # always, in a finally block

Standalone, to check the stream and its frame rate:
    python camera_process/camera_process.py            # compressed topic
    python camera_process/camera_process.py --raw      # uncompressed topic
    python camera_process/camera_process.py --depth    # also report depth
"""

import multiprocessing
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STARTUP_TIMEOUT_S = 30.0    # new interpreter + ROS2 discovery + first frame
# The parent must outwait the child, or it gives up at the same instant the
# child is reporting WHY it failed — and that message names the topic and tells
# you what to check. With both at 30 s the only error ever seen was the useless
# "Camera process did not reply within 30s".
STARTUP_REPLY_MARGIN_S = 10.0
REPLY_TIMEOUT_S = 10.0      # a 'latest' reply is a memory read; it should be instant
CLOSE_TIMEOUT_S = 3.0       # wait for a clean exit, then kill


def _camera_main(conn, startup_timeout_s, raw, want_depth):
    """Entry point of the camera process. Loads ROS2 here and nowhere else."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl+C belongs to the parent
    from camera_process.camera_node import serve
    serve(conn, startup_timeout_s, raw=raw, want_depth=want_depth)


class CameraProcess:
    """One request at a time: latest() blocks until the reply to that request
    arrives, so a reply meant for an earlier request is never mistaken for this
    one. Every wait is bounded.
    """

    def __init__(self, raw=False, want_depth=False):
        # "spawn" starts a fresh interpreter, so no SDK state or DDS thread from
        # this process is inherited. Linux would otherwise default to "fork".
        context = multiprocessing.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(
            target=_camera_main,
            args=(child_conn, STARTUP_TIMEOUT_S, raw, want_depth),
            name="k1_person_camera", daemon=True)
        self._request_id = 0
        self._decode_failures = 0
        self._last_seq = -1
        self._is_new = False
        self.info = {}
        self._process.start()
        child_conn.close()        # keep only our end, so an exited child reads as EOF
        try:
            self.info = self._receive(STARTUP_TIMEOUT_S + STARTUP_REPLY_MARGIN_S) or {}
        except BaseException:
            self.close()
            raise

    def latest(self):
        """Newest (bgr, age_s, depth_or_None); (None, inf, None) before the first
        frame. age_s is how old that frame is, which the caller uses to decide
        whether it is still worth acting on.

        Reading twice between arrivals returns the same frame; is_new tells the
        two cases apart without decoding anything.
        """
        import cv2
        import numpy as np

        self._request_id += 1
        try:
            self._conn.send((self._request_id, "latest"))
        except OSError as exc:
            raise RuntimeError("Camera process is not running") from exc
        payload = self._receive(REPLY_TIMEOUT_S)
        if payload is None:
            self._is_new = False
            return None, float("inf"), None
        jpeg, age, depth, seq = payload
        self._is_new = seq != self._last_seq
        self._last_seq = seq
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            # A truncated JPEG is a dropped frame, not a crash. Report it once
            # per run so a steady stream of them is still visible.
            self._decode_failures += 1
            if self._decode_failures == 1:
                print("camera: a JPEG frame failed to decode (dropped)")
            return None, float("inf"), None
        return bgr, age, depth

    def is_new(self):
        """True when the last latest() returned a frame not seen before."""
        return self._is_new

    def _receive(self, timeout_s):
        """Payload of the reply to the current request. None means 'no frame yet'."""
        if not self._conn.poll(timeout_s):
            raise RuntimeError("Camera process did not reply within %.0fs" % timeout_s)
        try:
            reply_id, status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError("Camera process exited unexpectedly") from None
        if reply_id != self._request_id:
            raise RuntimeError("Camera reply %s does not match request %s"
                               % (reply_id, self._request_id))
        if status == "empty":
            return None
        if status != "ok":
            raise RuntimeError("Camera process: %s" % (payload,))
        return payload

    def close(self):
        """Ask the camera process to exit; kill it if it will not."""
        if self._conn.closed:
            return
        try:
            self._conn.send(None)
        except OSError:
            pass                                  # it already exited
        self._process.join(CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CLOSE_TIMEOUT_S)
        self._conn.close()


if __name__ == "__main__":
    raw = "--raw" in sys.argv
    want_depth = "--depth" in sys.argv

    camera = CameraProcess(raw=raw, want_depth=want_depth)
    try:
        print("Streaming from %s" % camera.info.get("topic", "?"))
        last = None
        for _ in range(5):
            new_frames = 0
            window_start = time.monotonic()
            while time.monotonic() - window_start < 1.0:
                bgr, age, depth = camera.latest()
                if bgr is not None and camera.is_new():
                    new_frames += 1
                    last = (bgr, age, depth)
                time.sleep(0.02)
            elapsed = time.monotonic() - window_start
            if last is None:
                print("no frame decoded yet")
                continue
            height, width = last[0].shape[:2]
            print("%5.1f Hz   newest frame age %5.0f ms   %dx%d   depth %s"
                  % (new_frames / elapsed, last[1] * 1000, width, height,
                     "yes" if last[2] is not None else "no"))
        print("\nAges above ~300 ms mean the head will lag behind the person.")
    finally:
        camera.close()
