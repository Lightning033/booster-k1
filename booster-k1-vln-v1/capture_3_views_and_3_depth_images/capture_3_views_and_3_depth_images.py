import multiprocessing
import signal
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot_control.robot_control import get_client, check_result, VIEW_TURN

TIMEOUT_S = 15.0
REPLY_MARGIN_S = 5.0
STARTUP_TIMEOUT_S = 30.0
CLOSE_TIMEOUT_S = 3.0
BODY_SETTLE_S = 3.0
SETTLE_S = 1.0

YAW_LEFT = VIEW_TURN["left"]
YAW_RIGHT = VIEW_TURN["right"]
YAW_FRONT = VIEW_TURN["front"]


def retry(func, *args, tries=5, wait=1.0):
    for i in range(tries):
        try:
            result = func(*args)
            return check_result(result, func.__name__)
        except RuntimeError as e:
            if "code = 100" not in str(e) or i == tries - 1:
                raise
            print(f"Timeout, retry {i + 1}/{tries} ...")
            time.sleep(wait)


class Head:
    def __init__(self, client, pitch=0.0, step=0.10, dt=0.15):
        self.client = client
        self.pitch = pitch
        self.step = step
        self.dt = dt
        self.yaw = YAW_FRONT

    def move_to(self, target_yaw):
        while abs(target_yaw - self.yaw) > 1e-6:
            delta = target_yaw - self.yaw
            next_yaw = self.yaw + max(-self.step, min(self.step, delta))
            retry(self.client.RotateHead, self.pitch, next_yaw)
            self.yaw = next_yaw
            time.sleep(self.dt)

    def front(self):
        if abs(self.yaw - YAW_FRONT) <= 1e-6:
            retry(self.client.RotateHead, self.pitch, YAW_FRONT)
        self.move_to(YAW_FRONT)

    def left(self):
        self.move_to(YAW_LEFT)

    def right(self):
        self.move_to(YAW_RIGHT)


def write_png(path, width, height, rgb_bytes):
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
    height, width = bgr.shape[:2]
    rgb_bytes = np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()
    write_png(path, width, height, rgb_bytes)


def _camera_main(conn, timeout_s):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from capture_3_views_and_3_depth_images.camera_node import serve
    serve(conn, timeout_s)


class CameraProcess:

    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(target=_camera_main, args=(child_conn, TIMEOUT_S),
                                        name="k1_camera", daemon=True)
        self._request_id = 0
        self._process.start()
        child_conn.close()
        try:
            self._receive(STARTUP_TIMEOUT_S)
        except BaseException:
            self.close()
            raise

    def capture(self, label):
        self._request_id += 1
        try:
            self._conn.send((self._request_id, label))
        except OSError as exc:
            raise RuntimeError("Camera process is not running") from exc
        return self._receive(TIMEOUT_S + REPLY_MARGIN_S)

    def _receive(self, timeout_s):
        if not self._conn.poll(timeout_s):
            raise RuntimeError(f"Camera process did not reply within {timeout_s}s")
        try:
            reply_id, status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError("Camera process exited unexpectedly") from None
        if reply_id != self._request_id:
            raise RuntimeError(f"Camera reply {reply_id} does not match request {self._request_id}")
        if status != "ok":
            raise RuntimeError(f"Camera process: {payload}")
        return payload

    def close(self):
        if self._conn.closed:
            return
        try:
            self._conn.send(None)
        except OSError:
            pass
        self._process.join(CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CLOSE_TIMEOUT_S)
        self._conn.close()


def initialize_camera():
    return CameraProcess()


def capture(camera, label=""):
    time.sleep(SETTLE_S)
    image, depth = camera.capture(label)
    print(f"{label}: captured {image.shape[1]}x{image.shape[0]} rgb + depth")
    return image, depth


def get_3_views_and_3_depth_images(camera):
    client = get_client()

    head = Head(client)
    views, depth_images = {}, {}

    try:
        time.sleep(BODY_SETTLE_S)
        head.front()

        head.left()
        views["left"], depth_images["left"] = capture(camera, "left")

        head.right()
        views["right"], depth_images["right"] = capture(camera, "right")

        head.front()
        views["front"], depth_images["front"] = capture(camera, "front")
    finally:
        head.front()
        time.sleep(SETTLE_S)

    return views, depth_images


def close_camera(camera=None):
    if camera is not None:
        camera.close()


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
                stop()
        finally:
            close_camera(camera)
