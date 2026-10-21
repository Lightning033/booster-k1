import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image

TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"


def msg_to_bgr(msg):
    if msg.encoding.lower() != "rgb8":
        raise ValueError(f"Expected 'rgb8' but got '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = buf.reshape(msg.height, msg.step)
    rgb = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return rgb[:, :, ::-1].copy()


def msg_to_depth(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc == "16uc1":
        rows = buf.view(np.uint16).reshape(msg.height, msg.step // 2)
        return rows[:, : msg.width].astype(np.float32) / 1000.0
    if enc == "32fc1":
        rows = buf.view(np.float32).reshape(msg.height, msg.step // 4)
        return rows[:, : msg.width].astype(np.float32)
    raise ValueError(f"Unsupported depth encoding '{msg.encoding}' (expected 16UC1 or 32FC1)")


class ImageGrabber(Node):
    def __init__(self, timeout_s):
        super().__init__("k1_image_grabber")
        self.timeout_s = timeout_s
        self.image = None
        self.depth = None
        self._color_subscription = self.create_subscription(
            Image, TOPIC, self.on_image, qos_profile_sensor_data)
        self._depth_subscription = self.create_subscription(
            Image, DEPTH_TOPIC, self.on_depth, qos_profile_sensor_data)

    def on_image(self, msg):
        if self.image is not None or len(msg.data) == 0:
            return
        self.image = msg_to_bgr(msg)

    def on_depth(self, msg):
        if self.depth is not None or len(msg.data) == 0:
            return
        self.depth = msg_to_depth(msg)

    def ready(self):
        return self.image is not None and self.depth is not None

    def reset_for_capture(self, deadline):
        self.image = None
        self.depth = None
        for subscription in (self._color_subscription, self._depth_subscription):
            with subscription.handle:
                while time.monotonic() < deadline:
                    if subscription.handle.take_message(
                            subscription.msg_type, subscription.raw) is None:
                        break
                else:
                    raise RuntimeError("Timed out discarding queued camera frames")

    def wait_for_frames(self, label, deadline):
        while not self.ready() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not self.ready():
            raise RuntimeError(f"No RGB+depth frame received for '{label}' within {self.timeout_s}s "
                               f"(RGB received: {self.image is not None}, "
                               f"depth received: {self.depth is not None})")
        if self.depth.shape != self.image.shape[:2]:
            raise ValueError(f"RGB/depth shape mismatch for '{label}': "
                             f"{self.image.shape[:2]} vs {self.depth.shape}")
        return self.image, self.depth

    def capture(self, label):
        deadline = time.monotonic() + self.timeout_s
        self.reset_for_capture(deadline)
        return self.wait_for_frames(label, deadline)


def serve(conn, timeout_s):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    grabber = None
    try:
        try:
            grabber = ImageGrabber(timeout_s)
            grabber.wait_for_frames("startup", time.monotonic() + timeout_s)
        except Exception as exc:
            conn.send((0, "error", f"{type(exc).__name__}: {exc}"))
            return
        conn.send((0, "ok", None))
        while True:
            request = conn.recv()
            if request is None:
                return
            request_id, label = request
            try:
                reply = (request_id, "ok", grabber.capture(label))
            except Exception as exc:
                reply = (request_id, "error", f"{type(exc).__name__}: {exc}")
            conn.send(reply)
    except (EOFError, BrokenPipeError):
        return
    finally:
        try:
            if grabber is not None:
                grabber.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
