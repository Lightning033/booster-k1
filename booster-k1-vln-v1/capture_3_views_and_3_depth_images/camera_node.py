"""
Camera node (Booster K1 VLN). Runs ONLY inside the camera process.

capture_3_views_and_3_depth_images.py starts this in its own process, so ROS2
never shares a process with the Booster SDK. On the lab PC, depth frames went
missing when both ran in one process, while both topics worked from separate
processes. Nothing here imports the Booster SDK.

Messages on the pipe (one request at a time):
    startup   camera   -> (0, "ok", None) after one RGB and one depth frame,
                          or (0, "error", text), then exits
    capture   pipeline -> (request_id, label)
              camera   -> (request_id, "ok", (bgr, depth_m))
                          or (request_id, "error", text)
    stop      pipeline -> None      (a closed pipe also stops this process)

Not run directly: test the camera with capture_3_views_and_3_depth_images.py.
"""

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image

TOPIC = "/camera/camera/color/image_raw"
# Depth ALIGNED to the color frame (same resolution + pixel grid as TOPIC), so
# color pixel (u, v) and depth pixel (u, v) are the same point.
# CONFIRM on the robot with `ros2 topic list` and adjust if your K1 differs.
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"


def msg_to_bgr(msg):
    """Convert a ROS rgb8 Image message to a BGR numpy array (H, W, 3)."""
    if msg.encoding.lower() != "rgb8":
        raise ValueError(f"Expected 'rgb8' but got '{msg.encoding}'")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = buf.reshape(msg.height, msg.step)          # step may include padding
    rgb = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return rgb[:, :, ::-1].copy()                     # RGB -> BGR


def msg_to_depth(msg):
    """Convert a ROS depth Image message to a float32 (H, W) array in METERS.

    Handles the two common depth encodings:
      - '16UC1' : uint16 millimeters -> meters
      - '32FC1' : float32 meters (kept as-is)
    """
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
        self.image = None          # BGR array, set once a color frame arrives
        self.depth = None          # float32 meters (H, W), set once a depth frame arrives
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
        """Clear saved frames and discard messages queued during head movement.

        The caller spins this node synchronously; no background executor is used.
        Draining is bounded by the same deadline as the frame wait.
        """
        self.image = None
        self.depth = None
        for subscription in (self._color_subscription, self._depth_subscription):
            # Take directly, as rclpy's executor does. Only a None result proves
            # this queue has no available message; an idle spin does not.
            with subscription.handle:
                while time.monotonic() < deadline:
                    if subscription.handle.take_message(
                            subscription.msg_type, subscription.raw) is None:
                        break
                else:
                    raise RuntimeError("Timed out discarding queued camera frames")

    def wait_for_frames(self, label, deadline):
        """Spin until one RGB and one depth frame arrive; say which one is missing."""
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
        """Discard frames queued during head movement, then wait for fresh ones."""
        deadline = time.monotonic() + self.timeout_s
        self.reset_for_capture(deadline)
        return self.wait_for_frames(label, deadline)


def serve(conn, timeout_s):
    """Answer capture requests until the pipeline says stop or goes away."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # keep Ctrl+C ignored here
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
            except Exception as exc:              # report it; the pipeline aborts the run
                reply = (request_id, "error", f"{type(exc).__name__}: {exc}")
            conn.send(reply)
    except (EOFError, BrokenPipeError):
        return                                     # the pipeline process is gone
    finally:
        try:
            if grabber is not None:
                grabber.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
