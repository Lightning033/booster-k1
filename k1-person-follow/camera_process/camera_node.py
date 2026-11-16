"""
Camera node — runs ONLY inside the camera process. Never imports the Booster SDK.

Why its own process: on this lab PC, depth frames go missing when ROS2 and the
Booster SDK share one process, even with ROS2 started first. Both topics work
from separate processes. This is the same split the VLN project uses.

Which topic it uses is DISCOVERED, not assumed. An earlier version hard-coded
the compressed topic, the CompressedImage type and qos_profile_sensor_data, and
received nothing on a robot where `ros2 topic hz` on that same topic reported
30 Hz. `ros2 topic hz` works on any topic precisely because it looks the type
and QoS up first. So this does too:

  1. For each candidate topic, in order, ask the graph what is publishing:
     its message TYPE and its QoS (reliability, durability, history, depth).
  2. Subscribe with a QoS matched to the publisher's and the type it reported.
  3. Give it PROBE_SECONDS to deliver a frame. If it does, keep it.
  4. If it does not, tear that subscription down and try the next candidate.
  5. If none deliver, raise an error that prints what each candidate offered,
     so the failure is diagnosable from one run instead of several.

Candidate order is compressed first, then raw. The robot publishes JPEG on the
compressed topic already, so nothing is re-encoded and the WiFi carries roughly
a twentieth of the raw bytes — raw 1280x720 at 30 Hz would be far past this
link's ceiling. raw=True skips straight to the uncompressed topic.

Frames cross the pipe as JPEG bytes plus their age, so a 1280x720 frame is
about 100 KB rather than 2.7 MB pickled.

Messages on the pipe (one request at a time):
    startup   camera   -> (0, "ok", info_dict) once a first frame arrives,
                          or (0, "error", text), then exits
    latest    parent   -> (request_id, "latest")
              camera   -> (request_id, "ok", (jpeg_bytes, age_s, depth_or_None, seq))
                          or (request_id, "empty", None)   no frame yet
                          or (request_id, "error", text)
              seq counts colour frames, so the parent can tell a new frame from
              the same one read twice.
    stop      parent   -> None     (a closed pipe also stops this process)

Not run directly: use camera_process.py, which starts this.
"""

import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image

COLOR_TOPIC = "/camera/camera/color/image_raw"
COLOR_COMPRESSED_TOPIC = "/camera/camera/color/image_raw/compressed"
# Depth ALIGNED to colour, so colour pixel (u, v) and depth pixel (u, v) are the
# same point. Depth is optional throughout: nothing in the follow loop needs it.
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"

JPEG_QUALITY = 85
PROBE_SECONDS = 4.0        # per candidate: long enough for discovery at 14-30 Hz
DISCOVERY_SECONDS = 3.0    # wait for the graph to report publishers at all

_TYPE_BY_NAME = {
    "sensor_msgs/msg/CompressedImage": CompressedImage,
    "sensor_msgs/msg/Image": Image,
}


# -----------------------------------------------------------------------------
# Conversions
# -----------------------------------------------------------------------------
def msg_to_bgr(msg):
    """A ROS rgb8/bgr8 Image message as a BGR numpy array (H, W, 3)."""
    encoding = msg.encoding.lower()
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError("Expected rgb8 or bgr8 but got %r" % (msg.encoding,))
    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = buffer.reshape(msg.height, msg.step)        # step may include padding
    image = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()                # RGB -> BGR
    return image.copy()


def msg_to_depth(msg):
    """A ROS depth Image message as float32 (H, W) in METRES.

    16UC1 is uint16 millimetres; 32FC1 is already metres.
    """
    encoding = msg.encoding.lower()
    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding == "16uc1":
        rows = buffer.view(np.uint16).reshape(msg.height, msg.step // 2)
        return rows[:, : msg.width].astype(np.float32) / 1000.0
    if encoding == "32fc1":
        rows = buffer.view(np.float32).reshape(msg.height, msg.step // 4)
        return rows[:, : msg.width].astype(np.float32)
    raise ValueError("Unsupported depth encoding %r (expected 16UC1 or 32FC1)"
                     % (msg.encoding,))


# -----------------------------------------------------------------------------
# Discovery: match the publisher instead of assuming
# -----------------------------------------------------------------------------
def qos_matching(publisher_qos, depth_fallback=5):
    """A subscriber QoS that can receive from a publisher with publisher_qos.

    Reliability and durability are copied rather than guessed. The spec says a
    weaker subscriber is compatible with a stronger publisher, but DDS builds
    differ in how they treat the edges, and copying removes the question. History
    is forced to KEEP_LAST with a small depth: a follower wants the newest frame,
    never a backlog of old ones.
    """
    if publisher_qos is None:
        return qos_profile_sensor_data
    depth = getattr(publisher_qos, "depth", 0) or depth_fallback
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, min(depth, 10)),
        reliability=getattr(publisher_qos, "reliability",
                            ReliabilityPolicy.BEST_EFFORT),
        durability=getattr(publisher_qos, "durability",
                           DurabilityPolicy.VOLATILE),
    )


def describe_publishers(node, topic, settle_s=DISCOVERY_SECONDS):
    """[(type_name, qos_profile)] for whatever publishes topic, after waiting.

    The graph needs a moment after a node is created before it reports anything,
    so an immediate empty answer means "not yet", not "nothing there".
    """
    deadline = time.monotonic() + settle_s
    while True:
        try:
            infos = node.get_publishers_info_by_topic(topic)
        except Exception:
            infos = []
        if infos or time.monotonic() >= deadline:
            return [(info.topic_type, info.qos_profile) for info in infos]
        time.sleep(0.2)


def message_type_for(type_name, topic):
    """The message class to subscribe with.

    The publisher's reported type wins. Falling back to the topic name only
    matters when the graph says nothing, and then compressed topics are
    CompressedImage and everything else is Image.
    """
    if type_name in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[type_name]
    return CompressedImage if topic.endswith("/compressed") else Image


def qos_text(qos):
    """QoS as one readable line, for the diagnostics in an error message."""
    if qos is None:
        return "unknown"
    return "reliability=%s durability=%s history=%s depth=%s" % (
        getattr(qos, "reliability", "?"), getattr(qos, "durability", "?"),
        getattr(qos, "history", "?"), getattr(qos, "depth", "?"))


# -----------------------------------------------------------------------------
# The node
# -----------------------------------------------------------------------------
class _StreamNode(Node):
    """Keeps only the NEWEST frame. Old frames are of no interest to a follower:
    acting on a stale picture means turning the head towards where someone was.

    One colour subscription at a time. try_candidate() swaps it for the next
    candidate topic, so a topic that is advertised but silent costs PROBE_SECONDS
    rather than the whole run.
    """

    def __init__(self, raw=False, want_depth=False):
        super().__init__("k1_person_follow_camera")
        self._lock = threading.Lock()
        self._jpeg = None
        self._stamp = 0.0
        self._depth = None
        self._depth_stamp = 0.0
        self.color_count = 0
        self.depth_count = 0
        self.want_depth = want_depth

        self.color_topic = None
        self.color_type = None          # the type actually subscribed with
        self.publisher_type = None      # what the graph reported; None = nobody
        self.color_qos = None
        self._color_subscription = None
        self.attempts = []          # [(topic, publisher_type, qos_text, frames)]

        self.candidates = [COLOR_TOPIC] if raw else [COLOR_COMPRESSED_TOPIC,
                                                     COLOR_TOPIC]
        if want_depth:
            depth_publishers = describe_publishers(self, DEPTH_TOPIC)
            depth_qos = qos_matching(depth_publishers[0][1]
                                     if depth_publishers else None)
            self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, depth_qos)

    # -- subscription management ---------------------------------------------
    def try_candidate(self, topic):
        """Subscribe to topic with a QoS and type matched to its publisher.

        Returns the (type_name, qos) actually used. The previous colour
        subscription, if any, is destroyed first: two image subscriptions on a
        WiFi link is bandwidth spent on a topic already known to be silent.
        """
        if self._color_subscription is not None:
            self.destroy_subscription(self._color_subscription)
            self._color_subscription = None

        publishers = describe_publishers(self, topic)
        type_name = publishers[0][0] if publishers else None
        publisher_qos = publishers[0][1] if publishers else None
        message_type = message_type_for(type_name, topic)
        qos = qos_matching(publisher_qos)

        handler = (self._on_compressed_color if message_type is CompressedImage
                   else self._on_raw_color)
        self._color_subscription = self.create_subscription(
            message_type, topic, handler, qos)
        self.color_topic = topic
        # Kept apart on purpose: publisher_type stays None when the graph reports
        # nobody publishing, so a failure can say "NOTHING PUBLISHING" instead of
        # naming the type we guessed and subscribed with anyway.
        self.publisher_type = type_name
        self.color_type = type_name or message_type.__name__
        self.color_qos = qos_text(publisher_qos)
        return self.color_type, self.color_qos

    def record_attempt(self, frames):
        self.attempts.append((self.color_topic, self.publisher_type,
                              self.color_qos, frames))

    # -- callbacks ------------------------------------------------------------
    def _store_color(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._stamp = time.monotonic()
            self.color_count += 1

    def _on_compressed_color(self, msg):
        if len(msg.data) == 0:
            return
        self._store_color(bytes(msg.data))        # already JPEG; pass straight through

    def _on_raw_color(self, msg):
        if len(msg.data) == 0:
            return
        import cv2                                # only needed on the raw path

        ok, encoded = cv2.imencode(".jpg", msg_to_bgr(msg),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            self._store_color(encoded.tobytes())

    def _on_depth(self, msg):
        if len(msg.data) == 0:
            return
        depth = msg_to_depth(msg)
        with self._lock:
            self._depth = depth
            self._depth_stamp = time.monotonic()
            self.depth_count += 1

    # -- reading --------------------------------------------------------------
    def snapshot(self):
        """(jpeg_bytes, age_s, depth_or_None, seq), or (None, inf, None, 0) before
        the first frame. seq counts colour frames received, so the parent can tell
        a genuinely new frame from the same one read twice. Depth is returned only
        when it is within 0.5 s of the colour frame: a fresh picture paired with an
        old depth map is a lie.
        """
        with self._lock:
            if self._jpeg is None:
                return None, float("inf"), None, 0
            now = time.monotonic()
            depth = None
            if self._depth is not None and abs(self._stamp - self._depth_stamp) < 0.5:
                depth = self._depth
            return self._jpeg, now - self._stamp, depth, self.color_count

    def counts(self):
        with self._lock:
            return self.color_count, self.depth_count


def failure_report(node):
    """Why no candidate worked, with what each one offered. One run, all the facts."""
    lines = ["No camera frames from any candidate topic."]
    for topic, publisher_type, qos, frames in node.attempts:
        lines.append("  %s" % topic)
        lines.append("      publisher type: %s"
                     % (publisher_type or "NOTHING PUBLISHING"))
        lines.append("      publisher QoS : %s" % qos)
        lines.append("      frames in %.0fs: %d" % (PROBE_SECONDS, frames))
    lines.append("  Check with: ros2 topic hz <topic>")
    lines.append("  If a topic shows frames there but 0 here, send this whole block.")
    return "\n".join(lines)


def acquire_stream(node, probe_seconds=PROBE_SECONDS):
    """Try each candidate until one delivers a frame. True when one did."""
    for topic in node.candidates:
        type_name, qos = node.try_candidate(topic)
        print("camera: trying %s (type %s, publisher %s)" % (topic, type_name, qos),
              flush=True)
        before = node.counts()[0]
        deadline = time.monotonic() + probe_seconds
        while time.monotonic() < deadline:
            if node.counts()[0] > before:
                print("camera: using %s" % topic, flush=True)
                return True
            time.sleep(0.05)
        node.record_attempt(node.counts()[0] - before)
        print("camera: no frames on %s" % topic, flush=True)
    return False


def serve(conn, startup_timeout_s, raw=False, want_depth=False):
    """Answer 'latest' requests until the parent says stop or goes away."""
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl+C is the parent's
    node = None
    executor = None
    spin_thread = None
    try:
        try:
            node = _StreamNode(raw=raw, want_depth=want_depth)
            from rclpy.executors import SingleThreadedExecutor

            executor = SingleThreadedExecutor()
            executor.add_node(node)
            spin_thread = threading.Thread(target=executor.spin, daemon=True,
                                           name="k1_camera_spin")
            spin_thread.start()

            if not acquire_stream(node):
                raise RuntimeError(failure_report(node))
        except Exception as exc:
            conn.send((0, "error", "%s: %s" % (type(exc).__name__, exc)))
            return

        conn.send((0, "ok", {"topic": node.color_topic, "type": node.color_type,
                             "depth": want_depth}))

        while True:
            request = conn.recv()
            if request is None:
                return
            request_id = request[0]
            try:
                jpeg, age, depth, seq = node.snapshot()
                if jpeg is None:
                    conn.send((request_id, "empty", None))
                else:
                    conn.send((request_id, "ok", (jpeg, age, depth, seq)))
            except Exception as exc:
                conn.send((request_id, "error", "%s: %s" % (type(exc).__name__, exc)))
    except (EOFError, BrokenPipeError):
        return                                   # the parent process is gone
    finally:
        try:
            if executor is not None:
                executor.shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=2.0)
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
