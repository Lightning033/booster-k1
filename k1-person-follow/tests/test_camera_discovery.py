"""
The camera's topic/QoS discovery, with ROS2 stubbed out.

camera_node imports rclpy at module level, which is absent off the robot, so
this installs stand-in modules in sys.modules and then imports the REAL
camera_node against them. That way the actual discovery logic is tested, not a
copy of it.

Why this matters: the first version hard-coded the compressed topic, the
CompressedImage type and qos_profile_sensor_data. On the real robot it received
zero frames from a topic where `ros2 topic hz` reported 30 Hz. These tests cover
the adaptive behaviour that replaced it.
"""

import sys
import types
import unittest

from support import PROJECT_ROOT  # noqa: F401


# -----------------------------------------------------------------------------
# ROS2 stand-ins
# -----------------------------------------------------------------------------
class StubEnum:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, StubEnum) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class StubReliability:
    RELIABLE = StubEnum("RELIABLE")
    BEST_EFFORT = StubEnum("BEST_EFFORT")


class StubDurability:
    TRANSIENT_LOCAL = StubEnum("TRANSIENT_LOCAL")
    VOLATILE = StubEnum("VOLATILE")


class StubHistory:
    KEEP_LAST = StubEnum("KEEP_LAST")
    KEEP_ALL = StubEnum("KEEP_ALL")


class StubQoSProfile:
    def __init__(self, depth=10, reliability=None, durability=None, history=None):
        self.depth = depth
        self.reliability = reliability or StubReliability.RELIABLE
        self.durability = durability or StubDurability.VOLATILE
        self.history = history or StubHistory.KEEP_LAST


class StubNode:
    """Enough of rclpy.node.Node for _StreamNode to be built and driven."""

    def __init__(self, name=None):
        self.node_name = name
        self.subscriptions_made = []      # (msg_type, topic, qos)
        self.destroyed = []
        self.publisher_table = {}         # topic -> [(type_name, qos)]

    def create_subscription(self, msg_type, topic, handler, qos):
        handle = ("subscription", msg_type, topic, handler, qos)
        self.subscriptions_made.append((msg_type, topic, qos))
        return handle

    def destroy_subscription(self, handle):
        self.destroyed.append(handle)

    def get_publishers_info_by_topic(self, topic):
        entries = self.publisher_table.get(topic, [])
        return [types.SimpleNamespace(topic_type=t, qos_profile=q) for t, q in entries]


class StubImage:
    pass


class StubCompressedImage:
    pass


def install_ros_stubs():
    """Put stand-in ROS2 modules in sys.modules. Returns a restore callable."""
    saved = {name: sys.modules.get(name) for name in (
        "rclpy", "rclpy.node", "rclpy.qos", "rclpy.signals", "rclpy.executors",
        "sensor_msgs", "sensor_msgs.msg", "camera_process.camera_node")}

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda **_kwargs: None
    rclpy.shutdown = lambda: None
    rclpy.ok = lambda: False

    node_module = types.ModuleType("rclpy.node")
    node_module.Node = StubNode

    qos_module = types.ModuleType("rclpy.qos")
    qos_module.QoSProfile = StubQoSProfile
    qos_module.ReliabilityPolicy = StubReliability
    qos_module.DurabilityPolicy = StubDurability
    qos_module.HistoryPolicy = StubHistory
    qos_module.qos_profile_sensor_data = StubQoSProfile(
        depth=5, reliability=StubReliability.BEST_EFFORT,
        durability=StubDurability.VOLATILE)

    signals_module = types.ModuleType("rclpy.signals")
    signals_module.SignalHandlerOptions = types.SimpleNamespace(NO="NO")

    executors_module = types.ModuleType("rclpy.executors")
    executors_module.SingleThreadedExecutor = object

    sensor_msgs = types.ModuleType("sensor_msgs")
    msg_module = types.ModuleType("sensor_msgs.msg")
    msg_module.Image = StubImage
    msg_module.CompressedImage = StubCompressedImage
    sensor_msgs.msg = msg_module

    rclpy.node = node_module
    rclpy.qos = qos_module
    rclpy.signals = signals_module
    rclpy.executors = executors_module

    sys.modules.update({
        "rclpy": rclpy, "rclpy.node": node_module, "rclpy.qos": qos_module,
        "rclpy.signals": signals_module, "rclpy.executors": executors_module,
        "sensor_msgs": sensor_msgs, "sensor_msgs.msg": msg_module,
    })
    # The package object caches `camera_node` as an attribute, so popping only
    # the submodule still hands back the module built against the PREVIOUS
    # stubs. Drop the package too.
    sys.modules.pop("camera_process.camera_node", None)
    sys.modules.pop("camera_process", None)

    def restore():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return restore


class DiscoveryTestCase(unittest.TestCase):
    def setUp(self):
        self.restore = install_ros_stubs()
        from camera_process import camera_node

        self.cn = camera_node

    def tearDown(self):
        self.restore()


# -----------------------------------------------------------------------------
class TestQoSMatching(DiscoveryTestCase):
    def test_reliability_is_copied_not_guessed(self):
        """The spec says a weaker subscriber is compatible, but DDS builds differ
        at the edges. Copying removes the question."""
        publisher = StubQoSProfile(depth=5, reliability=StubReliability.RELIABLE,
                                   durability=StubDurability.VOLATILE)
        matched = self.cn.qos_matching(publisher)
        self.assertEqual(matched.reliability, StubReliability.RELIABLE)

    def test_durability_is_copied(self):
        publisher = StubQoSProfile(
            depth=5, reliability=StubReliability.RELIABLE,
            durability=StubDurability.TRANSIENT_LOCAL)
        matched = self.cn.qos_matching(publisher)
        self.assertEqual(matched.durability, StubDurability.TRANSIENT_LOCAL)

    def test_best_effort_publisher_is_matched_too(self):
        publisher = StubQoSProfile(depth=1,
                                   reliability=StubReliability.BEST_EFFORT,
                                   durability=StubDurability.VOLATILE)
        matched = self.cn.qos_matching(publisher)
        self.assertEqual(matched.reliability, StubReliability.BEST_EFFORT)

    def test_history_is_always_keep_last(self):
        """A follower wants the newest frame, never a backlog of old ones."""
        publisher = StubQoSProfile(depth=500, history=StubHistory.KEEP_ALL)
        matched = self.cn.qos_matching(publisher)
        self.assertEqual(matched.history, StubHistory.KEEP_LAST)
        self.assertLessEqual(matched.depth, 10, "a deep queue would serve stale frames")

    def test_depth_has_a_floor(self):
        matched = self.cn.qos_matching(StubQoSProfile(depth=0))
        self.assertGreaterEqual(matched.depth, 1)

    def test_no_publisher_falls_back_to_sensor_data(self):
        self.assertIs(self.cn.qos_matching(None),
                      sys.modules["rclpy.qos"].qos_profile_sensor_data)


class TestMessageType(DiscoveryTestCase):
    def test_the_publishers_reported_type_wins(self):
        """Assuming the type is how the first version failed."""
        self.assertIs(
            self.cn.message_type_for("sensor_msgs/msg/CompressedImage",
                                     "/anything/at/all"),
            StubCompressedImage)
        self.assertIs(
            self.cn.message_type_for("sensor_msgs/msg/Image", "/x/compressed"),
            StubImage)

    def test_unknown_type_falls_back_to_the_topic_name(self):
        self.assertIs(self.cn.message_type_for(None, "/camera/x/compressed"),
                      StubCompressedImage)
        self.assertIs(self.cn.message_type_for(None, "/camera/x/image_raw"),
                      StubImage)

    def test_an_unexpected_type_string_still_resolves(self):
        self.assertIs(self.cn.message_type_for("some/other/Type", "/x/compressed"),
                      StubCompressedImage)


class TestCandidateOrder(DiscoveryTestCase):
    def test_compressed_is_tried_first(self):
        """Raw 1280x720 at 30 Hz is far past this WiFi link's ceiling."""
        node = self.cn._StreamNode(raw=False)
        self.assertEqual(node.candidates,
                         [self.cn.COLOR_COMPRESSED_TOPIC, self.cn.COLOR_TOPIC])

    def test_raw_flag_skips_compressed(self):
        node = self.cn._StreamNode(raw=True)
        self.assertEqual(node.candidates, [self.cn.COLOR_TOPIC])


class TestAcquireStream(DiscoveryTestCase):
    def _node(self, publisher_table, delivering=None):
        """A node whose colour count rises only for topics in `delivering`.

        The frame has to land AFTER acquire_stream has read its baseline count,
        which it does once the subscription exists. Incrementing inside
        try_candidate was too early: the increment was already in the baseline,
        so the probe then waited for a count that could never rise.
        """
        import threading as _threading

        node = self.cn._StreamNode(raw=False)
        node.publisher_table = publisher_table
        self._timers = getattr(self, "_timers", [])

        original = node.try_candidate

        def try_candidate(topic):
            result = original(topic)
            if delivering and topic in delivering:
                def deliver():
                    node.color_count += 1      # as a real frame callback would
                timer = _threading.Timer(0.05, deliver)
                timer.daemon = True
                timer.start()
                self._timers.append(timer)
            return result

        node.try_candidate = try_candidate
        return node

    def tearDown(self):
        for timer in getattr(self, "_timers", []):
            timer.cancel()
        super().tearDown()

    def test_it_keeps_the_first_topic_that_delivers(self):
        table = {self.cn.COLOR_COMPRESSED_TOPIC:
                 [("sensor_msgs/msg/CompressedImage", StubQoSProfile())]}
        node = self._node(table, delivering={self.cn.COLOR_COMPRESSED_TOPIC})
        self.assertTrue(self.cn.acquire_stream(node, probe_seconds=0.2))
        self.assertEqual(node.color_topic, self.cn.COLOR_COMPRESSED_TOPIC)
        self.assertEqual(node.attempts, [], "a working topic is not a failed attempt")

    def test_a_silent_topic_falls_through_to_the_next(self):
        """Exactly the real failure: compressed is advertised but gives nothing."""
        table = {
            self.cn.COLOR_COMPRESSED_TOPIC:
                [("sensor_msgs/msg/CompressedImage", StubQoSProfile())],
            self.cn.COLOR_TOPIC: [("sensor_msgs/msg/Image", StubQoSProfile())],
        }
        node = self._node(table, delivering={self.cn.COLOR_TOPIC})
        self.assertTrue(self.cn.acquire_stream(node, probe_seconds=0.2))
        self.assertEqual(node.color_topic, self.cn.COLOR_TOPIC,
                         "should have fallen back to raw")
        self.assertEqual(len(node.attempts), 1, "the silent topic should be recorded")

    def test_all_silent_returns_false(self):
        table = {self.cn.COLOR_COMPRESSED_TOPIC: [], self.cn.COLOR_TOPIC: []}
        node = self._node(table, delivering=set())
        self.assertFalse(self.cn.acquire_stream(node, probe_seconds=0.1))
        self.assertEqual(len(node.attempts), 2, "both candidates should be recorded")

    def test_the_old_subscription_is_destroyed_before_the_next(self):
        """Two image subscriptions on a WiFi link is bandwidth spent on a topic
        already known to be silent."""
        table = {self.cn.COLOR_COMPRESSED_TOPIC: [], self.cn.COLOR_TOPIC: []}
        node = self._node(table, delivering=set())
        self.cn.acquire_stream(node, probe_seconds=0.1)
        self.assertGreaterEqual(len(node.destroyed), 1)

    def test_each_candidate_subscribes_with_its_own_type(self):
        table = {
            self.cn.COLOR_COMPRESSED_TOPIC:
                [("sensor_msgs/msg/CompressedImage", StubQoSProfile())],
            self.cn.COLOR_TOPIC: [("sensor_msgs/msg/Image", StubQoSProfile())],
        }
        node = self._node(table, delivering=set())
        self.cn.acquire_stream(node, probe_seconds=0.1)
        by_topic = {topic: msg_type for msg_type, topic, _ in node.subscriptions_made}
        self.assertIs(by_topic[self.cn.COLOR_COMPRESSED_TOPIC], StubCompressedImage)
        self.assertIs(by_topic[self.cn.COLOR_TOPIC], StubImage)


class TestFailureReport(DiscoveryTestCase):
    def test_it_names_every_topic_tried_and_what_each_offered(self):
        """One run should produce every fact needed, instead of several probes."""
        node = self.cn._StreamNode(raw=False)
        node.publisher_table = {
            self.cn.COLOR_COMPRESSED_TOPIC:
                [("sensor_msgs/msg/CompressedImage",
                  StubQoSProfile(depth=5,
                                 reliability=StubReliability.RELIABLE,
                                 durability=StubDurability.TRANSIENT_LOCAL))],
            self.cn.COLOR_TOPIC: [],
        }
        self.cn.acquire_stream(node, probe_seconds=0.1)
        report = self.cn.failure_report(node)

        self.assertIn(self.cn.COLOR_COMPRESSED_TOPIC, report)
        self.assertIn(self.cn.COLOR_TOPIC, report)
        self.assertIn("CompressedImage", report)
        self.assertIn("TRANSIENT_LOCAL", report)
        self.assertIn("NOTHING PUBLISHING", report,
                      "a topic with no publisher should say so plainly")
        self.assertIn("ros2 topic hz", report, "should say how to cross-check")


if __name__ == "__main__":
    unittest.main()
