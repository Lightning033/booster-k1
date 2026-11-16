"""
The camera pipe protocol, with a real child process but no ROS2.

The child is replaced by a stand-in that speaks the same protocol, so the
parent-side contract is exercised for real: a spawned process, a pipe, bounded
waits, request/reply matching and cleanup.
"""

import multiprocessing
import time
import unittest

import cv2
import numpy as np

from support import PROJECT_ROOT  # noqa: F401

from camera_process import camera_process as camera_module
from camera_process.camera_process import CameraProcess


def _jpeg(width=64, height=48, value=120):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = value
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


# These run in a spawned process, so they must be importable at module level.
def stand_in_ok(conn, *_args):
    """Replies to every request with a fresh frame and an increasing seq."""
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    seq = 0
    while True:
        request = conn.recv()
        if request is None:
            return
        seq += 1
        conn.send((request[0], "ok", (_jpeg(), 0.01, None, seq)))


def stand_in_same_frame(conn, *_args):
    """Always replies with the SAME seq, as if no new frame has arrived."""
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    while True:
        request = conn.recv()
        if request is None:
            return
        conn.send((request[0], "ok", (_jpeg(), 0.01, None, 1)))


def stand_in_empty(conn, *_args):
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    while True:
        request = conn.recv()
        if request is None:
            return
        conn.send((request[0], "empty", None))


def slow_stand_in_error(conn, startup_timeout_s, *_args):
    """Reports its failure only AFTER its own deadline, as the real child does."""
    time.sleep(startup_timeout_s + 0.3)
    conn.send((0, "error", "no colour frame on /fake within %.0fs" % startup_timeout_s))


def stand_in_startup_error(conn, *_args):
    conn.send((0, "error", "no colour frame on /fake within 30s"))


def stand_in_bad_jpeg(conn, *_args):
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    seq = 0
    while True:
        request = conn.recv()
        if request is None:
            return
        seq += 1
        conn.send((request[0], "ok", (b"this is not a jpeg", 0.01, None, seq)))


def stand_in_wrong_reply_id(conn, *_args):
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    while True:
        request = conn.recv()
        if request is None:
            return
        conn.send((request[0] + 100, "ok", (_jpeg(), 0.01, None, 1)))


def stand_in_silent(conn, *_args):
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    while True:
        if conn.recv() is None:
            return
        time.sleep(30)                 # never replies within the timeout


def stand_in_dies(conn, *_args):
    conn.send((0, "ok", {"topic": "stand-in", "depth": False}))
    conn.recv()
    return                             # exits without replying -> EOF


class CameraProcessTestCase(unittest.TestCase):
    """Swaps the child entry point for a stand-in, then restores it."""

    entry_point = None

    def setUp(self):
        self.original = camera_module._camera_main
        camera_module._camera_main = self.entry_point
        self.camera = None

    def tearDown(self):
        if self.camera is not None:
            self.camera.close()
        camera_module._camera_main = self.original


class TestNormalStreaming(CameraProcessTestCase):
    entry_point = staticmethod(stand_in_ok)

    def test_startup_reports_the_topic(self):
        self.camera = CameraProcess()
        self.assertEqual(self.camera.info["topic"], "stand-in")

    def test_frames_decode(self):
        self.camera = CameraProcess()
        frame, age, depth = self.camera.latest()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertLess(age, 1.0)
        self.assertIsNone(depth)

    def test_each_reply_is_a_new_frame(self):
        self.camera = CameraProcess()
        for _ in range(3):
            self.camera.latest()
            self.assertTrue(self.camera.is_new())

    def test_close_is_safe_to_call_twice(self):
        self.camera = CameraProcess()
        self.camera.close()
        self.camera.close()


class TestDuplicateFrames(CameraProcessTestCase):
    entry_point = staticmethod(stand_in_same_frame)

    def test_the_same_frame_is_reported_as_not_new(self):
        """The loop uses this to avoid acting twice on one picture."""
        self.camera = CameraProcess()
        self.camera.latest()
        self.assertTrue(self.camera.is_new(), "the first frame is new")
        self.camera.latest()
        self.assertFalse(self.camera.is_new(), "the same frame is not new")


class TestNoFrameYet(CameraProcessTestCase):
    entry_point = staticmethod(stand_in_empty)

    def test_empty_gives_none_and_infinite_age(self):
        self.camera = CameraProcess()
        frame, age, depth = self.camera.latest()
        self.assertIsNone(frame)
        self.assertEqual(age, float("inf"))
        self.assertIsNone(depth)
        self.assertFalse(self.camera.is_new())


class TestFailures(unittest.TestCase):
    def setUp(self):
        self.original = camera_module._camera_main

    def tearDown(self):
        camera_module._camera_main = self.original

    def _expect_failure(self, entry_point, message_fragment):
        camera_module._camera_main = entry_point
        with self.assertRaises(RuntimeError) as caught:
            CameraProcess()
        self.assertIn(message_fragment, str(caught.exception))

    def test_a_startup_error_is_raised_before_the_robot_moves(self):
        """The camera starts first precisely so this stops the run early."""
        self._expect_failure(stand_in_startup_error, "no colour frame")

    def test_a_truncated_jpeg_is_a_dropped_frame_not_a_crash(self):
        camera_module._camera_main = stand_in_bad_jpeg
        camera = CameraProcess()
        try:
            frame, age, _ = camera.latest()
            self.assertIsNone(frame)
            self.assertEqual(age, float("inf"))
            frame, _, _ = camera.latest()       # and it keeps working
            self.assertIsNone(frame)
        finally:
            camera.close()

    def test_a_mismatched_reply_id_is_refused(self):
        camera_module._camera_main = stand_in_wrong_reply_id
        camera = CameraProcess()
        try:
            with self.assertRaises(RuntimeError) as caught:
                camera.latest()
            self.assertIn("does not match request", str(caught.exception))
        finally:
            camera.close()

    def test_a_silent_child_times_out_rather_than_hanging(self):
        camera_module._camera_main = stand_in_silent
        camera = CameraProcess()
        try:
            original_timeout = camera_module.REPLY_TIMEOUT_S
            camera_module.REPLY_TIMEOUT_S = 0.4
            with self.assertRaises(RuntimeError) as caught:
                camera.latest()
            self.assertIn("did not reply", str(caught.exception))
        finally:
            camera_module.REPLY_TIMEOUT_S = original_timeout
            camera.close()

    def test_a_dead_child_is_reported(self):
        camera_module._camera_main = stand_in_dies
        camera = CameraProcess()
        try:
            with self.assertRaises(RuntimeError) as caught:
                camera.latest()
            self.assertIn("exited unexpectedly", str(caught.exception))
        finally:
            camera.close()


class TestSpawnContext(unittest.TestCase):
    def test_spawn_is_used_not_fork(self):
        """Fork would copy this process's DDS threads into the camera process,
        which is the thing the split exists to prevent."""
        source = (PROJECT_ROOT / "camera_process" / "camera_process.py").read_text()
        self.assertIn('get_context("spawn")', source)
        self.assertNotIn('get_context("fork")', source)

    def test_the_camera_node_never_imports_the_sdk(self):
        """The whole reason for the separate process."""
        source = (PROJECT_ROOT / "camera_process" / "camera_node.py").read_text()
        self.assertNotIn("booster_robotics_sdk", source)

    def test_spawn_is_available_on_this_platform(self):
        self.assertIn("spawn", multiprocessing.get_all_start_methods())


if __name__ == "__main__":
    unittest.main()


class TestStartupTimeoutMargin(unittest.TestCase):
    """The parent must outwait the child during startup.

    With both deadlines equal, the parent gave up at the same instant the child
    was reporting WHY it failed, so the only error ever seen was the useless
    "Camera process did not reply within 30s" instead of the child's message
    naming the topic. Observed on the real robot.
    """

    def test_parent_waits_longer_than_the_child(self):
        self.assertGreater(camera_module.STARTUP_REPLY_MARGIN_S, 0.0)

    def test_the_childs_own_error_survives_the_wait(self):
        original = camera_module._camera_main
        camera_module._camera_main = slow_stand_in_error
        original_timeout = camera_module.STARTUP_TIMEOUT_S
        camera_module.STARTUP_TIMEOUT_S = 0.5
        try:
            with self.assertRaises(RuntimeError) as caught:
                CameraProcess()
            message = str(caught.exception)
            self.assertIn("no colour frame", message,
                          "the child's diagnostic was lost; got: %s" % message)
            self.assertNotIn("did not reply", message)
        finally:
            camera_module.STARTUP_TIMEOUT_S = original_timeout
            camera_module._camera_main = original
