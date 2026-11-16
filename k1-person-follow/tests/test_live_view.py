"""The live page: the MJPEG endpoints, the status JSON, and the name slot."""

import json
import socket
import unittest
import urllib.error
import urllib.request

import numpy as np

from support import PROJECT_ROOT  # noqa: F401

from live_view.live_view import LiveView, annotate
from support import FakePerson


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get(url, timeout=4.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def post(url, timeout=4.0):
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


class LiveViewTestCase(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.view = LiveView(port=self.port, serve_web=True, window=False)
        self.base = "http://127.0.0.1:%d" % self.port

    def tearDown(self):
        self.view.close()


class TestEndpoints(LiveViewTestCase):
    def test_the_page_is_served(self):
        status, body = get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn(b"K1 Person Follow", body)
        self.assertIn(b"/stream.mjpg", body)

    def test_status_is_json(self):
        status, body = get(self.base + "/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        for key in ("state", "name", "fps", "asking", "enrolled_count"):
            self.assertIn(key, payload)

    def test_status_reflects_set_status(self):
        self.view.set_status(state="following", name="Harsh", track_id=7, fps=12.5)
        payload = json.loads(get(self.base + "/status")[1])
        self.assertEqual(payload["state"], "following")
        self.assertEqual(payload["name"], "Harsh")
        self.assertEqual(payload["track_id"], 7)
        self.assertAlmostEqual(payload["fps"], 12.5)

    def test_a_frame_is_served_once_published(self):
        self.assertEqual(get(self.base + "/frame.jpg", timeout=4.0)[0]
                         if self.view.current_jpeg() else 503, 503)
        self.view.publish(np.zeros((48, 64, 3), dtype=np.uint8))
        status, body = get(self.base + "/frame.jpg")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\xff\xd8"), "not a JPEG")

    def test_unknown_paths_are_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            get(self.base + "/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_the_frame_endpoint_says_so_before_any_frame(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            get(self.base + "/frame.jpg")
        self.assertEqual(caught.exception.code, 503)


class TestNameSlot(LiveViewTestCase):
    def test_nothing_pending_returns_none(self):
        self.assertIsNone(self.view.take_name())

    def test_a_submitted_name_reaches_the_loop(self):
        post(self.base + "/name?value=Harsh%20P")
        self.assertEqual(self.view.take_name(), "Harsh P")

    def test_taking_clears_the_slot(self):
        post(self.base + "/name?value=Harsh")
        self.assertEqual(self.view.take_name(), "Harsh")
        self.assertIsNone(self.view.take_name())

    def test_skip_gives_an_empty_string_not_none(self):
        """None means 'nothing happened'; "" means 'do not enrol this person'.
        The loop needs to tell those apart."""
        post(self.base + "/name?skip=1")
        self.assertEqual(self.view.take_name(), "")

    def test_blank_names_are_ignored(self):
        post(self.base + "/name?value=%20%20")
        self.assertIsNone(self.view.take_name())

    def test_ask_name_toggles_the_flag(self):
        self.view.ask_name(True)
        self.assertTrue(json.loads(get(self.base + "/status")[1])["asking"])
        self.view.ask_name(False)
        self.assertFalse(json.loads(get(self.base + "/status")[1])["asking"])

    def test_a_posted_name_survives_odd_characters(self):
        post(self.base + "/name?value=%C3%89lodie%20O%27Neill")
        self.assertEqual(self.view.take_name(), "Élodie O'Neill")


class TestWebDisabled(unittest.TestCase):
    def test_no_server_is_started(self):
        view = LiveView(port=free_port(), serve_web=False, window=False)
        try:
            self.assertEqual(view.urls, [])
            view.publish(np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertIsNotNone(view.current_jpeg())
        finally:
            view.close()

    def test_pump_window_is_a_noop_when_disabled(self):
        view = LiveView(port=free_port(), serve_web=False, window=False)
        try:
            self.assertEqual(view.pump_window(np.zeros((10, 10, 3), np.uint8)), -1)
        finally:
            view.close()


class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_it_does_not_modify_the_original(self):
        """The frame is also what gets enrolled, so it must stay unmarked."""
        before = self.frame.copy()
        annotate(self.frame, people=[FakePerson((100, 100, 300, 600), track_id=1)],
                 locked_id=1, locked_name="Harsh")
        self.assertTrue(np.array_equal(self.frame, before),
                        "annotate drew on the original frame")

    def test_it_returns_a_frame_of_the_same_shape(self):
        canvas = annotate(self.frame, people=[], banner="hello")
        self.assertEqual(canvas.shape, self.frame.shape)

    def test_it_draws_something(self):
        canvas = annotate(self.frame,
                          people=[FakePerson((100, 100, 300, 600), track_id=1)],
                          locked_id=1, locked_name="Harsh",
                          face_box=(180, 140, 240, 210), aim=(210, 175),
                          banner="following Harsh", bearing_rad=0.42)
        self.assertGreater(int(canvas.sum()), 0, "nothing was drawn")

    def test_it_copes_with_nothing_to_draw(self):
        canvas = annotate(self.frame)
        self.assertEqual(canvas.shape, self.frame.shape)

    def test_it_copes_with_a_face_and_no_people(self):
        canvas = annotate(self.frame, face_box=(10, 10, 60, 60))
        self.assertGreater(int(canvas.sum()), 0)

    def test_boxes_outside_the_frame_do_not_crash(self):
        canvas = annotate(self.frame,
                          people=[FakePerson((-50, -50, 2000, 2000), track_id=1)],
                          locked_id=1, locked_name="Harsh")
        self.assertEqual(canvas.shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()
