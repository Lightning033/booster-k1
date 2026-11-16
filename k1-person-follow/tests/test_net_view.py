"""The detector view: tapping the class branch, throttling, and serving."""

import base64
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

from support import PROJECT_ROOT  # noqa: F401

from net_view.net_view import (
    MIN_PUBLISH_INTERVAL_S, SCALES, NetView, ScaleTap, encode_grid)


class FakeOutput:
    """Stands in for the tensor a cv3 branch returns."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype="float32")


class FakeBranch:
    """One cv3 branch: records hooks and can be fired on demand."""

    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        branch = self

        class Handle:
            def remove(self):
                if fn in branch.hooks:
                    branch.hooks.remove(fn)
        return Handle()

    def fire(self, tensor):
        for fn in list(self.hooks):
            fn(self, None, tensor)


class FakeHead:
    def __init__(self, n=3):
        self.cv3 = [FakeBranch() for _ in range(n)]


class FakeSequential:
    def __init__(self, head):
        self._items = [object(), head]

    def __getitem__(self, i):
        return self._items[i]


class FakeDetector:
    """A PersonDetector whose predictor appears only after the first run."""

    def __init__(self, head=None):
        self.head = head or FakeHead()
        self.model = type("M", (), {"predictor": None})()

    def run_once(self):
        backend = type("B", (), {"model": type("D", (), {
            "model": FakeSequential(self.head)})()})()
        self.model.predictor = type("P", (), {"model": backend})()


class TestEncodeGrid(unittest.TestCase):
    def test_round_trip_of_values(self):
        grid = np.array([[0.0, 0.5], [1.0, 0.25]], dtype="float32")
        out = encode_grid(grid)
        self.assertEqual((out["w"], out["h"]), (2, 2))
        self.assertAlmostEqual(out["peak"], 1.0, places=4)
        raw = np.frombuffer(base64.b64decode(out["d"]), dtype="uint8")
        self.assertEqual(list(raw), [0, 128, 255, 64])

    def test_values_outside_the_range_are_clipped(self):
        out = encode_grid(np.array([[-2.0, 7.0]], dtype="float32"))
        raw = np.frombuffer(base64.b64decode(out["d"]), dtype="uint8")
        self.assertEqual(list(raw), [0, 255])

    def test_empty_or_wrong_shape_is_refused(self):
        self.assertIsNone(encode_grid(np.zeros((0, 0), dtype="float32")))
        self.assertIsNone(encode_grid(np.zeros((2, 2, 2), dtype="float32")))


class TestScaleTap(unittest.TestCase):
    def test_it_cannot_attach_before_the_first_inference(self):
        """ultralytics has no predictor until the model has run once."""
        detector = FakeDetector()
        tap = ScaleTap(detector)
        self.assertFalse(tap.maybe_attach())
        self.assertIsNone(tap.latest())

    def test_it_attaches_once_the_predictor_exists(self):
        detector = FakeDetector()
        detector.run_once()
        tap = ScaleTap(detector)
        self.assertTrue(tap.maybe_attach())
        self.assertTrue(tap.attached)

    def test_attaching_twice_does_not_add_a_second_hook(self):
        detector = FakeDetector()
        detector.run_once()
        tap = ScaleTap(detector)
        tap.maybe_attach()
        tap.maybe_attach()
        for branch in detector.head.cv3:
            self.assertEqual(len(branch.hooks), 1)

    def test_it_hooks_the_predictor_model_not_the_original(self):
        """The trap this module exists around.

        ultralytics copies the model into its predictor on the first inference,
        and a module's hooks are copied with it. Hooks registered on the
        original therefore fire on a model that is no longer executing, and
        removing them does nothing. The tap must attach to the predictor's
        model, which is reachable only after that first run.
        """
        detector = FakeDetector()
        detector.run_once()
        tap = ScaleTap(detector)
        tap.maybe_attach()
        head = detector.model.predictor.model.model.model[-1]
        self.assertIs(head, detector.head)
        self.assertEqual(len(head.cv3[0].hooks), 1)

    def test_detach_removes_every_hook_and_the_captured_maps(self):
        detector = FakeDetector()
        detector.run_once()
        tap = ScaleTap(detector)
        tap.maybe_attach()
        tap.detach()
        for branch in detector.head.cv3:
            self.assertEqual(branch.hooks, [],
                             "a hook survived detach; it would keep capturing")
        self.assertFalse(tap.attached)
        self.assertIsNone(tap.latest())

    def test_a_head_without_a_class_branch_fails_quietly(self):
        detector = FakeDetector(head=object())
        tap = ScaleTap(detector)
        detector.run_once()
        self.assertFalse(tap.maybe_attach())
        self.assertIsNone(tap.latest())


class TestNetViewPublishing(unittest.TestCase):
    def setUp(self):
        self.view = NetView(serve=False)

    def _items(self, peaks=(0.03, 0.05, 0.82)):
        items = []
        for index, (name, stride) in enumerate(SCALES):
            grid = np.zeros((4, 6), dtype="float32")
            grid[1, 2] = peaks[index]
            items.append((name, stride, grid))
        return items

    def test_the_first_publish_goes_through(self):
        self.assertTrue(self.view.publish(self._items(), {"fps": 12}))
        state = self.view.snapshot()
        self.assertEqual(len(state["scales"]), 3)
        self.assertEqual(state["scales"][2]["name"], "P5")
        self.assertAlmostEqual(state["scales"][2]["peak"], 0.82, places=2)

    def test_a_second_publish_straight_after_is_dropped(self):
        """Encoding on every frame is work the page never reads."""
        self.view.publish(self._items(), {"fps": 12})
        self.assertFalse(self.view.publish(self._items(), {"fps": 12}))
        self.assertEqual(self.view.published, 1)
        self.assertEqual(self.view.skipped, 1)

    def test_a_dropped_publish_still_refreshes_the_numbers(self):
        """Stats are cheap; only the grids are worth throttling."""
        self.view.publish(self._items(), {"fps": 12.0})
        self.view.publish(self._items(), {"fps": 3.5})
        self.assertAlmostEqual(self.view.snapshot()["stats"]["fps"], 3.5)

    def test_force_overrides_the_throttle(self):
        self.view.publish(self._items(), {"fps": 12})
        self.assertTrue(self.view.publish(self._items(), {"fps": 12}, force=True))

    def test_the_throttle_matches_the_pages_poll_rate(self):
        self.assertGreater(MIN_PUBLISH_INTERVAL_S, 0.1)
        self.assertLess(MIN_PUBLISH_INTERVAL_S, 1.0)

    def test_publishing_nothing_leaves_a_valid_state(self):
        self.view.publish(None, {"fps": 0})
        self.assertEqual(self.view.snapshot()["scales"], [])


class TestNetViewServing(unittest.TestCase):
    def setUp(self):
        self.view = NetView(port=8600, attempts=20)
        if not self.view.serve:
            self.skipTest("no port available in this environment")

    def tearDown(self):
        self.view.close()

    def _get(self, path):
        url = "http://127.0.0.1:%d%s" % (self.view.port, path)
        return urllib.request.urlopen(url, timeout=5).read()

    def test_it_serves_the_page(self):
        body = self._get("/").decode("utf-8")
        self.assertIn("<canvas>", body)
        self.assertIn("state.json", body)

    def test_state_is_json_and_carries_the_grids(self):
        grid = np.zeros((3, 4), dtype="float32")
        grid[0, 0] = 0.9
        self.view.publish([("P5", 32, grid)], {"fps": 11.5, "name": "Harsh"},
                          force=True)
        state = json.loads(self._get("/state.json"))
        self.assertEqual(state["stats"]["name"], "Harsh")
        self.assertEqual(state["scales"][0]["stride"], 32)
        self.assertEqual((state["scales"][0]["w"], state["scales"][0]["h"]), (4, 3))

    def test_an_unknown_path_is_404_not_a_crash(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_a_busy_port_is_stepped_over(self):
        """A leftover run must not stop the view from starting."""
        blocker = NetView(port=self.view.port, attempts=1)
        self.addCleanup(blocker.close)
        self.assertFalse(blocker.serve, "expected the first bind to fail")
        stepped = NetView(port=self.view.port, attempts=6)
        self.addCleanup(stepped.close)
        self.assertTrue(stepped.serve)
        self.assertNotEqual(stepped.port, self.view.port)

    def test_close_is_safe_to_call_twice(self):
        self.view.close()
        self.view.close()


if __name__ == "__main__":
    unittest.main()
