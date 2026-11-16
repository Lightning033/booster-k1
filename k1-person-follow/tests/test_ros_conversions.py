"""
The ROS image conversions, with synthetic messages and no ROS2 installed.

camera_node.py imports rclpy at module level, which is absent off the robot, so
the two pure functions are loaded from source instead. They are worth testing
directly because `step` padding and the RGB/BGR swap are both easy to get
subtly wrong, and a wrong channel order would quietly ruin face recognition
while still looking like a picture.
"""

import types
import unittest

import numpy as np

from support import PROJECT_ROOT  # noqa: F401


def _load_conversions():
    """Compile just the conversion functions out of camera_node.py.

    Importing the module would pull in rclpy. Taking the functions by source
    keeps this test honest — it exercises the real code, not a copy.
    """
    source = (PROJECT_ROOT / "camera_process" / "camera_node.py").read_text()
    wanted = ("def msg_to_bgr", "def msg_to_depth")
    blocks = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(wanted):
            end = index + 1
            while end < len(lines) and (lines[end].startswith((" ", "\t"))
                                        or not lines[end].strip()):
                end += 1
            blocks.append("\n".join(lines[index:end]))
    module = types.ModuleType("conversions")
    module.__dict__["np"] = np
    exec(compile("\n\n".join(blocks), "camera_node.py", "exec"), module.__dict__)
    return module


conversions = _load_conversions()


class FakeImageMessage:
    def __init__(self, height, width, encoding, data, step=None):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.data = data
        self.step = step if step is not None else width * 3


class TestColourConversion(unittest.TestCase):
    def test_rgb8_is_swapped_to_bgr(self):
        """A red pixel in RGB must come out as red in BGR, i.e. channel 2.

        Getting this backwards still produces a plausible-looking image, so it
        would not be obvious — but every face embedding would be computed from
        colour-swapped pixels.
        """
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255                        # pure red in RGB
        message = FakeImageMessage(2, 2, "rgb8", rgb.tobytes())
        bgr = conversions.msg_to_bgr(message)
        self.assertEqual(bgr.shape, (2, 2, 3))
        self.assertEqual(int(bgr[0, 0, 2]), 255, "red should land in BGR channel 2")
        self.assertEqual(int(bgr[0, 0, 0]), 0)

    def test_bgr8_is_left_alone(self):
        bgr_in = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr_in[:, :, 0] = 255                     # blue in BGR
        message = FakeImageMessage(2, 2, "bgr8", bgr_in.tobytes())
        bgr = conversions.msg_to_bgr(message)
        self.assertEqual(int(bgr[0, 0, 0]), 255, "bgr8 must not be swapped")

    def test_row_padding_is_stripped(self):
        """ROS rows may be padded: step can exceed width * 3.

        Ignoring that shears the image diagonally — recognisable as a picture,
        useless for detection.
        """
        width, height, pad = 3, 2, 5
        step = width * 3 + pad
        raw = bytearray()
        for row in range(height):
            for column in range(width):
                raw += bytes((row * 10 + column, 0, 0))   # R varies by position
            raw += bytes(pad)                              # padding
        message = FakeImageMessage(height, width, "rgb8", bytes(raw), step=step)
        bgr = conversions.msg_to_bgr(message)
        self.assertEqual(bgr.shape, (height, width, 3))
        for row in range(height):
            for column in range(width):
                self.assertEqual(int(bgr[row, column, 2]), row * 10 + column,
                                 "padding was not stripped correctly")

    def test_encoding_is_case_insensitive(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        message = FakeImageMessage(1, 1, "RGB8", rgb.tobytes())
        self.assertEqual(conversions.msg_to_bgr(message).shape, (1, 1, 3))

    def test_an_unknown_encoding_is_refused(self):
        message = FakeImageMessage(1, 1, "yuv422", b"\x00" * 4)
        with self.assertRaises(ValueError) as caught:
            conversions.msg_to_bgr(message)
        self.assertIn("rgb8", str(caught.exception))

    def test_the_result_owns_its_memory(self):
        """The buffer comes from the message; the array must not alias it."""
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        message = FakeImageMessage(2, 2, "rgb8", rgb.tobytes())
        bgr = conversions.msg_to_bgr(message)
        bgr[0, 0, 0] = 99                         # must not raise on a read-only view
        self.assertEqual(int(bgr[0, 0, 0]), 99)


class TestDepthConversion(unittest.TestCase):
    def test_16uc1_millimetres_become_metres(self):
        depth_mm = np.array([[1000, 2500], [0, 65535]], dtype=np.uint16)
        message = FakeImageMessage(2, 2, "16UC1", depth_mm.tobytes(), step=2 * 2)
        metres = conversions.msg_to_depth(message)
        self.assertEqual(metres.dtype, np.float32)
        self.assertAlmostEqual(float(metres[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(metres[0, 1]), 2.5, places=5)
        self.assertAlmostEqual(float(metres[1, 0]), 0.0, places=5)

    def test_32fc1_is_already_metres(self):
        depth_m = np.array([[1.5, 3.25]], dtype=np.float32)
        message = FakeImageMessage(1, 2, "32FC1", depth_m.tobytes(), step=2 * 4)
        metres = conversions.msg_to_depth(message)
        self.assertAlmostEqual(float(metres[0, 0]), 1.5, places=5)
        self.assertAlmostEqual(float(metres[0, 1]), 3.25, places=5)

    def test_16uc1_padding_is_stripped(self):
        width, height = 2, 2
        step_bytes = width * 2 + 4                 # 2 extra uint16 of padding
        raw = bytearray()
        for row in range(height):
            raw += np.array([row * 1000 + 500, row * 1000 + 600],
                            dtype=np.uint16).tobytes()
            raw += bytes(4)
        message = FakeImageMessage(height, width, "16UC1", bytes(raw), step=step_bytes)
        metres = conversions.msg_to_depth(message)
        self.assertEqual(metres.shape, (height, width))
        self.assertAlmostEqual(float(metres[1, 1]), 1.6, places=5)

    def test_an_unknown_depth_encoding_is_refused(self):
        message = FakeImageMessage(1, 1, "32SC1", b"\x00" * 4)
        with self.assertRaises(ValueError) as caught:
            conversions.msg_to_depth(message)
        self.assertIn("16UC1", str(caught.exception))


class TestTopics(unittest.TestCase):
    def test_the_compressed_topic_is_the_default(self):
        """Raw 1280x720 at 5 Hz is about 14 MB/s, near this link's ceiling and
        the measured bottleneck of the previous project."""
        source = (PROJECT_ROOT / "camera_process" / "camera_node.py").read_text()
        self.assertIn("image_raw/compressed", source)
        self.assertIn("raw=False", source)

    def test_depth_is_aligned_to_colour(self):
        """An unaligned depth map would not share the colour pixel grid, so
        depth at (u, v) would be a different point from colour at (u, v)."""
        source = (PROJECT_ROOT / "camera_process" / "camera_node.py").read_text()
        self.assertIn("aligned_depth_to_color", source)


if __name__ == "__main__":
    unittest.main()
