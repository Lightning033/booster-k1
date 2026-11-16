"""The control law: sign conventions, convergence, deadband, stale frames."""

import math
import unittest

from support import PROJECT_ROOT  # noqa: F401  (puts the project on sys.path)

from follow_controller.follow_controller import (
    CX, CY, FX, FollowController, _simulate, aim_point, intrinsics_for, pixel_error)


class TestPixelError(unittest.TestCase):
    """The signs here decide whether the head turns towards a person or away."""

    def test_centre_has_no_error(self):
        yaw, pitch = pixel_error(CX, CY)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)

    def test_person_left_of_centre_gives_positive_yaw(self):
        # Positive yaw turns the head to the robot's LEFT, so a person on the
        # left must produce a positive error. Getting this backwards drives the
        # head away from them and into its limit.
        yaw, _ = pixel_error(CX - 300, CY)
        self.assertGreater(yaw, 0.0)

    def test_person_right_of_centre_gives_negative_yaw(self):
        yaw, _ = pixel_error(CX + 300, CY)
        self.assertLess(yaw, 0.0)

    def test_person_above_centre_gives_negative_pitch(self):
        # Positive pitch tilts DOWN, so looking UP at a tall person is negative.
        _, pitch = pixel_error(CX, CY - 200)
        self.assertLess(pitch, 0.0)

    def test_person_below_centre_gives_positive_pitch(self):
        _, pitch = pixel_error(CX, CY + 200)
        self.assertGreater(pitch, 0.0)

    def test_known_bearing(self):
        # A person 320 px left of the optical centre, at the recorded fx.
        yaw, _ = pixel_error(CX - 320.0, CY)
        self.assertAlmostEqual(yaw, math.atan2(320.0, FX), places=9)

    def test_intrinsics_scale_with_frame_size(self):
        # Running on a half-size frame must halve the intrinsics, or every
        # bearing comes out twice as large as it should be.
        full = intrinsics_for(1280, 720)
        half = intrinsics_for(640, 360)
        for full_value, half_value in zip(full, half):
            self.assertAlmostEqual(half_value, full_value / 2.0, places=6)

    def test_same_bearing_on_a_downscaled_frame(self):
        full = pixel_error(CX - 320.0, CY, 1280, 720)
        half = pixel_error((CX - 320.0) / 2, CY / 2, 640, 360)
        self.assertAlmostEqual(full[0], half[0], places=6)


class TestAimPoint(unittest.TestCase):
    def test_face_box_is_used_when_present(self):
        point = aim_point((100, 100, 300, 700), face_box=(180, 120, 240, 190))
        self.assertEqual(point, (210.0, 155.0))

    def test_falls_back_to_top_sixth_not_centre(self):
        # The centre of a body box is somebody's stomach.
        x1, y1, x2, y2 = 100, 100, 300, 700
        point = aim_point((x1, y1, x2, y2))
        self.assertEqual(point[0], 200.0)
        self.assertAlmostEqual(point[1], y1 + (y2 - y1) / 6.0)
        self.assertLess(point[1], (y1 + y2) / 2.0)


class TestFollowController(unittest.TestCase):
    def test_converges_and_holds(self):
        log = _simulate()
        self.assertIsNone(log[-1][3], "should end by holding inside the deadband")
        self.assertLessEqual(len(log), 12, "should settle in about a second at 10 Hz")

    def test_converges_towards_the_true_bearing(self):
        person_x, person_top = 320.0, 200.0
        expected_yaw, expected_pitch = pixel_error(person_x, person_top + 500.0 / 6.0)
        log = _simulate(person_x=person_x, person_top=person_top)
        _, final_pitch, final_yaw, _ = log[-1]
        # Within the deadband of the true bearing, which is all that is asked.
        self.assertLess(abs(final_yaw - expected_yaw), 0.03)
        self.assertLess(abs(final_pitch - expected_pitch), 0.03)

    def test_does_not_overshoot(self):
        person_x = 320.0
        expected_yaw, _ = pixel_error(person_x, 200.0 + 500.0 / 6.0)
        for _, _, yaw, _ in _simulate(person_x=person_x):
            self.assertLessEqual(yaw, expected_yaw + 0.02,
                                 "overshot the target bearing")

    def test_both_sides_converge(self):
        for person_x in (120.0, 320.0, 960.0, 1160.0):
            with self.subTest(person_x=person_x):
                log = _simulate(person_x=person_x, ticks=40)
                self.assertIsNone(log[-1][3], "did not settle for x=%s" % person_x)

    def test_deadband_returns_none(self):
        controller = FollowController()
        # A person already dead centre: nothing worth moving for.
        box = (CX - 60, CY - 10, CX + 60, CY + 50)
        face = (CX - 20, CY - 20, CX + 20, CY + 20)
        self.assertIsNone(controller.update(box, face, (720, 1280, 3), 0.0, 0.0))

    def test_stale_frame_holds_and_abandons_the_move(self):
        """A stale frame must freeze the head, not finish the planned move.

        Finishing it would mean turning the head on the strength of a photograph
        of the past. This was a real bug in the earlier project.
        """
        controller = FollowController()
        box = (100, 100, 300, 700)
        result = controller.update(box, None, (720, 1280, 3), 0.0, 0.0)
        self.assertIsNotNone(result)
        moving_target = controller.target_yaw
        self.assertNotAlmostEqual(moving_target, 0.0, places=3)

        # Now the same person, but the frame is older than the stale threshold.
        held = controller.update(box, None, (720, 1280, 3), 0.05, 0.05,
                                 frame_age_s=5.0)
        self.assertIsNone(held)
        # The target was reset to where the head actually is.
        self.assertAlmostEqual(controller.target_yaw, 0.05, places=6)
        self.assertAlmostEqual(controller.target_pitch, 0.05, places=6)

    def test_sync_commands_nothing(self):
        controller = FollowController()
        before = controller.updates
        controller.sync(0.2, -0.3)
        self.assertEqual(controller.updates, before)
        self.assertAlmostEqual(controller.target_pitch, 0.2)
        self.assertAlmostEqual(controller.target_yaw, -0.3)

    def test_correction_is_relative_to_the_current_head_pose(self):
        # The same picture, with the head already turned, must give a different
        # absolute target: the error is measured from where the head is now.
        controller = FollowController()
        box = (100, 100, 300, 700)
        first = controller.update(box, None, (720, 1280, 3), 0.0, 0.0)
        second = controller.update(box, None, (720, 1280, 3), 0.0, 0.4)
        self.assertAlmostEqual(second[1] - first[1], 0.4, places=6)


if __name__ == "__main__":
    unittest.main()
