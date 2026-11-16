"""Arm pointing: geometry, hand choice, rate limit, deadband, rest.

The arm's frame assumptions are unverified on the real robot (see
arm_control.py). These tests check that the module does what it says it does —
they cannot check that the robot's torso frame matches the assumption. Only
`python arm_control/arm_control.py armtest` can do that.
"""

import math
import unittest

from support import FakeClient, PROJECT_ROOT, install_fake_sdk  # noqa: F401

from arm_control.arm_control import (
    PITCH_LIMIT, REACH_M, YAW_LIMIT, ArmPointer)


class ArmTestCase(unittest.TestCase):
    def setUp(self):
        self.restore_sdk = install_fake_sdk()
        self.client = FakeClient()
        self.arm = ArmPointer(client=self.client, min_interval_s=0.0,
                              min_bearing_change=0.0)

    def tearDown(self):
        self.restore_sdk()


class TestGeometry(ArmTestCase):
    def test_straight_ahead_is_forward_and_centred(self):
        (x, y, z), orientation, use_left = self.arm.target_for(0.0)
        self.assertAlmostEqual(x, REACH_M, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(orientation[2], 0.0, places=6)
        del z, use_left

    def test_positive_bearing_reaches_left(self):
        # Positive bearing is the robot's left, matching head yaw.
        (_, y, _), _, use_left = self.arm.target_for(math.radians(45))
        self.assertGreater(y, 0.0)
        self.assertTrue(use_left, "a person on the left should use the left hand")

    def test_negative_bearing_reaches_right(self):
        (_, y, _), _, use_left = self.arm.target_for(math.radians(-45))
        self.assertLess(y, 0.0)
        self.assertFalse(use_left, "a person on the right should use the right hand")

    def test_the_arm_never_crosses_the_body(self):
        for degrees in range(-70, 71, 5):
            with self.subTest(degrees=degrees):
                (_, y, _), _, use_left = self.arm.target_for(math.radians(degrees))
                if use_left:
                    self.assertGreaterEqual(y, -1e-9)
                else:
                    self.assertLessEqual(y, 1e-9)

    def test_reach_is_constant_in_the_horizontal_plane(self):
        for degrees in (-60, -30, 0, 30, 60):
            (x, y, _), _, _ = self.arm.target_for(math.radians(degrees))
            self.assertAlmostEqual(math.hypot(x, y), REACH_M, places=6)

    def test_orientation_follows_the_bearing(self):
        bearing = math.radians(30)
        _, orientation, _ = self.arm.target_for(bearing)
        self.assertAlmostEqual(orientation[2], bearing, places=6)
        self.assertAlmostEqual(orientation[0], 0.0, places=6)

    def test_bearing_and_pitch_are_clamped(self):
        _, orientation, _ = self.arm.target_for(99.0, 99.0)
        self.assertLessEqual(abs(orientation[2]), YAW_LIMIT + 1e-9)
        self.assertLessEqual(abs(orientation[1]), PITCH_LIMIT + 1e-9)


class TestRateLimiting(unittest.TestCase):
    def setUp(self):
        self.restore_sdk = install_fake_sdk()
        self.client = FakeClient()

    def tearDown(self):
        self.restore_sdk()

    def test_second_call_too_soon_is_ignored(self):
        """An end-effector move takes most of a second to plan and run.

        Commanding it at camera frame rate would queue stale targets that fight
        each other, so the module refuses to send faster than its interval.
        """
        arm = ArmPointer(client=self.client, min_interval_s=5.0,
                         min_bearing_change=0.0)
        self.assertTrue(arm.point_at(0.3))
        self.assertFalse(arm.point_at(0.9))
        self.assertEqual(len(self.client.arm_commands), 1)

    def test_small_bearing_change_is_ignored(self):
        arm = ArmPointer(client=self.client, min_interval_s=0.0,
                         min_bearing_change=0.2)
        self.assertTrue(arm.point_at(0.3))
        self.assertFalse(arm.point_at(0.35), "0.05 rad is inside the deadband")
        self.assertTrue(arm.point_at(0.8), "0.5 rad should get through")
        self.assertEqual(len(self.client.arm_commands), 2)

    def test_force_overrides_both_gates(self):
        arm = ArmPointer(client=self.client, min_interval_s=99.0,
                         min_bearing_change=99.0)
        self.assertTrue(arm.point_at(0.3, force=True))
        self.assertTrue(arm.point_at(0.3, force=True))
        self.assertEqual(len(self.client.arm_commands), 2)

    def test_move_time_is_passed_through(self):
        arm = ArmPointer(client=self.client, move_time_ms=1234,
                         min_interval_s=0.0, min_bearing_change=0.0)
        arm.point_at(0.2)
        _, time_millis, _ = self.client.arm_commands[0]
        self.assertEqual(time_millis, 1234)


class TestRest(ArmTestCase):
    def test_rest_lowers_both_hands(self):
        self.arm.point_at(0.4)
        self.client.arm_commands.clear()
        self.assertTrue(self.arm.rest())
        hands = {command[2] for command in self.client.arm_commands}
        self.assertEqual(len(hands), 2, "both hands should be lowered")
        self.assertTrue(self.arm.at_rest)

    def test_rest_is_idempotent(self):
        self.arm.point_at(0.4)
        self.arm.rest()
        self.client.arm_commands.clear()
        self.assertFalse(self.arm.rest(), "already resting; nothing more to send")
        self.assertEqual(self.client.arm_commands, [])

    def test_rest_tries_the_other_hand_after_a_failure(self):
        """Leaving one arm raised is worse than reporting a failure."""
        self.arm.point_at(0.4)
        failing = FakeClient(fail_times=1, fail_code=500)
        self.arm.client = failing
        self.arm.rest()
        self.assertEqual(len(failing.arm_commands), 1,
                         "the second hand should still be attempted")


class TestNoLocomotion(ArmTestCase):
    def test_arm_pointing_sends_no_locomotion_command(self):
        for degrees in (-60, -20, 0, 20, 60):
            self.arm.point_at(math.radians(degrees), force=True)
        self.arm.rest()
        self.assertEqual(self.client.violations, [],
                         "arm control reached for a locomotion call")
        self.assertGreater(len(self.client.arm_commands), 0)


if __name__ == "__main__":
    unittest.main()
