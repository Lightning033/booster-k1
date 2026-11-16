"""Head control: stepping, limits, sign application, retry, and no legs."""

import unittest

from support import FakeClient, PROJECT_ROOT  # noqa: F401

from head_control.head_control import (
    MAX_STEP, PITCH_LIMIT, PITCH_SIGN, YAW_LIMIT, YAW_SIGN, HeadController, clamp)


class TestClamp(unittest.TestCase):
    def test_clamps_both_ways(self):
        self.assertEqual(clamp(5.0, 1.0), 1.0)
        self.assertEqual(clamp(-5.0, 1.0), -1.0)
        self.assertEqual(clamp(0.3, 1.0), 0.3)


class TestStepping(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.head = HeadController(client=self.client, step_pause_s=0.0)

    def test_one_step_per_call(self):
        done = self.head.step_towards(0.0, 0.5)
        self.assertFalse(done)
        self.assertEqual(len(self.client.head_commands), 1)
        self.assertAlmostEqual(self.head.yaw, MAX_STEP, places=6)

    def test_no_command_jumps_further_than_the_step_limit(self):
        """A single large jump snaps the head, which is why this limit exists."""
        self.head.look_at(0.0, YAW_LIMIT)
        previous_yaw = 0.0
        for _, yaw in self.client.head_commands:
            commanded = yaw / YAW_SIGN
            self.assertLessEqual(abs(commanded - previous_yaw), MAX_STEP + 1e-9)
            previous_yaw = commanded

    def test_reaches_the_target(self):
        self.assertTrue(self.head.look_at(0.0, 0.5))
        self.assertAlmostEqual(self.head.yaw, 0.5, places=6)

    def test_already_there_sends_nothing(self):
        self.assertTrue(self.head.step_towards(0.0, 0.0))
        self.assertEqual(self.client.head_commands, [])

    def test_limits_are_clamped_before_sending(self):
        self.head.look_at(10.0, 10.0)
        for pitch, yaw in self.client.head_commands:
            self.assertLessEqual(abs(pitch / PITCH_SIGN), PITCH_LIMIT + 1e-9)
            self.assertLessEqual(abs(yaw / YAW_SIGN), YAW_LIMIT + 1e-9)

    def test_signs_are_applied_on_the_wire_only(self):
        """The believed pose is in robot terms; the sign is applied when sending.

        Keeping them separate means flipping a sign after a calibration test
        cannot silently invert the control law as well.
        """
        self.head.step_towards(0.0, 0.5)
        _, sent_yaw = self.client.head_commands[-1]
        self.assertAlmostEqual(sent_yaw, YAW_SIGN * self.head.yaw, places=9)

    def test_center_returns_to_zero(self):
        self.head.look_at(0.0, 0.6)
        self.head.center()
        self.assertAlmostEqual(self.head.yaw, 0.0, places=6)
        self.assertAlmostEqual(self.head.pitch, 0.0, places=6)
        last_pitch, last_yaw = self.client.head_commands[-1]
        self.assertAlmostEqual(last_pitch, 0.0, places=6)
        self.assertAlmostEqual(last_yaw, 0.0, places=6)


class TestFailureHandling(unittest.TestCase):
    def test_a_failed_send_does_not_update_the_believed_pose(self):
        """Otherwise the controller thinks the head moved when it did not."""
        client = FakeClient(fail_times=10, fail_code=400)
        head = HeadController(client=client, step_pause_s=0.0)
        with self.assertRaises(RuntimeError):
            head.step_towards(0.0, 0.5)
        self.assertEqual(head.position(), (0.0, 0.0))

    def test_timeout_is_retried(self):
        # Code 100 is a WiFi RPC timeout and does get retried.
        client = FakeClient(fail_times=2, fail_code=100)
        head = HeadController(client=client, step_pause_s=0.0)
        head.step_towards(0.0, 0.5)
        self.assertEqual(len(client.head_commands), 1)

    def test_bad_request_is_not_retried(self):
        # Code 400 means the robot refused; repeating it will not help.
        client = FakeClient(fail_times=1, fail_code=400)
        head = HeadController(client=client, step_pause_s=0.0)
        with self.assertRaises(RuntimeError):
            head.step_towards(0.0, 0.5)
        self.assertEqual(client.head_commands, [])

    def test_sync_adopts_a_pose_without_commanding(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        head.sync(0.2, -0.4)
        self.assertEqual(client.head_commands, [])
        self.assertAlmostEqual(head.pitch, 0.2)
        self.assertAlmostEqual(head.yaw, -0.4)

    def test_sync_still_respects_limits(self):
        head = HeadController(client=FakeClient(), step_pause_s=0.0)
        head.sync(99.0, 99.0)
        self.assertLessEqual(abs(head.pitch), PITCH_LIMIT)
        self.assertLessEqual(abs(head.yaw), YAW_LIMIT)


class TestNoLocomotion(unittest.TestCase):
    """The head must never be a route to moving the legs.

    FakeClient adds Move, ChangeMode and stop_motion, which do not exist in this
    project's own code. Anything that reaches for one is recorded here.
    """

    def test_head_moves_send_no_locomotion_command(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        head.look_at(0.3, 0.6)
        head.look_at(-0.3, -0.6)
        head.center()
        self.assertEqual(client.violations, [],
                         "head control reached for a locomotion call")
        self.assertGreater(len(client.head_commands), 0)

    def test_the_bait_actually_works(self):
        """Confirm the trap would catch a violation, so a pass means something."""
        client = FakeClient()
        client.Move(0.1, 0.0, 0.0)
        self.assertEqual(len(client.violations), 1)


if __name__ == "__main__":
    unittest.main()
