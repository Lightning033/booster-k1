"""
The whole flow with no robot, no ROS2 and no YOLO:
    unknown person -> name asked -> enrolled -> locked -> head follows
and then, on the next run, recognised immediately from disk.

This is the test that says the demo works. Everything crossing a boundary is a
fake: the camera, the detector, the face recognizer and the robot client.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from support import (  # noqa: F401
    PROJECT_ROOT, FakeClient, FakeDetector, FakeFace, FakePerson, FakeRecognizer,
    grey_frame, install_fake_sdk, unit_embedding)

from face_db.face_db import FaceDB
from follow_controller.follow_controller import CX, CY
from head_control.head_control import HeadController
from pipeline.person_follow import (
    DEFAULT_SPEED, FollowSession, parse_args)


class StubView:
    """A LiveView that records instead of serving. No sockets, no windows."""

    serve_web = False
    window = False

    def __init__(self):
        self.status = {}
        self.asking = False
        self.published = 0
        self._pending = None

    def set_status(self, **fields):
        self.status.update(fields)

    def ask_name(self, asking=True):
        self.asking = bool(asking)

    def queue_name(self, name):
        self._pending = name

    def take_name(self):
        name, self._pending = self._pending, None
        return name

    def publish(self, _frame):
        self.published += 1

    def pump_window(self, _frame):
        return -1

    def close(self):
        pass


def make_session(detector, recognizer, database, head=None, arm=None,
                 dry_run=False, hold=False, speed=None, lock_nearest=False,
                 no_ask=False):
    """A FollowSession with every boundary faked and nothing started.

    Flags are passed through parse_args rather than set on the namespace, so a
    test exercises the same argument handling the real run does -- including
    the clamping of --speed.
    """
    argv = []
    if dry_run:
        argv.append("--dry-run")
    if not hold:
        argv.append("--no-hold")
    if no_ask:
        argv.append("--no-ask")
    if lock_nearest:
        argv.append("--lock-nearest")
    if speed is not None:
        argv += ["--speed", str(speed)]
    args = parse_args(argv)
    session = FollowSession(args)
    session.detector = detector
    session.recognizer = recognizer
    session.database = database
    session.view = StubView()
    session.head = head
    session.arm = arm
    return session


# A person standing left of centre, with a readable face inside their box.
PERSON_LEFT = FakePerson((200, 150, 460, 700), track_id=7)
FACE_LEFT = FakeFace((300, 180, 380, 270))        # 80 px wide: big enough to enrol
TINY_FACE = FakeFace((300, 180, 320, 200))        # 20 px wide: too small

# A person dead centre, so the controller has nothing to correct. The FACE box
# is what the head aims at when one is found, so it is the box that has to be
# centred on (CX, CY) — centring the body box instead leaves a pitch error.
PERSON_CENTRE = FakePerson((int(CX) - 130, int(CY) - 220, int(CX) + 130,
                            int(CY) + 280), track_id=7)
FACE_CENTRE = FakeFace((int(CX) - 40, int(CY) - 40, int(CX) + 40, int(CY) + 40))


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="k1_flow_"))
        self.database = FaceDB({}, root=self.root)
        self.frame = grey_frame()
        self.restore_sdk = install_fake_sdk()

    def tearDown(self):
        self.restore_sdk()
        shutil.rmtree(self.root, ignore_errors=True)


class TestEnrolmentFlow(PipelineTestCase):
    def test_an_unknown_face_triggers_the_name_prompt(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertTrue(session.view.asking, "the name field should be showing")
        self.assertEqual(session.asking_about, 7)
        self.assertIsNone(session.locked_id, "must not lock before a name is given")

    def test_a_face_too_small_is_not_asked_about(self):
        """A tiny face across the room makes a poor stored sample, and that
        sample is what every later recognition is compared against."""
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[TINY_FACE]], unit_embedding(5)),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertFalse(session.view.asking)
        self.assertEqual(len(self.database), 0)

    def test_a_too_small_face_says_so_and_gives_the_distance(self):
        """A run that sits in "searching" forever must say WHY on the console.

        Without this the operator cannot tell a stalled run from an empty room,
        and the only way to find out is to ask someone who knows the code.
        """
        from pipeline.person_follow import too_small_message

        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[TINY_FACE]], unit_embedding(5)),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertIsNotNone(session.reason, "no reason recorded for the console")
        self.assertIn("too small", session.reason)
        self.assertIn("come inside", session.reason,
                      "the message must say how much closer to stand")

        message = too_small_message(38)
        self.assertIn("38 px", message)
        self.assertIn("54", message)
        self.assertIn("2.7 m", message)

    def test_the_reason_is_cleared_once_somebody_is_locked(self):
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertIsNotNone(session.locked_id)
        self.assertIsNone(session.reason,
                          "a locked run must not keep printing a stale reason")

    def test_distance_estimate_matches_the_measured_detector_table(self):
        """face_distance_m must agree with the range table in recognize_face.py,
        which was measured separately on real frames."""
        from pipeline.person_follow import face_distance_m

        for width_px, expected_m in ((29, 3.6), (23, 4.5)):
            self.assertAlmostEqual(face_distance_m(width_px), expected_m,
                                   delta=0.1,
                                   msg="%d px should be about %.1f m"
                                       % (width_px, expected_m))
        self.assertIsNone(face_distance_m(0))

    def test_typing_a_name_enrols_and_locks(self):
        embedding = unit_embedding(5)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)             # asks
        session.view.queue_name("Harsh")
        session._step(self.frame, 0.0)             # name arrives

        self.assertEqual(len(self.database), 1)
        self.assertEqual(self.database.names(), ["Harsh"])
        self.assertEqual(session.locked_id, 7)
        self.assertEqual(session.locked_name, "Harsh")
        self.assertFalse(session.view.asking, "the prompt should close")

    def test_the_enrolment_is_on_disk_for_the_next_run(self):
        """The requirement: saved once, recognised every run after."""
        embedding = unit_embedding(5)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        session.view.queue_name("Harsh")
        session._step(self.frame, 0.0)

        # A brand new run: a fresh database read off disk, nothing in memory.
        reloaded = FaceDB.load(root=self.root)
        self.assertEqual(reloaded.names(), ["Harsh"])

        second = make_session(FakeDetector([[PERSON_LEFT]]),
                              FakeRecognizer([[FACE_LEFT]], embedding),
                              reloaded, dry_run=True)
        second._step(self.frame, 0.0)
        self.assertEqual(second.locked_name, "Harsh",
                         "should lock on immediately, with no prompt")
        self.assertFalse(second.view.asking)

    def test_skip_does_not_enrol_and_is_not_asked_again(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        session.view.queue_name("")                # the Skip button
        session._step(self.frame, 0.0)

        self.assertEqual(len(self.database), 0)
        self.assertIsNone(session.locked_id)
        self.assertIn(7, session.skipped_ids)

        # Later frames must not re-ask about the same person.
        session._step(self.frame, 0.0)
        self.assertFalse(session.view.asking)

    def test_a_frame_with_no_usable_face_does_not_crash_enrolment(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], None),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        session.view.queue_name("Harsh")
        session._step(self.frame, 0.0)             # embedding is None
        self.assertEqual(len(self.database), 0, "nothing should be stored")


class TestFollowing(PipelineTestCase):
    def _locked_session(self, head=None, arm=None, person=PERSON_LEFT,
                        face=FACE_LEFT, dry_run=False):
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[person]]),
                               FakeRecognizer([[face]], embedding),
                               self.database, head=head, arm=arm, dry_run=dry_run)
        session._step(self.frame, 0.0)
        return session

    def test_a_known_person_is_locked_without_being_asked(self):
        session = self._locked_session(dry_run=True)
        self.assertEqual(session.locked_name, "Harsh")
        self.assertFalse(session.view.asking)

    def test_the_head_is_commanded_towards_a_person_on_the_left(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        session = self._locked_session(head=head)
        session._step(self.frame, 0.0)

        self.assertGreater(len(client.head_commands), 0, "no head command sent")
        _, yaw = client.head_commands[-1]
        self.assertGreater(yaw, 0.0,
                           "a person on the left should turn the head left")

    def test_a_centred_person_produces_no_head_command(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        self._locked_session(head=head, person=PERSON_CENTRE, face=FACE_CENTRE)
        self.assertEqual(client.head_commands, [],
                         "a centred person is inside the deadband")

    def test_a_stale_frame_sends_nothing(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        session = self._locked_session(head=head)
        client.head_commands.clear()
        session._step(self.frame, 5.0)             # five seconds old
        self.assertEqual(client.head_commands, [],
                         "the head must freeze on a stale frame, not act on it")

    def test_dry_run_never_touches_the_robot(self):
        client = FakeClient()
        session = self._locked_session(dry_run=True)
        session._step(self.frame, 0.0)
        self.assertIsNone(session.head)
        self.assertEqual(client.head_commands, [])

    def test_losing_the_track_unlocks_after_a_while(self):
        session = self._locked_session(dry_run=True)
        self.assertEqual(session.locked_id, 7)
        session.detector = FakeDetector([[]])      # nobody in view any more
        for _ in range(40):
            session._step(self.frame, 0.0)
        self.assertIsNone(session.locked_id)

    def test_a_brief_disappearance_keeps_the_lock(self):
        """People step behind things. One missed frame is not a lost person."""
        session = self._locked_session(dry_run=True)
        session.detector = FakeDetector([[]])
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7, "one missed frame dropped the lock")

    def test_a_new_track_id_is_re_acquired_by_face(self):
        """A dropped frame makes ByteTrack issue NEW ids for the same person.

        Observed for real: one dropout and the locked id was gone for good, so
        the head sat frozen for 25 ticks — five seconds at 5 Hz — with the person
        standing in plain view. The lock has to be on the identity, not the id.
        """
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)

        # Same person, same face, new track id — exactly what ByteTrack does.
        renumbered = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[renumbered]])
        session._step(self.frame, 0.0)

        self.assertEqual(session.locked_id, 42, "did not re-acquire the new id")
        self.assertEqual(session.locked_name, "Harsh")
        self.assertEqual(session.lost_ticks, 0, "should not still count as lost")

    def test_re_acquire_keeps_commanding_the_head(self):
        """The point of re-acquiring: the head keeps moving, not just the lock."""
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        session = self._locked_session(head=head)
        renumbered = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[renumbered]])
        client.head_commands.clear()
        session._step(self.frame, 0.0)
        self.assertGreater(len(client.head_commands), 0,
                           "head stopped moving after the id changed")

    def test_a_different_person_is_not_re_acquired(self):
        """Re-acquiring must not grab whoever happens to be standing there."""
        harsh = unit_embedding(5)
        self.database.enroll_embedding(harsh, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], harsh),
                               self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)

        stranger = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[stranger]])
        session.recognizer = FakeRecognizer([[FACE_LEFT]], unit_embedding(888))
        session._step(self.frame, 0.0)

        self.assertNotEqual(session.locked_id, 42,
                            "locked onto a stranger while re-acquiring")
        self.assertGreater(session.lost_ticks, 0)

    def test_re_acquire_needs_a_known_identity(self):
        """Nothing to compare against means no re-acquisition, not a crash."""
        session = self._locked_session(dry_run=True)
        session.locked_slug = None
        renumbered = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[renumbered]])
        session._step(self.frame, 0.0)
        self.assertNotEqual(session.locked_id, 42)


class TestArmFollowing(PipelineTestCase):
    def test_the_arm_points_towards_a_person_on_the_left(self):
        from arm_control.arm_control import ArmPointer

        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        arm = ArmPointer(client=client, min_interval_s=0.0, min_bearing_change=0.0)
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, head=head, arm=arm)
        session._step(self.frame, 0.0)
        session._step(self.frame, 0.0)

        self.assertGreater(len(client.arm_commands), 0, "no arm command sent")
        posture, _, hand_index = client.arm_commands[-1]
        self.assertGreater(posture.orientation_.yaw_, 0.0,
                           "should point to the robot's left")
        self.assertEqual(hand_index, 0, "the left hand, for a person on the left")

    def test_an_arm_failure_disables_the_arm_and_keeps_following(self):
        """A refused arm pose must not end the demo."""
        from arm_control.arm_control import ArmPointer

        head_client = FakeClient()
        arm_client = FakeClient(fail_times=99, fail_code=502)
        head = HeadController(client=head_client, step_pause_s=0.0)
        arm = ArmPointer(client=arm_client, min_interval_s=0.0,
                         min_bearing_change=0.0)
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, head=head, arm=arm)
        session._step(self.frame, 0.0)
        session._step(self.frame, 0.0)

        self.assertIsNone(session.arm, "a failing arm should switch itself off")
        head_client.head_commands.clear()
        session._step(self.frame, 0.0)
        self.assertGreater(len(head_client.head_commands), 0,
                           "the head should keep following")


class TestNoLocomotionEndToEnd(PipelineTestCase):
    def test_a_whole_follow_sequence_sends_no_locomotion_command(self):
        """The legs must never receive anything, arm pointing or not.

        FakeClient adds Move, ChangeMode and stop_motion, which this project does
        not contain. Any call is recorded as a violation.
        """
        from arm_control.arm_control import ArmPointer

        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        arm = ArmPointer(client=client, min_interval_s=0.0, min_bearing_change=0.0)
        embedding = unit_embedding(5)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, head=head, arm=arm)

        session._step(self.frame, 0.0)             # unknown -> asks
        session.view.queue_name("Harsh")
        session._step(self.frame, 0.0)             # enrolled and locked
        for _ in range(10):
            session._step(self.frame, 0.0)         # following
        session._recenter_once()
        session.stop()

        self.assertEqual(client.violations, [],
                         "something reached for a locomotion call")
        self.assertGreater(len(client.head_commands), 0)

    def test_shutdown_recentres_the_head_and_lowers_the_arm(self):
        from arm_control.arm_control import ArmPointer

        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        arm = ArmPointer(client=client, min_interval_s=0.0, min_bearing_change=0.0)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, head=head, arm=arm)
        head.look_at(0.2, 0.5)
        arm.point_at(0.5)
        session.stop()

        last_pitch, last_yaw = client.head_commands[-1]
        self.assertAlmostEqual(last_pitch, 0.0, places=6)
        self.assertAlmostEqual(last_yaw, 0.0, places=6)
        self.assertTrue(arm.at_rest)

    def test_shutdown_survives_a_failing_head(self):
        """Cleanup must still close the camera and view if the robot errors."""
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        session = make_session(FakeDetector([[]]), FakeRecognizer(),
                               self.database, head=head)
        closed = []
        session.camera = type("C", (), {"close": lambda _self: closed.append(True)})()
        head.client = FakeClient(fail_times=99, fail_code=500)
        session.stop()
        self.assertEqual(closed, [True], "the camera was not closed")


# --- --lock-nearest: one person, followed, with others in the room ----------

# A second person further away: a much smaller box, so "nearest" is unambiguous.
PERSON_FAR = FakePerson((900, 300, 1000, 600), track_id=9)
FACE_FAR = FakeFace((930, 320, 970, 360))


class PerFaceRecognizer:
    """Like FakeRecognizer but the embedding depends on WHICH face is passed,
    so a test can put two different people in one frame."""

    def __init__(self, faces, embeddings):
        self.faces = list(faces)
        self.embeddings = dict(embeddings)

    def detect(self, _frame):
        return list(self.faces)

    def embed(self, _frame, face):
        return self.embeddings.get(tuple(face.box))


def nearest_session(detector, recognizer, database, head=None):
    """A FollowSession in --lock-nearest mode with every boundary faked."""
    session = FollowSession(parse_args(["--dry-run", "--lock-nearest"]))
    session.detector = detector
    session.recognizer = recognizer
    session.database = database
    session.view = StubView()
    session.head = head
    return session


class TestLockNearest(PipelineTestCase):
    def test_it_locks_at_once_without_waiting_for_a_name(self):
        """Locking must not depend on a name being supplied first.

        This used to assert that no name was EVER requested. That changed on
        purpose: the mode now asks who it locked onto once a storable face is
        visible, so a person enrolled in one run is recognised in the next. The
        guarantee that still matters is the ordering -- the head starts tracking
        on the first tick, and the naming happens afterwards.
        """
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                                  self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)
        self.assertEqual(len(self.database), 0,
                         "nothing may be saved before a name is given")

    def test_no_ask_keeps_the_old_save_nothing_behaviour(self):
        session = make_session(
            FakeDetector([[PERSON_LEFT]]),
            FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
            self.database, lock_nearest=True, dry_run=True, no_ask=True)
        for _ in range(8):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)
        self.assertFalse(session.view.asking)
        self.assertEqual(len(self.database), 0)

    def test_it_locks_even_when_no_face_can_be_read(self):
        """The point of this mode is that it never waits. A person facing away
        must still be locked and followed."""
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([], None), self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)
        self.assertEqual(session.anon_embeddings, [])
        self.assertIsNotNone(session.last_centre, "position must be remembered")

    def test_it_picks_the_nearest_of_several_people(self):
        """Scripted far-person-first, so passing proves it chooses by size and
        does not merely take whatever the detector listed first."""
        session = nearest_session(
            FakeDetector([[PERSON_FAR, PERSON_LEFT]]),
            PerFaceRecognizer([FACE_LEFT, FACE_FAR],
                              {FACE_LEFT.box: unit_embedding(5),
                               FACE_FAR.box: unit_embedding(600)}),
            self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7,
                         "locked onto the far person instead of the nearest")

    def test_a_new_track_id_is_re_acquired_by_face(self):
        mine = unit_embedding(5)
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], mine),
                                  self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)

        renumbered = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[renumbered]])
        session.recognizer = FakeRecognizer([[FACE_LEFT]], mine)
        session.tick = 0
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 42,
                         "same face on a new id should be re-acquired")

    def test_a_stranger_on_a_new_id_is_not_followed(self):
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                                  self.database)
        session._step(self.frame, 0.0)

        stranger = FakePerson(PERSON_LEFT.box, track_id=42)
        session.detector = FakeDetector([[stranger]])
        session.recognizer = FakeRecognizer([[FACE_LEFT]], unit_embedding(888))
        session.tick = 0
        session._step(self.frame, 0.0)
        self.assertNotEqual(session.locked_id, 42,
                            "followed a stranger who inherited the track id")

    def test_position_re_acquires_only_before_any_face_was_read(self):
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([], None), self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)

        nudged = FakePerson((220, 150, 480, 700), track_id=42)
        session.detector = FakeDetector([[nudged]])
        session.tick = 0
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 42,
                         "a small step should be recognised as the same person")

    def test_position_refuses_a_box_across_the_room(self):
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([], None), self.database)
        session._step(self.frame, 0.0)

        far = FakePerson((1000, 150, 1260, 700), track_id=42)
        session.detector = FakeDetector([[far]])
        session.tick = 0
        session._step(self.frame, 0.0)
        self.assertNotEqual(session.locked_id, 42,
                            "jumped the lock clear across the frame")

    def test_position_refuses_a_wildly_different_size(self):
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([], None), self.database)
        session._step(self.frame, 0.0)

        tiny = FakePerson((320, 400, 360, 500), track_id=42)
        session.detector = FakeDetector([[tiny]])
        session.tick = 0
        session._step(self.frame, 0.0)
        self.assertNotEqual(session.locked_id, 42)

    def test_the_lock_moves_when_the_track_id_is_handed_over(self):
        """Two people cross and the tracker gives our id to the wrong one. The
        lock must follow the FACE, not the id."""
        mine = unit_embedding(5)
        theirs = unit_embedding(600)
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], mine),
                                  self.database)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)

        # Track 7 now shows the other person's face; track 9 shows mine.
        impostor = FakePerson(PERSON_LEFT.box, track_id=7)
        real = FakePerson(PERSON_FAR.box, track_id=9)
        session.detector = FakeDetector([[impostor, real]])
        session.recognizer = PerFaceRecognizer(
            [FACE_LEFT, FACE_FAR],
            {FACE_LEFT.box: theirs, FACE_FAR.box: mine})
        session.tick = 0            # tick 0 runs the verify step
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 9,
                         "stayed on the id instead of following the person")

    def test_the_head_is_commanded_towards_the_locked_person(self):
        head = HeadController(client=FakeClient())
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                                  self.database, head=head)
        session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7)
        self.assertTrue(head.client.head_commands,
                        "locked on but sent the head nothing")
        _pitch, yaw = head.client.head_commands[-1]
        self.assertGreater(yaw, 0.0, "person is left of centre; yaw should be +")

    def test_nothing_is_written_to_disk_in_this_mode(self):
        session = nearest_session(FakeDetector([[PERSON_LEFT]]),
                                  FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                                  self.database)
        for _ in range(5):
            session.tick = 0
            session._step(self.frame, 0.0)
        self.assertEqual(len(self.database), 0)
        self.assertEqual(list((self.root / "people").glob("*")) if
                         (self.root / "people").is_dir() else [], [])


class TestArguments(unittest.TestCase):
    def test_defaults_are_the_safe_ones(self):
        args = parse_args([])
        self.assertFalse(args.arm, "arm pointing must be opt-in; its frame is unverified")
        self.assertFalse(args.raw, "the compressed topic is the default")
        self.assertFalse(args.dry_run)
        self.assertFalse(args.lock_nearest, "enrolment is the default flow")
        self.assertEqual(args.port, 8080)
        # These two are the behaviour the robot shows without being asked, so
        # they are the ones most worth pinning.
        self.assertTrue(args.hold,
                        "the lock must be held by default; re-locking onto a "
                        "different person is too surprising to be the default")
        self.assertEqual(args.speed, DEFAULT_SPEED)
        self.assertLess(args.speed, 1.0,
                        "the default head speed must be gentler than full gain")

    def test_no_hold_turns_holding_off(self):
        self.assertFalse(parse_args(["--no-hold"]).hold)
        self.assertTrue(parse_args([]).hold)

    def test_speed_can_be_raised_back_to_full(self):
        self.assertEqual(parse_args(["--speed", "1.0"]).speed, 1.0)

    def test_flags_parse(self):
        args = parse_args(["--dry-run", "--arm", "--window", "--port", "9000",
                           "--max-ticks", "5", "--lock-nearest"])
        self.assertTrue(args.dry_run and args.arm and args.window)
        self.assertTrue(args.lock_nearest)
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.max_ticks, 5)


if __name__ == "__main__":
    unittest.main()


# A stranger standing NEARER the robot than the enrolled person. The box is
# bigger, so the detector lists them first: this is the case where "lock onto
# one person" has to mean the right person, not the closest one.
STRANGER_NEAR = FakePerson((700, 100, 1100, 719), track_id=11)
STRANGER_FACE = FakeFace((820, 150, 960, 290))      # 140 px: readable


class TestMultiplePeople(PipelineTestCase):
    """Several people in the room, only one of them the one to follow."""

    def _two_people(self, dry_run=True, head=None):
        """Harsh enrolled and further away; an unknown person standing closer."""
        harsh = unit_embedding(5)
        stranger = unit_embedding(99)
        self.database.enroll_embedding(harsh, "Harsh")
        detector = FakeDetector([[STRANGER_NEAR, PERSON_LEFT]])
        recognizer = FakeRecognizer(
            [[STRANGER_FACE, FACE_LEFT]],
            embedding_for={tuple(FACE_LEFT.box): harsh,
                           tuple(STRANGER_FACE.box): stranger})
        return make_session(detector, recognizer, self.database,
                            head=head, dry_run=dry_run)

    def test_the_enrolled_person_is_locked_not_the_nearest(self):
        """The whole point: it must follow who it knows, not who is closest."""
        session = self._two_people()
        for _ in range(4):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh",
                         "locked %r instead of the enrolled person"
                         % (session.locked_name,))
        self.assertEqual(session.locked_id, PERSON_LEFT.track_id)

    def test_a_stranger_walking_closer_does_not_steal_the_lock(self):
        session = self._two_people()
        for _ in range(4):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, PERSON_LEFT.track_id)
        # Now hold both in view for a long stretch, past VERIFY_EVERY, so the
        # identity re-check runs repeatedly with the stranger still present.
        for _ in range(40):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh", "the lock was stolen")
        self.assertEqual(session.locked_id, PERSON_LEFT.track_id)

    def test_the_head_follows_the_enrolled_person_not_the_stranger(self):
        client = FakeClient()
        head = HeadController(client=client, step_pause_s=0.0)
        session = self._two_people(dry_run=False, head=head)
        for _ in range(4):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh")
        self.assertGreater(len(client.head_commands), 0, "no head command sent")
        _, yaw = client.head_commands[-1]
        # Harsh is on the LEFT of frame, the stranger on the RIGHT. Aiming at
        # the stranger would drive yaw the other way, so the sign is the proof.
        self.assertGreater(yaw, 0.0,
                           "head turned towards the stranger, not the "
                           "enrolled person (yaw %+.3f)" % yaw)

    def test_an_unknown_person_is_still_asked_about_when_nobody_is_known(self):
        """The improvement must not break first-time enrolment."""
        stranger = unit_embedding(99)
        detector = FakeDetector([[STRANGER_NEAR]])
        recognizer = FakeRecognizer([[STRANGER_FACE]], stranger)
        session = make_session(detector, recognizer, self.database, dry_run=True)
        session._step(self.frame, 0.0)
        self.assertTrue(session.view.asking,
                        "an empty database must still prompt for a name")


class TestHoldAndSpeed(PipelineTestCase):
    """--hold keeps the lock; --speed scales how fast the head travels."""

    def _locked(self, **flags):
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, dry_run=True, **flags)
        session._step(self.frame, 0.0)
        return session

    def test_with_no_hold_a_long_absence_unlocks(self):
        """--no-hold restores the old timer behaviour."""
        session = self._locked(hold=False)
        self.assertEqual(session.locked_id, 7)
        session.detector = FakeDetector([[]])
        session.recognizer = FakeRecognizer([[]])
        for _ in range(40):
            session._step(self.frame, 0.0)
        self.assertIsNone(session.locked_id, "--no-hold must still time out")

    def test_with_hold_the_lock_survives_a_long_absence(self):
        session = self._locked(hold=True)
        self.assertEqual(session.locked_id, 7)
        session.detector = FakeDetector([[]])
        session.recognizer = FakeRecognizer([[]])
        for _ in range(200):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_id, 7,
                         "--hold gave up the lock after %d ticks"
                         % session.lost_ticks)
        self.assertEqual(session.locked_name, "Harsh")

    def test_with_hold_a_different_person_never_takes_over(self):
        """The point of --hold: nobody else gets followed, ever."""
        session = self._locked(hold=True)
        stranger = unit_embedding(99)
        session.detector = FakeDetector([[STRANGER_NEAR]])
        session.recognizer = FakeRecognizer([[STRANGER_FACE]], stranger)
        for _ in range(200):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh",
                         "followed %r instead" % (session.locked_name,))
        self.assertNotEqual(session.locked_id, STRANGER_NEAR.track_id)

    def test_speed_scales_the_controller_gain(self):
        from follow_controller.follow_controller import KP_YAW as DEFAULT_YAW

        slow = self._locked(speed=0.3)
        self.assertAlmostEqual(slow.controller.kp_yaw, DEFAULT_YAW * 0.3,
                               places=6)
        normal = self._locked()
        self.assertAlmostEqual(normal.controller.kp_yaw,
                               DEFAULT_YAW * DEFAULT_SPEED, places=6)
        full = self._locked(speed=1.0)
        self.assertAlmostEqual(full.controller.kp_yaw, DEFAULT_YAW, places=6)

    def test_speed_is_clamped_to_something_sane(self):
        for given, expected in ((0.0, 0.05), (-3.0, 0.05), (99.0, 2.0)):
            session = self._locked(speed=given)
            self.assertAlmostEqual(session.speed, expected, places=6,
                                   msg="--speed %s became %s" % (given, session.speed))

    def test_a_slower_speed_moves_the_head_less_per_tick(self):
        """The observable effect: smaller commanded steps for the same error."""
        def travel(speed):
            client = FakeClient()
            head = HeadController(client=client, step_pause_s=0.0,
                                  max_step=0.10 * speed)
            embedding = unit_embedding(5)
            database = FaceDB({}, root=Path(self.root) / ("db%s" % speed))
            database.enroll_embedding(embedding, "Harsh")
            session = make_session(FakeDetector([[PERSON_LEFT]]),
                                   FakeRecognizer([[FACE_LEFT]], embedding),
                                   database, head=head, dry_run=False,
                                   speed=speed)
            session._step(self.frame, 0.0)
            return abs(client.head_commands[-1][1]) if client.head_commands else 0.0

        fast = travel(1.0)
        slow = travel(0.3)
        self.assertGreater(fast, 0.0, "no head command at full speed")
        self.assertLess(slow, fast,
                        "slow speed moved %.4f, full speed %.4f" % (slow, fast))


class TestNamingTheLockedPerson(PipelineTestCase):
    """--lock-nearest locks first, then asks who it locked onto."""

    def _session(self, database=None, **flags):
        flags.setdefault("lock_nearest", True)
        flags.setdefault("dry_run", True)
        return make_session(FakeDetector([[PERSON_LEFT]]),
                            FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                            database if database is not None else self.database,
                            **flags)

    def test_it_locks_before_it_knows_who_they_are(self):
        """The head must start tracking without waiting for a name."""
        session = self._session()
        session._step(self.frame, 0.0)
        self.assertIsNotNone(session.locked_id, "did not lock immediately")

    def test_an_unknown_locked_person_is_asked_about(self):
        session = self._session()
        for _ in range(6):
            session._step(self.frame, 0.0)
        self.assertTrue(session.view.asking,
                        "locked an unknown person without asking for a name")

    def test_a_name_is_saved_with_an_id_and_upgrades_the_lock(self):
        session = self._session()
        for _ in range(6):
            session._step(self.frame, 0.0)
        session.view.queue_name("Harsh")
        for _ in range(6):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh")
        self.assertEqual(session.locked_slug, "harsh", "lock did not adopt the identity")
        self.assertEqual(session.locked_person_id, "P001")
        self.assertEqual(len(self.database), 1)

    def test_the_saved_person_is_recognised_on_a_later_run(self):
        """The whole point of saving: the next run knows them without asking."""
        embedding = unit_embedding(5)
        first = self._session()
        for _ in range(6):
            first._step(self.frame, 0.0)
        first.view.queue_name("Harsh")
        for _ in range(6):
            first._step(self.frame, 0.0)
        self.assertEqual(len(self.database), 1)

        # A fresh run reading the same folder from disk, as a restart would.
        reloaded = FaceDB.load(root=self.root)
        self.assertEqual(len(reloaded), 1)
        second = make_session(FakeDetector([[PERSON_LEFT]]),
                              FakeRecognizer([[FACE_LEFT]], embedding),
                              reloaded, lock_nearest=True, dry_run=True)
        for _ in range(6):
            second._step(self.frame, 0.0)
        self.assertEqual(second.locked_name, "Harsh",
                         "a saved person was not recognised on the next run")
        self.assertEqual(second.locked_person_id, "P001")
        self.assertFalse(second.view.asking,
                         "asked for a name for somebody already saved")

    def test_a_known_person_is_never_asked_about(self):
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = self._session()
        for _ in range(8):
            session._step(self.frame, 0.0)
        self.assertFalse(session.view.asking)
        self.assertEqual(session.locked_slug, "harsh")

    def test_no_ask_follows_without_saving(self):
        session = self._session(no_ask=True)
        for _ in range(8):
            session._step(self.frame, 0.0)
        self.assertIsNotNone(session.locked_id, "--no-ask must still follow")
        self.assertFalse(session.view.asking, "--no-ask must not prompt")
        self.assertEqual(len(self.database), 0, "--no-ask must save nothing")

    def test_no_ask_still_recognises_a_known_face(self):
        embedding = unit_embedding(5)
        self.database.enroll_embedding(embedding, "Harsh")
        session = self._session(no_ask=True)
        for _ in range(8):
            session._step(self.frame, 0.0)
        self.assertEqual(session.locked_name, "Harsh",
                         "--no-ask should still use names it already has")

    def test_a_face_too_small_to_store_is_not_asked_about(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[TINY_FACE]], unit_embedding(5)),
                               self.database, lock_nearest=True, dry_run=True)
        for _ in range(8):
            session._step(self.frame, 0.0)
        self.assertFalse(session.view.asking)
        self.assertEqual(len(self.database), 0)
        self.assertIsNotNone(session.locked_id, "should still be following them")


class TestDuplicateEnrolment(PipelineTestCase):
    """One face must never end up under two names."""

    def test_the_same_face_under_a_second_name_is_merged(self):
        """Two identities sharing a face make the margin gate refuse both, so
        recognition would silently stop working for that person."""
        embedding = unit_embedding(5)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], embedding),
                               self.database, lock_nearest=True, dry_run=True)
        for _ in range(6):
            session._step(self.frame, 0.0)
        session.view.queue_name("Harsh")
        for _ in range(6):
            session._step(self.frame, 0.0)
        self.assertEqual(len(self.database), 1)

        # The same face offered again under a different spelling.
        session.view.queue_name("Harsh P")
        session._unlock("test")
        session.skipped_ids.clear()
        session.last_ask_ended = 0.0
        for _ in range(10):
            session._step(self.frame, 0.0)

        self.assertEqual(len(self.database), 1,
                         "a second identity was created for the same face: %s"
                         % list(self.database.names()))
        self.assertEqual(session.locked_name, "Harsh")

    def test_a_different_face_still_gets_its_own_identity(self):
        """The guard must not merge genuinely different people."""
        harsh = unit_embedding(5)
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], harsh),
                               self.database, lock_nearest=True, dry_run=True)
        for _ in range(6):
            session._step(self.frame, 0.0)
        session.view.queue_name("Harsh")
        for _ in range(6):
            session._step(self.frame, 0.0)

        stranger = unit_embedding(99)
        session.recognizer = FakeRecognizer([[STRANGER_FACE]], stranger)
        session.detector = FakeDetector([[STRANGER_NEAR]])
        session._unlock("test")
        session.skipped_ids.clear()
        session.last_ask_ended = 0.0
        for _ in range(6):
            session._step(self.frame, 0.0)
        session.view.queue_name("Zerui")
        for _ in range(6):
            session._step(self.frame, 0.0)

        self.assertEqual(len(self.database), 2,
                         "a different person was wrongly merged: %s"
                         % list(self.database.names()))


class TestQuietWhileAsking(PipelineTestCase):
    """Nothing may print between the prompt and the name being entered.

    Observed for real: a status line landed mid-input and a typed name came out
    as "h" on one line and "arsh" on the next, which makes the terminal path
    unusable at 12 Hz.
    """

    def _asking(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, lock_nearest=True, dry_run=True)
        for _ in range(6):
            session._step(self.frame, 0.0)
        self.assertTrue(session.view.asking, "fixture did not reach the prompt")
        return session

    def test_messages_are_held_back_while_asking(self):
        session = self._asking()
        session._say("  something that would interrupt typing")
        self.assertEqual(session._deferred,
                         ["  something that would interrupt typing"])

    def test_held_messages_are_printed_once_the_name_arrives(self):
        session = self._asking()
        session._say("  re-acquired the same person (0.71)")
        self.assertTrue(session._deferred)
        session.view.queue_name("Harsh")
        for _ in range(4):
            session._step(self.frame, 0.0)
        self.assertEqual(session._deferred, [],
                         "messages were still held after the prompt closed")

    def test_held_messages_are_released_on_skip_too(self):
        session = self._asking()
        session._say("  locked on person (track id 9)")
        session.view.queue_name("")          # Enter alone = skip
        for _ in range(3):
            session._step(self.frame, 0.0)
        self.assertEqual(session._deferred, [])

    def test_say_prints_normally_when_not_asking(self):
        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        session._say("  ordinary line")
        self.assertEqual(session._deferred, [],
                         "a line was held back when nothing was being typed")

    def test_re_acquiring_mid_prompt_keeps_the_question_open(self):
        """The real cause of a name never arriving.

        ByteTrack renumbers a person after a dropped frame, so re-acquisition
        fires while someone is still typing. Cancelling the prompt there throws
        away what they typed and asks again from scratch.
        """
        session = self._asking()
        asked_about = session.asking_about
        moved = FakePerson(PERSON_LEFT.box, track_id=asked_about + 93)
        session._lock_on(moved, "person", None, keep_prompt=True)
        self.assertIsNotNone(session.asking_about,
                             "the prompt was cancelled by a track-id change")
        self.assertEqual(session.asking_about, moved.track_id,
                         "the prompt must follow the person to their new id")
        self.assertTrue(session.view.asking, "the page stopped asking")

    def test_a_normal_lock_still_closes_the_prompt(self):
        """keep_prompt is for re-acquisition only; enrolment must end the ask."""
        session = self._asking()
        session._lock_on(PERSON_LEFT, "Harsh", "harsh")
        self.assertIsNone(session.asking_about)
        self.assertFalse(session.view.asking)

    def test_the_lock_message_does_not_interrupt_typing(self):
        session = self._asking()
        moved = FakePerson(PERSON_LEFT.box, track_id=(session.asking_about or 0) + 5)
        before = len(session._deferred)
        session._lock_on(moved, "person", None, keep_prompt=True)
        self.assertGreater(len(session._deferred), before,
                           "locked-on message printed straight through a prompt")


class TestMatchAcceptance(PipelineTestCase):
    """A name is only adopted when consecutive frames agree on it.

    A visitor was named as an enrolled colleague off a single frame scoring
    0.41. One frame above the threshold is not evidence.
    """

    def test_one_agreeing_frame_is_not_enough(self):
        from face_db.face_db import Match

        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        match = Match("harsh", "Harsh", 0.56, 0.0, person_id="P001")
        self.assertFalse(session._accept(match), "named somebody off one frame")

    def test_two_agreeing_frames_are(self):
        from face_db.face_db import Match

        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        match = Match("harsh", "Harsh", 0.56, 0.0, person_id="P001")
        session._accept(match)
        self.assertTrue(session._accept(match))

    def test_a_disagreeing_frame_resets_the_count(self):
        from face_db.face_db import Match

        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        harsh = Match("harsh", "Harsh", 0.56, 0.0, person_id="P001")
        other = Match("aditya", "Aditya", 0.58, 0.0, person_id="P002")
        session._accept(harsh)
        self.assertFalse(session._accept(other), "switched identity on one frame")
        self.assertTrue(session._accept(other))

    def test_no_match_resets_the_count(self):
        from face_db.face_db import Match

        session = make_session(FakeDetector([[PERSON_LEFT]]),
                               FakeRecognizer([[FACE_LEFT]], unit_embedding(5)),
                               self.database, dry_run=True)
        match = Match("harsh", "Harsh", 0.56, 0.0, person_id="P001")
        session._accept(match)
        session._accept(None)
        self.assertFalse(session._accept(match), "an unreadable frame left the vote standing")


class TestThresholdIsStrict(PipelineTestCase):
    def test_the_observed_false_positive_is_now_rejected(self):
        """0.41 named a visitor as an enrolled colleague. It must not again."""
        from face_db.face_db import MATCH_THRESHOLD

        self.assertGreater(MATCH_THRESHOLD, 0.41,
                           "the score that caused the misnaming still passes")

    def test_the_threshold_stays_under_measured_true_matches(self):
        """Degraded same-person views measured 0.70 and up; leave a cushion."""
        from face_db.face_db import MATCH_THRESHOLD

        self.assertLess(MATCH_THRESHOLD, 0.70)

    def test_recognition_has_a_face_size_floor(self):
        from pipeline.person_follow import (
            MIN_FACE_WIDTH_PX, MIN_FACE_WIDTH_RECOGNISE_PX)

        self.assertGreater(MIN_FACE_WIDTH_RECOGNISE_PX, 0)
        self.assertLessEqual(MIN_FACE_WIDTH_RECOGNISE_PX, MIN_FACE_WIDTH_PX,
                             "recognising at distance is useful; only STORING "
                             "a poor sample is harmful")
