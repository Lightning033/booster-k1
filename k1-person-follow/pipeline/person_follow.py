"""
Recognise a person, remember them, and follow them with the head (and arm).

What one run does:
  1. starts the camera and puts the live view on http://<this machine>:8080
  2. detects people in every frame
  3. the first time it sees a face it does not recognise, it asks for a name on
     that page (or in this terminal) and saves them to face_db/people/
  4. locks onto them and keeps the head pointed at them
  5. with --arm, also points a hand along the same bearing

Start order matters. The camera process starts BEFORE any Booster SDK call, so a
dead camera stops the run before the robot is sent a single command. ROS2 and
the SDK never share a process; see camera_process/camera_node.py.

The robot must already be standing in WALK mode. This file never changes the
robot's mode: a mode change can make the K1 shift its feet to rebalance, and
that should be a person's decision, not a side effect of starting a demo.

    python sdk_client/sdk_client.py walk      # separately, with a spotter

Run it:
    python pipeline/person_follow.py --dry-run     # watch only; head never moves
    python pipeline/person_follow.py               # head follows
    python pipeline/person_follow.py --arm         # head follows and a hand points

Keys in the desktop window (with --window): q quits, c recentres, u unlocks.
Ctrl+C anywhere stops: the head recentres, the arm lowers, the camera closes.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_process.camera_process import CameraProcess
from detect_person.detect_person import PersonDetector, person_with_id
from face_db.face_db import FaceDB, slugify
from follow_controller.follow_controller import (
    KP_PITCH, KP_YAW, FollowController, aim_point, pixel_error)
from live_view.live_view import LiveView, annotate
from recognize_face.recognize_face import (FaceRecognizer, cosine, face_in_box,
                                           models_present)

# How often to do the expensive things, in loop ticks.
FACE_SEARCH_EVERY = 2       # look for a face to identify, while nobody is locked
AIM_FACE_EVERY = 3          # refresh the face box, while locked (so the head aims at it)
VERIFY_EVERY = 30           # re-check the locked person's identity (about 3 s)

LOST_TICKS_BEFORE_UNLOCK = 25     # track gone this long -> give up the lock
LOST_TICKS_BEFORE_CENTER = 60     # nobody for this long -> recentre the head
REACQUIRE_EVERY = 2               # ticks between attempts to find the locked
                                  # person again by face after their track id
                                  # disappeared
MAX_REACQUIRE_CANDIDATES = 3      # how many people to face-check per attempt
MAX_SEARCH_CANDIDATES = 3         # how many people to face-check while looking
                                  # for someone to lock onto. More than one
                                  # matters as soon as the room has more than
                                  # one person in it.
MIN_FACE_WIDTH_PX = 54      # smaller faces are too far away to enrol reliably

# A name is only adopted after this many CONSECUTIVE frames agree on it. One
# frame is not evidence: a visitor was named as an enrolled colleague off a
# single 0.41, and a spurious score on one frame is exactly what that looks
# like. Two agreeing frames cost about a sixth of a second at 12 Hz.
MATCHES_TO_ACCEPT = 2

# Recognition is not attempted on a face narrower than this. A small face gives
# an unreliable embedding, and an unreliable embedding is what lands near the
# threshold by accident. Lower than the enrolment floor, because recognising a
# known person at distance is useful and only STORING a poor sample is harmful.
MIN_FACE_WIDTH_RECOGNISE_PX = 40
ASK_COOLDOWN_S = 8.0        # after a skip, wait before asking about anyone again
STATUS_EVERY_S = 1.0        # console line cadence

# How fast the head tracks, as a multiplier on the controller's full gain.
# Full gain settles a 30-degree error in 0.6 s, which peaks around 69 deg/s and
# reads as abrupt to anyone watching. 0.4 takes 1.5 s, which still keeps up with
# someone walking and looks deliberate rather than twitchy. Raise it with
# --speed if the head starts lagging behind.
DEFAULT_SPEED = 0.4

# --lock-nearest: follow the nearest person straight away, with no name and no
# enrolment. The lock still has to survive other people walking through the
# room, and the track id alone cannot do that -- ByteTrack reissues ids, and
# hands them over when two people cross. So on locking, the person's face is
# remembered IN MEMORY as an unnamed signature: enough to tell them from someone
# else for the rest of the run, written nowhere, gone when the run ends.
ANON_MATCH = 0.40           # same bar as the on-disk database
ANON_ADD_MIN = 0.55         # only add a second view of a face this certain
MAX_ANON_SAMPLES = 3        # a few poses is plenty; more just costs time

# When no face is readable at all, fall back to continuity: the same person is
# the box nearest where they were, at roughly the size they were. Loose enough
# to survive walking, tight enough not to jump to someone across the room.
SPATIAL_MAX_JUMP_FRAC = 0.25    # of frame width
SPATIAL_MAX_AREA_RATIO = 2.5

# For turning a face width in pixels into a rough distance, so a "come closer"
# message can say HOW much closer. fx was recorded from the K1's own camera
# info at 1280x720; a face is about 0.16 m across the detected box.
FOCAL_PX = 643.864
FACE_WIDTH_M = 0.16


def face_distance_m(face_width_px):
    """Rough metres to a face of this pixel width. None if not measurable."""
    if not face_width_px or face_width_px <= 0:
        return None
    return FACE_WIDTH_M * FOCAL_PX / float(face_width_px)


def too_small_message(face_width_px):
    """Why a seen face cannot be enrolled yet, with the distance to close."""
    here = face_distance_m(face_width_px)
    needed = face_distance_m(MIN_FACE_WIDTH_PX)
    if here is None or needed is None:
        return "face too small to enrol - come closer"
    return ("face too small to enrol: %d px, need %d - about %.1f m away, "
            "come inside %.1f m" % (face_width_px, MIN_FACE_WIDTH_PX, here, needed))


class FollowSession:
    """One run of the loop. Holds the lock state and every piece it needs."""

    def __init__(self, args):
        self.args = args
        self.camera = None
        self.view = None
        self.head = None
        self.arm = None
        self.detector = None
        self.recognizer = None
        self.database = None
        # --speed scales the controller gain and, further down, the head's
        # per-command step limit. Both matter: the gain decides how much of the
        # remaining error it tries to take out at once, the step limit caps how
        # far the head may travel per command. Lowering only the gain still
        # permits a fast lunge on a large error.
        # Note the explicit None check. `value or 1.0` is wrong here: 0.0 is
        # falsy, so --speed 0 would have become FULL speed rather than the
        # slowest, which is the most dangerous possible way to read that flag.
        given = getattr(args, "speed", None)
        speed = 1.0 if given is None else float(given)
        speed = max(0.05, min(speed, 2.0))
        self.speed = speed
        self.controller = FollowController(kp_yaw=KP_YAW * speed,
                                           kp_pitch=KP_PITCH * speed)

        self.locked_id = None
        self.locked_name = None
        self.locked_slug = None
        self.locked_person_id = ""
        self.face_box = None
        self.lost_ticks = 0
        self.asking_about = None        # track id currently being asked about
        self.skipped_ids = set()
        self.last_ask_ended = 0.0
        self.tick = 0
        self.new_frames = 0
        self.fps = 0.0
        self.reason = None          # why it is not locked, for the console line
        self.net_view = None
        self.tap = None
        self.layer_tap = None
        self.last_score = None      # cosine of the most recent identity match
        self.people_count = 0       # people in the newest frame
        self._deferred = []         # console lines held back during a prompt
        self._vote = (None, 0)      # (slug, consecutive frames agreeing)
        self.anon_embeddings = []   # unnamed signature of the locked person
        self.last_centre = None     # where they were, for the no-face fallback
        self.last_area = None

    # -- start up -------------------------------------------------------------
    def start(self):
        """Camera first, then models, then the robot. Order is deliberate."""
        ok, message = models_present()
        if not ok:
            raise SystemExit(message)

        print("Starting camera process ...")
        self.camera = CameraProcess(raw=self.args.raw, want_depth=False)
        print("  streaming from %s" % self.camera.info.get("topic", "?"))

        print("Loading face models and database ...")
        self.recognizer = FaceRecognizer()
        self.database = FaceDB.load()
        print("  %d person(s) enrolled: %s"
              % (len(self.database), ", ".join(self.database.names()) or "nobody yet"))

        print("Loading person detector ...")
        self.detector = PersonDetector()

        if self.args.net_view:
            from net_view.net_view import LayerTap, NetView, ScaleTap

            stream = ("http://localhost:%d/stream.mjpg" % self.args.port
                      if not self.args.no_web else None)
            self.net_view = NetView(port=self.args.net_port, stream_url=stream)
            if self.net_view.serve:
                self.tap = ScaleTap(self.detector)
                self.layer_tap = LayerTap(self.detector)
                print("\nDetector view:")
                for url in self.net_view.urls:
                    print("  %s" % url)
            else:
                self.net_view = None

        self.view = LiveView(port=self.args.port, serve_web=not self.args.no_web,
                             window=self.args.window)
        if self.view.serve_web:
            print("\nLive view:")
            for url in self.view.urls:
                print("  %s" % url)

        print("\nHead speed: %.2fx  (--speed 1.0 for full, 0.3 for gentler)"
              % self.speed)
        if self.args.hold:
            print("Lock: held for the whole run (--no-hold to re-lock on "
                  "whoever is nearest instead).")
        else:
            print("Lock: released after a few seconds out of view, then "
                  "re-locked onto whoever is nearest.")

        if self.args.dry_run:
            print("\nDRY RUN: the head will not move and the SDK is not touched.")
        else:
            from sdk_client.sdk_client import ensure_walking
            from head_control.head_control import MAX_STEP, HeadController

            print("\nChecking the robot ...")
            print("  mode: %s" % ensure_walking())
            # Same multiplier on the step limit, so --speed slows the actual
            # travel and not just the approach.
            self.head = HeadController(max_step=MAX_STEP * self.speed)
            if self.args.arm:
                from arm_control.arm_control import ArmPointer

                print("  ARM POINTING ON. Keep the space around the robot clear.")
                print("  If you have not run `python arm_control/arm_control.py armtest`,")
                print("  stop now and run it first: the arm frame is unverified.")
                self.arm = ArmPointer()
        self.view.set_status(state="searching", dry_run=self.args.dry_run,
                             enrolled_count=len(self.database),
                             arm="on" if self.arm else "off")

    # -- identity -------------------------------------------------------------
    def _identify(self, frame, person):
        """(match, face) for this person in this frame. Either may be None."""
        faces = self.recognizer.detect(frame)
        face = face_in_box(faces, person.box)
        if face is None:
            return None, None
        embedding = self.recognizer.embed(frame, face)
        if embedding is None:
            return None, face
        return self.database.match(embedding), face

    def _accept(self, match):
        """True once the same identity has been seen on enough frames in a row.

        A single frame above the threshold is not enough to put a name on
        somebody. Any disagreement — a different identity, or no match at all —
        resets the count, so only a steady reading is acted on.
        """
        if match is None:
            self._vote = (None, 0)
            return False
        slug, count = self._vote
        count = count + 1 if slug == match.slug else 1
        self._vote = (match.slug, count)
        return count >= MATCHES_TO_ACCEPT

    def _say(self, message):
        """Print, unless a name is being typed — then hold it until after.

        Anything printed while someone is typing lands in the middle of their
        input and breaks it up. These messages are worth keeping, so they wait
        rather than being dropped.
        """
        if self.asking_about is not None:
            self._deferred.append(message)
            return
        print(message)

    def _flush_deferred(self):
        for message in self._deferred:
            print(message)
        self._deferred = []

    def _lock_on(self, person, name, slug=None, embedding=None, person_id="",
                 keep_prompt=False):
        """Point the lock at this person.

        keep_prompt matters when the lock moves while a name is being typed.
        ByteTrack renumbers a person after a dropped frame, so re-acquisition
        fires mid-prompt — and cancelling the prompt there discards whatever has
        been typed so far and asks again from scratch. Observed for real: a name
        entered during a track-id change never arrived. The prompt is about the
        PERSON, so when it is the same person under a new id, it follows them.
        """
        was_asking = self.asking_about is not None
        self.locked_id = person.track_id
        self.locked_name = name
        self.locked_slug = slug
        self.locked_person_id = person_id or ""
        self.lost_ticks = 0
        if embedding is not None and len(self.anon_embeddings) < MAX_ANON_SAMPLES:
            self.anon_embeddings.append(embedding)
        self._remember_place(person)
        if keep_prompt and was_asking:
            self.asking_about = person.track_id     # the question follows them
        else:
            self.asking_about = None
            self.view.ask_name(False)
        label = ("%s %s" % (self.locked_person_id, name)).strip()
        self._say("  locked on %s (track id %s)" % (label, person.track_id))

    def _unlock(self, reason):
        if self.locked_id is not None:
            print("  unlocked (%s)" % reason)
        self.locked_id = None
        self.locked_name = None
        self.locked_slug = None
        self.locked_person_id = ""
        self.face_box = None
        self.anon_embeddings = []
        self.last_centre = None
        self.last_area = None

    def _remember_place(self, person):
        """Where the locked person is now, for the no-face fallback."""
        self.last_centre = person.centre
        self.last_area = person.area

    def _anon_score(self, embedding):
        """Best similarity between this face and the remembered signature."""
        if embedding is None or not self.anon_embeddings:
            return 0.0
        return max(cosine(embedding, known) for known in self.anon_embeddings)

    # -- enrolment ------------------------------------------------------------
    def _handle_enrolment(self, frame, person, face):
        """Ask for a name for this person, and enroll them when one arrives.

        Only asks when the face is big enough to be worth storing: a tiny face
        across the room makes a poor stored sample, and that sample is what every
        later recognition is compared against.
        """
        name = self.view.take_name()
        if name is not None:
            if name == "":
                print("  skipped; not enrolling this person")
                if person is not None and person.track_id is not None:
                    self.skipped_ids.add(person.track_id)
                self.asking_about = None
                self.last_ask_ended = time.monotonic()
                self.view.ask_name(False)
                self._flush_deferred()
                return
            # Guard against enrolling one face under two names. Two identities
            # holding the same face make every later match ambiguous, and the
            # margin gate then refuses to name either of them -- recognition
            # stops working for that person with no error anywhere. A typo or a
            # second spelling is all it takes, so the sample is added to the
            # person already on file instead of creating a rival identity.
            existing = self._already_enrolled(frame)
            if existing is not None and slugify(name) != existing.slug:
                print("  that face is already saved as %s." % existing.label)
                print("  Adding this view to them rather than creating a second")
                print("  identity, which would stop both from being recognised.")
                print("  (To rename, use: python face_db/face_db.py remove %s)"
                      % existing.slug)
                name = existing.name

            try:
                enrolled = self.database.enroll_from_frame(frame, name,
                                                           self.recognizer)
            except ValueError as exc:
                print("  could not enroll: %s (hold still, face the camera)" % exc)
                return
            self.asking_about = None           # the prompt is over; speak again
            self._flush_deferred()
            print("  saved %s to %s (%d sample(s))"
                  % (enrolled.label, enrolled.folder, len(enrolled.embeddings)))
            print("  they will be recognised by name on every later run.")
            self.view.set_status(enrolled_count=len(self.database))
            if person is not None:
                self._lock_on(person, enrolled.name, enrolled.slug,
                              person_id=enrolled.person_id)
            else:
                self.asking_about = None
                self.view.ask_name(False)
            return

        # Nothing typed yet. Keep asking only while a good face is still there.
        if face is None or face.width < MIN_FACE_WIDTH_PX:
            self.asking_about = None
            self.view.ask_name(False)
            self._flush_deferred()

    # -- one frame ------------------------------------------------------------
    def _step(self, frame, age_s):
        """Everything done for one new frame. Returns the annotated frame."""
        people = self.detector.find(frame)
        self.people_count = len(people)
        banner = None
        aim = None
        bearing = None

        locked_person = person_with_id(people, self.locked_id)

        # --- no lock: find somebody to lock onto ----------------------------
        if self.locked_id is None:
            self.view.set_status(state="searching", name=None, track_id=None)
            candidates = [person for person in people
                          if person.track_id not in self.skipped_ids]
            if not candidates:
                banner = "no one in view" if not people else "only skipped people in view"
            elif self.args.lock_nearest:
                # Lock the nearest person at once. Their face is remembered if it
                # can be read, but a readable face is NOT required to lock: the
                # point of this mode is that it never waits for anything.
                #
                # Nearest is chosen by box area here rather than by taking the
                # first candidate. The detector does sort biggest-first, but this
                # mode is defined by picking the nearest person, and that should
                # not rest on another module's ordering staying the way it is.
                person = max(candidates, key=lambda candidate: candidate.area)
                embedding = None
                face = face_in_box(self.recognizer.detect(frame), person.box)
                if face is not None:
                    self.face_box = face.box
                    embedding = self.recognizer.embed(frame, face)
                self._lock_on(person, "person", None, embedding=embedding)
                if embedding is None:
                    print("    (no face yet - holding them by position until"
                          " one can be read)")
                locked_person = person
            elif self.tick % FACE_SEARCH_EVERY == 0 or self.asking_about is not None:
                person, match, face = self._scan_candidates(frame, candidates)
                self.face_box = face.box if face else None
                if match is not None:
                    self._lock_on(person, match.name, match.slug)
                    locked_person = person
                elif face is not None and face.width >= MIN_FACE_WIDTH_PX:
                    cooling = time.monotonic() - self.last_ask_ended < ASK_COOLDOWN_S
                    if self.asking_about is None and not cooling:
                        self.asking_about = person.track_id
                        self.view.ask_name(True)
                        print("\n  unknown face. Who is this?")
                        if self.view.serve_web:
                            print("  type a name on the page, or here, then Enter:")
                        else:
                            print("  type a name here, then Enter:")
                    if self.asking_about is not None:
                        banner = "unknown person - waiting for a name"
                        self._handle_enrolment(frame, person, face)
                elif face is not None:
                    # A face IS being found and drawn, it is just too far away to
                    # make a stored sample worth comparing against later. This is
                    # the most common reason a run sits in "searching" forever,
                    # so say the measurement rather than something vague.
                    banner = too_small_message(face.width)
                else:
                    banner = "person seen, face not readable yet - face the camera"
            else:
                banner = "looking for a face"

        # --- locked: keep the head on them ----------------------------------
        if self.locked_id is not None:
            if locked_person is None:
                self.lost_ticks += 1
                banner = "lost sight of %s" % (self.locked_name or "them")
                # A single dropped frame makes ByteTrack issue NEW track ids, so
                # the locked id can vanish while the person is still plainly in
                # view. Waiting out the lost counter would leave the head frozen
                # for seconds with them standing right there. So look for them
                # again by FACE: the identity is the lock, the track id is only a
                # shortcut for frames where the face cannot be read.
                if people and self.tick % REACQUIRE_EVERY == 0:
                    locked_person = self._reacquire(frame, people)
                    if locked_person is not None:
                        banner = None
                # With --hold the lock is never given up on a timer. Dropping it
                # means the next tick locks whoever is nearest, and that is how
                # the robot ends up following a different person halfway through
                # a demo. Holding instead keeps looking for the same person for
                # as long as the run lasts; the head recentres and waits.
                if (locked_person is None
                        and not self.args.hold
                        and self.lost_ticks > LOST_TICKS_BEFORE_UNLOCK):
                    self._unlock("track gone for %d ticks" % self.lost_ticks)
                elif (locked_person is None and self.args.hold
                      and self.lost_ticks == LOST_TICKS_BEFORE_UNLOCK + 1):
                    print("  still holding %s; waiting for them to come back"
                          % (self.locked_name or "them"))

            if locked_person is not None:
                self.lost_ticks = 0
                self._remember_place(locked_person)
                self.view.set_status(state="following", name=self.locked_name,
                                     track_id=self.locked_id)

                if self.tick % AIM_FACE_EVERY == 0:
                    faces = self.recognizer.detect(frame)
                    found = face_in_box(faces, locked_person.box)
                    self.face_box = found.box if found else None
                    # Locked but not yet tied to a stored identity. Try to put a
                    # name to them: recognise them from the database, or ask.
                    if found is not None and self.locked_slug is None:
                        self._name_the_locked_person(frame, locked_person, found)

                    # Locked with no signature yet (they were facing away when
                    # the lock was taken). Take one now that a face is visible,
                    # so the lock stops depending on position.
                    if (found is not None and self.locked_slug is None
                            and not self.anon_embeddings):
                        embedding = self.recognizer.embed(frame, found)
                        if embedding is not None:
                            self.anon_embeddings.append(embedding)
                            self._say("  got a face for the locked person;"
                                      " holding them by identity now")

                if self.tick % VERIFY_EVERY == 0:
                    self._verify(frame, locked_person, people)

                aim = aim_point(locked_person.box, self.face_box or None)
                bearing = self._drive(frame, locked_person, aim, age_s)

        # Recentre after a long spell with nobody to aim at. With --hold the lock
        # is still held while this happens: the head goes back to centre and
        # waits there, rather than staying twisted towards wherever the person
        # was last seen.
        nobody_to_aim_at = self.locked_id is None or locked_person is None
        if nobody_to_aim_at and self.lost_ticks > LOST_TICKS_BEFORE_CENTER:
            self._recenter_once()

        self.reason = banner
        return annotate(frame, people=people, locked_id=self.locked_id,
                        locked_name=self.locked_name, face_box=self.face_box,
                        aim=aim, banner=banner, bearing_rad=bearing)

    def _scan_candidates(self, frame, candidates):
        """Look for a face we KNOW among the nearest few people.

        Returns (person, match, face); any of them may be None. A match means
        that person is enrolled and should be locked. No match means the
        returned person/face is the nearest readable face, which is the one to
        offer for enrolment.

        Checking several people rather than just the closest is what makes
        "lock onto one person" mean the right person. With only the nearest
        examined, anybody walking between the robot and the person it knows
        became an unknown face to ask about, and the person it was supposed to
        follow was never even looked at.

        Faces are detected once for the whole frame and the embedding is only
        computed for the few largest boxes, because the embedding is the
        expensive part of a tick.
        """
        faces = self.recognizer.detect(frame)
        if not faces:
            return None, None, None
        fallback = (None, None, None)
        for person in candidates[:MAX_SEARCH_CANDIDATES]:
            face = face_in_box(faces, person.box)
            if face is None:
                continue
            if fallback[0] is None:
                # Remember the nearest readable face whatever its size: a face
                # too small to enrol still needs to produce the "come closer"
                # message rather than silence.
                fallback = (person, None, face)
            embedding = self.recognizer.embed(frame, face)
            if embedding is None:
                continue
            match = self.database.match(embedding)
            if match is not None:
                return person, match, face
        return fallback

    def _already_enrolled(self, frame):
        """The person already on file with this face, or None.

        Used before saving a new name, so one face cannot end up under two
        identities. Looks at the largest face in the frame, which is the one
        enrol_from_frame is about to store.
        """
        try:
            found = self.recognizer.largest_face_embedding(frame)
        except Exception:
            return None
        if found is None:
            return None
        match = self.database.match(found[1])
        if match is None:
            return None
        return self.database.people.get(match.slug)

    def _name_the_locked_person(self, frame, person, face):
        """Put a stored identity on the person already locked.

        Two outcomes. If their face is in the database they are recognised and
        the lock adopts that name and id straight away, which is what makes a
        person enrolled in one run be known in the next. If they are not in it,
        a name is asked for on the terminal (and the page), and when one arrives
        they are saved and the lock upgrades from "some person" to that identity.

        Only reached while locked, which is the point: the robot has already
        committed to following somebody, so the question is who they are, not
        who to follow. That ordering is what lets the head start tracking
        immediately and the paperwork happen afterwards.
        """
        if self.args.no_ask and not self.database:
            return

        if face.width < MIN_FACE_WIDTH_RECOGNISE_PX:
            self.reason = too_small_message(face.width)
            return
        embedding = self.recognizer.embed(frame, face)
        if embedding is None:
            return

        match = self.database.match(embedding)
        if match is not None and not self._accept(match):
            return                      # promising, but one frame is not proof
        if match is not None:
            label = ("%s %s" % (match.person_id, match.name)).strip()
            print("  recognised %s (%.2f) - following them as a known person"
                  % (label, match.score))
            self.last_score = float(match.score)
            self.locked_name = match.name
            self.locked_slug = match.slug
            self.locked_person_id = match.person_id
            self.view.ask_name(False)
            self.asking_about = None
            self.view.set_status(name=label)
            return

        if self.args.no_ask:
            return

        # Unknown. Only worth asking once the face is big enough to store: the
        # sample saved now is what every later run compares against.
        if face.width < MIN_FACE_WIDTH_PX:
            self.reason = too_small_message(face.width)
            return
        if person.track_id in self.skipped_ids:
            return
        if time.monotonic() - self.last_ask_ended < ASK_COOLDOWN_S:
            return

        if self.asking_about != person.track_id:
            self.asking_about = person.track_id
            self.view.ask_name(True)
            print("\n  " + "-" * 62)
            print("  Locked onto an unknown person.")
            print("  Type a name here and press Enter to save their face.")
            if self.view.serve_web:
                print("  Or type it on the page: %s"
                      % (self.view.urls[0] if self.view.urls else "the live view"))
            print("  Press Enter alone to skip. The console is quiet until then.")
            print("  " + "-" * 62)
        self._handle_enrolment(frame, person, face)

    def _reacquire(self, frame, people):
        """Find the locked person again by face after their track id vanished.

        Returns the person now carrying their identity, or None. Only the few
        largest boxes are checked, because a face check is the expensive part of
        a tick and the person being followed is normally the nearest.
        """
        faces = self.recognizer.detect(frame)
        best = None            # (score, person, face) for the anon signature

        for person in people[:MAX_REACQUIRE_CANDIDATES]:
            face = face_in_box(faces, person.box) if faces else None
            if face is None:
                continue
            embedding = self.recognizer.embed(frame, face)
            if embedding is None:
                continue
            if self.locked_slug is not None:
                match = self.database.match(embedding)
                if match is not None and match.slug == self.locked_slug:
                    print("  re-acquired %s: track id %s -> %s"
                          % (match.name, self.locked_id, person.track_id))
                    self.face_box = face.box
                    self._lock_on(person, match.name, match.slug,
                                  keep_prompt=True)
                    return person
            else:
                score = self._anon_score(embedding)
                if score >= ANON_MATCH and (best is None or score > best[0]):
                    best = (score, person, face)

        if best is not None:
            score, person, face = best
            self._say("  re-acquired the same person (%.2f): track id %s -> %s"
                      % (score, self.locked_id, person.track_id))
            self.face_box = face.box
            self._lock_on(person, self.locked_name or "person", None,
                          keep_prompt=True)
            return person

        # No face to go on. Fall back to continuity, but only in --lock-nearest
        # and only while no signature exists yet. In the enrolment flow a lock
        # always came from a recognised face, so "nothing to compare against"
        # there means something has gone wrong, and guessing by position would
        # quietly hand the lock to whoever is standing nearby.
        if (getattr(self.args, "lock_nearest", False)
                and self.locked_slug is None and not self.anon_embeddings):
            return self._spatial_reacquire(frame, people)
        return None

    def _spatial_reacquire(self, frame, people):
        """Find the locked person again by position and size alone.

        Used only before any face has been read. The same person is the box
        nearest where they just were, at roughly the size they were; anything
        further or wildly different in size is somebody else.
        """
        if self.last_centre is None or not people:
            return None
        max_jump = frame.shape[1] * SPATIAL_MAX_JUMP_FRAC
        best = None
        for person in people:
            dx = person.centre[0] - self.last_centre[0]
            dy = person.centre[1] - self.last_centre[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > max_jump:
                continue
            if self.last_area:
                ratio = max(person.area, 1) / float(max(self.last_area, 1))
                if ratio > SPATIAL_MAX_AREA_RATIO or ratio < 1.0 / SPATIAL_MAX_AREA_RATIO:
                    continue
            if best is None or distance < best[0]:
                best = (distance, person)
        if best is None:
            return None
        distance, person = best
        print("  re-acquired by position (%.0f px): track id %s -> %s"
              % (distance, self.locked_id, person.track_id))
        self._lock_on(person, self.locked_name or "person", None)
        return person

    def _verify(self, frame, person, people=()):
        """Re-check that the locked track is still the person we think it is.

        Track ids get handed over when two people cross, which is the one real
        failure mode of the tracker. The lock is dropped only when a clearly
        visible face in the box is a DIFFERENT enrolled person: an unreadable
        face never drops it, because looking away is not a reason to lose someone.
        """
        if self.locked_slug is None:
            self._verify_anon(frame, person, people)
            return
        match, face = self._identify(frame, person)
        if face is None or match is None:
            return
        if self.locked_slug is not None and match.slug != self.locked_slug:
            print("  track %s is now %s, not %s" % (person.track_id, match.name,
                                                    self.locked_name))
            self._lock_on(person, match.name, match.slug)

    def _verify_anon(self, frame, person, people=()):
        """Keep an unnamed lock on the right person while others are about.

        Two jobs. While the face in the locked box still matches the remembered
        signature, add the occasional extra view so turning the head stops
        breaking recognition. When it clearly does NOT match, somebody else has
        inherited the track id -- so look at the others and move the lock to
        whoever does match, rather than following the wrong person.
        """
        if not self.anon_embeddings:
            return
        faces = self.recognizer.detect(frame)
        if not faces:
            return

        face = face_in_box(faces, person.box)
        if face is not None:
            embedding = self.recognizer.embed(frame, face)
            score = self._anon_score(embedding)
            if score >= ANON_MATCH:
                if (score >= ANON_ADD_MIN
                        and len(self.anon_embeddings) < MAX_ANON_SAMPLES):
                    self.anon_embeddings.append(embedding)
                return
            if embedding is None:
                return
        else:
            return      # nothing readable: never a reason to drop the lock

        # The face in this box is somebody else. Find the real one.
        for other in people:
            if other.track_id == person.track_id:
                continue
            other_face = face_in_box(faces, other.box)
            if other_face is None:
                continue
            other_embedding = self.recognizer.embed(frame, other_face)
            if other_embedding is None:
                continue
            if self._anon_score(other_embedding) >= ANON_MATCH:
                print("  track %s swapped to someone else; moving lock to %s"
                      % (person.track_id, other.track_id))
                self.face_box = other_face.box
                self._lock_on(other, self.locked_name or "person", None)
                return

    def _drive(self, frame, person, aim, age_s):
        """Point the head (and arm) at the person. Returns the bearing used."""
        head_pitch, head_yaw = (self.head.position() if self.head else
                                (self.controller.target_pitch, self.controller.target_yaw))
        result = self.controller.update(person.box, self.face_box, frame.shape,
                                        head_pitch, head_yaw, frame_age_s=age_s)

        # The person's bearing from the TORSO is where the head is pointing plus
        # the remaining error in the picture. That is what the arm needs: the head
        # may still be catching up.
        yaw_error, _ = pixel_error(aim[0], aim[1], frame.shape[1], frame.shape[0])
        bearing = head_yaw + yaw_error

        if result is None:
            self.view.set_status(head="%+.2f / %+.2f (hold)" % (head_pitch, head_yaw))
            return bearing

        target_pitch, target_yaw = result
        if self.head is not None:
            try:
                self.head.step_towards(target_pitch, target_yaw)
            except RuntimeError as exc:
                print("  head command failed: %s" % exc)
        else:
            # Dry run: move the believed pose so the overlay and the console still
            # show what the head WOULD do.
            self.controller.sync(target_pitch, target_yaw)
        self.view.set_status(head="%+.2f / %+.2f" % (target_pitch, target_yaw))

        if self.arm is not None:
            try:
                if self.arm.point_at(bearing):
                    self.view.set_status(arm="pointing %s (%+.2f)"
                                         % (self.arm.last_hand, bearing))
            except RuntimeError as exc:
                print("  arm command failed: %s" % exc)
                print("  turning arm pointing off for the rest of this run")
                self.arm = None
                self.view.set_status(arm="failed - off")
        return bearing

    def _recenter_once(self):
        """Bring the head back to centre after a long spell with nobody in view.

        step_towards moves one MAX_STEP at a time, so reaching centre from a
        large angle takes several calls. The lost counter is therefore only
        reset once centre is actually reached, otherwise the head would stop
        partway and sit there until the count climbed again.
        """
        pitch, yaw = (self.head.position() if self.head else
                      (self.controller.target_pitch, self.controller.target_yaw))
        if abs(pitch) < 0.02 and abs(yaw) < 0.02:
            self.lost_ticks = 0
            return
        if self.head is not None:
            try:
                self.head.step_towards(0.0, 0.0)
            except RuntimeError as exc:
                print("  head recentre failed: %s" % exc)
        else:
            self.controller.sync(0.0, 0.0)
        if self.arm is not None:
            try:
                self.arm.rest()
            except RuntimeError:
                pass

    # -- the loop -------------------------------------------------------------
    def run(self):
        print("\nRunning. Ctrl+C to stop.\n")
        window_start = time.monotonic()
        window_frames = 0
        last_status = 0.0

        last_annotated = None

        while True:
            frame, age_s, _ = self.camera.latest()
            if frame is None or not self.camera.is_new():
                # No new picture. Keep the window responsive to keys, but do no
                # perception and send no command: acting twice on one frame would
                # double-count the same error.
                time.sleep(0.01)
                if last_annotated is not None:
                    self._handle_key(self.view.pump_window(last_annotated))
                continue

            self.tick += 1
            self.new_frames += 1
            window_frames += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= 1.0:
                self.fps = window_frames / elapsed
                window_frames = 0
                window_start = time.monotonic()

            annotated = self._step(frame, age_s)
            last_annotated = annotated
            self.view.publish(annotated)
            self._feed_net_view(age_s)
            self.view.set_status(fps=self.fps, age_ms=age_s * 1000.0)
            self._handle_key(self.view.pump_window(annotated))

            now = time.monotonic()
            # While a name is being asked for, the console goes quiet. A status
            # line once a second lands in the middle of whatever is being typed
            # and scatters it across the output — observed for real: a name came
            # out as "h" on one line and "arsh" on the next. The prompt is the
            # only thing on screen until it is answered.
            if self.asking_about is not None:
                last_status = now
            elif now - last_status >= STATUS_EVERY_S:
                last_status = now
                state = ("following %s" % self.locked_name if self.locked_id is not None
                         else "searching")
                # The reason is the whole point of this line while nothing is
                # locked: without it a stalled run looks identical to an empty
                # room, and the only way to tell them apart is to ask someone.
                suffix = ""
                if self.locked_id is None and self.reason:
                    suffix = "   %s" % self.reason
                print("  %-22s %4.1f Hz   frame age %4.0f ms%s"
                      % (state, self.fps, age_s * 1000.0, suffix))

            if self.args.max_ticks and self.tick >= self.args.max_ticks:
                print("\nReached --max-ticks (%d); stopping." % self.args.max_ticks)
                return

    def _feed_net_view(self, age_s):
        """Hand the detector view its newest maps. Never fatal.

        The tap can only attach once the detector has run at least once, so
        maybe_attach is called every tick; it is a no-op after the first
        success. publish() throttles itself to the page's own rate.
        """
        if self.net_view is None:
            return
        try:
            items = None
            layers = None
            if self.tap is not None:
                self.tap.maybe_attach()
                items = self.tap.latest()
            if self.layer_tap is not None:
                self.layer_tap.maybe_attach()
                layers = self.layer_tap.latest()
            self.net_view.publish(items, layers=layers, stats={
                "fps": round(self.fps, 2),
                "age_ms": round(age_s * 1000.0),
                "people": self.people_count,
                "name": self.locked_name,
                "score": self.last_score,
            })
        except Exception as exc:
            print("  detector view failed (%s); switching it off" % exc)
            try:
                if self.tap is not None:
                    self.tap.detach()
                if self.layer_tap is not None:
                    self.layer_tap.detach()
                self.net_view.close()
            except Exception:
                pass
            self.net_view = None
            self.tap = None
            self.layer_tap = None

    def _handle_key(self, key):
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("c"):
            print("  recentring head")
            self._recenter_once()
        if key == ord("u"):
            self._unlock("unlock key pressed")

    # -- shut down ------------------------------------------------------------
    def stop(self):
        """Leave the robot tidy. Every step runs even if an earlier one fails."""
        print("\nStopping ...")
        try:
            if self.arm is not None:
                try:
                    print("  lowering arm")
                    self.arm.rest()
                except Exception as exc:
                    print("  arm rest failed: %s" % exc)
            if self.head is not None:
                try:
                    print("  recentring head")
                    self.head.center()
                except Exception as exc:
                    print("  head recentre failed: %s" % exc)
                # stop_motion() is deliberately NOT called. This project sends no
                # locomotion command, not even a zero one.
        finally:
            try:
                if self.tap is not None:
                    self.tap.detach()
                if self.layer_tap is not None:
                    self.layer_tap.detach()
                if self.net_view is not None:
                    self.net_view.close()
            except Exception:
                pass
            try:
                if self.view is not None:
                    self.view.close()
            finally:
                if self.camera is not None:
                    self.camera.close()
        print("  done. %d frame(s) processed." % self.new_frames)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Booster K1 — recognise, remember and head-follow a person")
    parser.add_argument("--dry-run", action="store_true",
                        help="never command the head; do not touch the SDK (run this first)")
    parser.add_argument("--arm", action="store_true",
                        help="also point a hand at the person (run arm_control armtest first)")
    parser.add_argument("--lock-nearest", action="store_true",
                        help="lock the nearest person at once and follow them; "
                             "no name asked, nothing saved to disk")
    parser.add_argument("--window", action="store_true",
                        help="also open a desktop window (q quit, c recentre, u unlock)")
    parser.add_argument("--no-web", action="store_true",
                        help="do not serve the live page")
    parser.add_argument("--port", type=int, default=8080, help="live page port")
    parser.add_argument("--raw", action="store_true",
                        help="use the uncompressed camera topic (slower over WiFi)")
    # Holding the lock is the DEFAULT. Re-locking onto whoever is nearest after
    # losing someone is what makes the robot follow a different person halfway
    # through a demo, and that is surprising behaviour to get without asking
    # for it. --no-hold restores the timer for the cases that want it.
    parser.add_argument("--net-view", action="store_true",
                        help="also serve a live view of the detector's own "
                             "per-scale response (read-only, separate port)")
    parser.add_argument("--net-port", type=int, default=8091,
                        help="port for the detector view (default 8091)")
    parser.add_argument("--no-ask", action="store_true",
                        help="never ask for a name. Known faces are still "
                             "recognised; unknown people are followed without "
                             "being saved")
    parser.add_argument("--no-hold", dest="hold", action="store_false",
                        default=True,
                        help="give up the lock after a few seconds of not "
                             "seeing the person, and then lock onto whoever is "
                             "nearest. The default is to hold the same person "
                             "for the whole run")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        metavar="FACTOR",
                        help="how fast the head moves, as a multiplier on the "
                             "controller's full gain. Scales both the gain and "
                             "the per-command step limit. 1.0 is full speed "
                             "(about 69 deg/s, which reads as abrupt); "
                             "default %.1f" % DEFAULT_SPEED)
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="stop after this many frames (for testing)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    session = FollowSession(args)
    try:
        session.start()
        session.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        session.stop()


if __name__ == "__main__":
    main()
