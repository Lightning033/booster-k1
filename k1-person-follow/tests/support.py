"""
Fakes for the offline tests. Nothing here touches a robot, ROS2 or a network.

The important one is FakeClient. It provides the head and arm calls this project
is allowed to make, and it BAIT-TRAPS the ones it is not: Move, ChangeMode and
stop_motion do not exist in this project's own code, so the fakes add them. If
anything ever reaches for one, it is recorded as a violation rather than failing
as an ImportError that a future author might "fix" by writing the call for real.

This project does move an arm, which the earlier head-only project did not. The
legs are still off limits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


class FakeClient:
    """Stands in for B1LocoClient.

    Allowed: RotateHead, MoveHandEndEffectorV2.
    Trapped: Move, ChangeMode, stop_motion — any call is a violation.
    """

    def __init__(self, fail_times=0, fail_code=100):
        self.head_commands = []          # (pitch, yaw) as actually sent
        self.arm_commands = []           # (posture, time_millis, hand_index)
        self.violations = []
        self._fail_times = fail_times
        self._fail_code = fail_code

    # -- allowed --------------------------------------------------------------
    def RotateHead(self, pitch, yaw):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("RotateHead failed: code = %d" % self._fail_code)
        self.head_commands.append((pitch, yaw))
        return 0

    def MoveHandEndEffectorV2(self, posture, time_millis, hand_index):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("MoveHandEndEffectorV2 failed: code = %d"
                               % self._fail_code)
        self.arm_commands.append((posture, time_millis, hand_index))
        return 0

    # -- bait: this project must never call these -----------------------------
    def Move(self, vx, vy, vyaw):
        self.violations.append("client.Move(%s, %s, %s)" % (vx, vy, vyaw))
        return 0

    def ChangeMode(self, mode):
        self.violations.append("client.ChangeMode(%s)" % (mode,))
        return 0

    def stop_motion(self):
        self.violations.append("client.stop_motion()")
        return 0


class FakePosture:
    """Mimics the SDK's Posture closely enough to assert on."""

    def __init__(self, position, orientation):
        self.position_ = position
        self.orientation_ = orientation


class FakePosition:
    def __init__(self, x, y, z):
        self.x_, self.y_, self.z_ = float(x), float(y), float(z)

    def as_tuple(self):
        return (self.x_, self.y_, self.z_)


class FakeOrientation:
    def __init__(self, roll, pitch, yaw):
        self.roll_, self.pitch_, self.yaw_ = float(roll), float(pitch), float(yaw)

    def as_tuple(self):
        return (self.roll_, self.pitch_, self.yaw_)


class FakeHandIndex:
    kLeftHand = 0
    kRightHand = 1


class FakeSdkModule:
    """Stands in for the booster_robotics_sdk_python module."""

    Posture = FakePosture
    Position = FakePosition
    Orientation = FakeOrientation
    HandIndex = FakeHandIndex


def install_fake_sdk(monkey_target="booster_robotics_sdk_python"):
    """Put a fake SDK module in sys.modules. Returns a callable that removes it."""
    previous = sys.modules.get(monkey_target)
    sys.modules[monkey_target] = FakeSdkModule()

    def restore():
        if previous is None:
            sys.modules.pop(monkey_target, None)
        else:
            sys.modules[monkey_target] = previous
    return restore


# -----------------------------------------------------------------------------
# Perception fakes
# -----------------------------------------------------------------------------
class FakePerson:
    """Matches detect_person.Person's interface where the tests need it."""

    def __init__(self, box, track_id=None, confidence=0.9):
        self.box = tuple(int(v) for v in box)
        self.track_id = track_id
        self.confidence = confidence

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class FakeFace:
    def __init__(self, box, score=0.95):
        self.box = tuple(int(v) for v in box)
        self.score = score

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self):
        x1, _, x2, _ = self.box
        return max(0, x2 - x1)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class FakeDetector:
    """Returns a scripted list of people per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def find(self, _frame):
        if not self.script:
            return []
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[index]


class FakeRecognizer:
    """Returns scripted faces and deterministic embeddings.

    embedding_for maps a face box to the vector to return, so a test can make
    the same face read as the same person or a different one.
    """

    def __init__(self, faces_script=None, embedding=None, embedding_for=None):
        self.faces_script = list(faces_script or [])
        self.embedding = embedding
        # box -> embedding, so several people can be in one frame and each read
        # as a different identity. Without this every face in a frame comes back
        # as the same person, which cannot exercise the multi-person case.
        self.embedding_for = dict(embedding_for or {})
        self.calls = 0

    def detect(self, _frame):
        if not self.faces_script:
            return []
        index = min(self.calls, len(self.faces_script) - 1)
        self.calls += 1
        return self.faces_script[index]

    def embed(self, _frame, face):
        if self.embedding_for:
            box = tuple(face.box) if face is not None else None
            return self.embedding_for.get(box, self.embedding)
        return self.embedding

    def largest_face_embedding(self, frame):
        faces = self.detect(frame)
        if not faces or self.embedding is None:
            return None
        return faces[0], self.embedding


def grey_frame(width=1280, height=720, value=40):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = value
    return frame


def unit_embedding(seed, length=128):
    """A reproducible normalised embedding. Different seeds point elsewhere."""
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=length).astype(np.float32)
    return vector / np.linalg.norm(vector)
