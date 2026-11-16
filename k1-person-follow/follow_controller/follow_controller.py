"""
The control law — turns "the person is over there in the picture" into head angles.

This is visual servoing, not aiming. The camera rides ON the head, so an error
measured in the image IS roughly the angle the head must turn, and once it turns
the next frame already shows a smaller error. That is why the gains are low: at
gain 1.0 the head tries to cancel the whole error in one step, arrives a frame
late, and oscillates. KP_YAW = 0.55 settles in about 0.7 s with no overshoot at
10 Hz, and is the single most important number in this file.

At the 4-5 Hz measured over WiFi it settles nearer 1.5 s. Using the compressed
camera topic should raise the frame rate and bring that back down.

Nothing here talks to the robot or imports the SDK, so it is fully testable
offline — see tests/test_follow_offline.py.
"""

import math

# Camera intrinsics recorded from this robot's RealSense D455 at 1280x720
# (documentation-reference/booster-k1-camera-details.txt in the VLN project).
NATIVE_W = 1280
NATIVE_H = 720
FX = 643.864013671875
FY = 643.1196899414062
CX = 640.4425048828125
CY = 366.6420593261719

KP_YAW = 0.55               # see the module docstring before changing this
KP_PITCH = 0.45             # slightly lower: vertical travel is more limited

DEADBAND_RAD = 0.02         # about 1.1 degrees; ignore smaller errors so the head
                            # is not nudged forever by detector jitter
STALE_FRAME_S = 0.60        # older than this and the head freezes instead of acting


def intrinsics_for(frame_w, frame_h):
    """Intrinsics scaled to an actual frame size.

    The follow loop may run on a downscaled frame; the recorded numbers are for
    1280x720, so they have to be scaled rather than used blindly.
    """
    scale_x = frame_w / float(NATIVE_W)
    scale_y = frame_h / float(NATIVE_H)
    return FX * scale_x, FY * scale_y, CX * scale_x, CY * scale_y


def aim_point(person_box, face_box=None):
    """The pixel to put in the centre of the frame.

    The face when there is one, otherwise the top sixth of the body box. Never
    the body box's centre, which is a person's stomach: aiming there makes the
    robot stare at a torso.
    """
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    x1, y1, x2, y2 = person_box
    height = max(1, y2 - y1)
    return ((x1 + x2) / 2.0, y1 + height / 6.0)


def pixel_error(px, py, frame_w=NATIVE_W, frame_h=NATIVE_H):
    """(yaw_error, pitch_error) in radians to bring pixel (px, py) to centre.

    Sign convention matches head_control: positive yaw is the robot's LEFT,
    positive pitch is DOWN. A person left of centre has px < cx, which gives a
    positive yaw error, so the head turns left towards them.
    """
    fx, fy, cx, cy = intrinsics_for(frame_w, frame_h)
    yaw_error = math.atan2(cx - px, fx)
    pitch_error = math.atan2(py - cy, fy)
    return yaw_error, pitch_error


class FollowController:
    """Keeps a target head pose and updates it from each frame.

    It holds the target, not the head: head_control owns the stepping and the
    limits. Separating them is what makes this testable without a robot.
    """

    def __init__(self, kp_yaw=KP_YAW, kp_pitch=KP_PITCH, deadband=DEADBAND_RAD,
                 stale_after_s=STALE_FRAME_S):
        self.kp_yaw = kp_yaw
        self.kp_pitch = kp_pitch
        self.deadband = deadband
        self.stale_after_s = stale_after_s
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.updates = 0

    def sync(self, pitch, yaw):
        """Adopt the head's actual pose as the target, abandoning any in-flight move.

        Called when a frame is too stale to trust. Without this the head keeps
        completing a move planned from an old picture, which was a real bug in the
        earlier project: the head carried on turning towards where someone had been.
        """
        self.target_pitch = pitch
        self.target_yaw = yaw

    def update(self, person_box, face_box, frame_shape, head_pitch, head_yaw,
               frame_age_s=0.0):
        """New (target_pitch, target_yaw), or None when the head should hold.

        None means hold: either the frame is too old to act on, or the error is
        inside the deadband and moving would just be jitter.
        """
        if frame_age_s > self.stale_after_s:
            self.sync(head_pitch, head_yaw)
            return None

        frame_h, frame_w = frame_shape[:2]
        px, py = aim_point(person_box, face_box)
        yaw_error, pitch_error = pixel_error(px, py, frame_w, frame_h)

        if abs(yaw_error) < self.deadband and abs(pitch_error) < self.deadband:
            return None

        # The error is relative to where the head is NOW, so the correction is
        # added to the current pose rather than used as an absolute angle.
        self.target_yaw = head_yaw + self.kp_yaw * yaw_error
        self.target_pitch = head_pitch + self.kp_pitch * pitch_error
        self.updates += 1
        return self.target_pitch, self.target_yaw


def _simulate(person_x=320.0, person_top=200.0, ticks=25, max_step=0.10):
    """Step a simulated head towards a stationary person. Returns the tick log.

    The camera is ON the head, so turning the head shifts the whole scene the
    other way in the image. Both axes are modelled, and their signs are
    opposite: turning LEFT (+yaw) moves things RIGHT in the image (+px), while
    tilting DOWN (+pitch) moves things UP (-py). Getting that second sign wrong
    made an earlier version of this simulation diverge while the controller was
    in fact correct.
    """
    controller = FollowController()
    head_pitch, head_yaw = 0.0, 0.0
    box_width, box_height = 120.0, 500.0
    log = []

    for tick in range(1, ticks + 1):
        # Where the person appears NOW, given where the head is pointing.
        px = person_x + math.tan(head_yaw) * FX
        py = person_top - math.tan(head_pitch) * FY
        person_box = (px - box_width / 2, py, px + box_width / 2, py + box_height)

        result = controller.update(person_box, None, (720, 1280, 3),
                                   head_pitch, head_yaw)
        if result is None:
            log.append((tick, head_pitch, head_yaw, None))
            break
        target_pitch, target_yaw = result
        head_yaw += max(-max_step, min(max_step, target_yaw - head_yaw))
        head_pitch += max(-max_step, min(max_step, target_pitch - head_pitch))
        log.append((tick, head_pitch, head_yaw, (target_pitch, target_yaw)))
    return log


if __name__ == "__main__":
    print("Simulated convergence: person left of centre and above centre.")
    print("Expected bearing %+.4f rad yaw, %+.4f rad pitch.\n"
          % pixel_error(320.0, 200.0 + 500.0 / 6.0))

    for tick, pitch, yaw, target in _simulate():
        if target is None:
            print("tick %2d  yaw %+.4f  pitch %+.4f   HOLDING (inside deadband)"
                  % (tick, yaw, pitch))
        else:
            print("tick %2d  yaw %+.4f  pitch %+.4f   target yaw %+.4f pitch %+.4f"
                  % (tick, yaw, pitch, target[1], target[0]))

    print("\nIt should reach the deadband and hold, without overshooting past it.")
    print("If it oscillates instead, KP_YAW is too high.")
