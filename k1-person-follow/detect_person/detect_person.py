"""
Person detection and tracking — YOLOv8n with ByteTrack.

Answers WHERE people are and gives each a track id that survives between
frames. It does not answer WHO they are; that is recognize_face's job.

Why tracking and not plain detection: a face is only readable when someone
faces the camera. Recognise the face once, pin that identity to the track id,
then follow the track. The head stays on the person when they turn away.

Model: yolov8n.pt. On a machine with no internet, put the file beside this
module or set K1_YOLO_MODEL to its path; ultralytics otherwise tries to
download it on first use. The robot itself has no internet.

Standalone, with no robot (reads an image file):
    python detect_person/detect_person.py some_photo.jpg
"""

import os
from pathlib import Path

PERSON_CLASS = 0            # "person" in the COCO classes YOLOv8 is trained on
CONFIDENCE = 0.30           # below the default; 0.25 flickers in and out
IMAGE_SIZE = 640            # YOLO input size; frames are letterboxed to this

_DEFAULT_MODEL = Path(__file__).resolve().parent / "yolov8n.pt"


class Person:
    """One detected person in one frame.

    box is (x1, y1, x2, y2) in pixels on the frame as given. track_id is None
    when the tracker has not settled on an id yet.
    """

    __slots__ = ("box", "confidence", "track_id")

    def __init__(self, box, confidence, track_id=None):
        self.box = tuple(int(round(value)) for value in box)
        self.confidence = float(confidence)
        self.track_id = track_id

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def contains(self, point):
        x, y = point
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def __repr__(self):
        return "Person(box=%s, conf=%.2f, id=%s)" % (self.box, self.confidence,
                                                     self.track_id)


def model_path():
    """Where the YOLO weights are expected. K1_YOLO_MODEL overrides."""
    override = os.environ.get("K1_YOLO_MODEL")
    if override:
        return Path(override)
    if _DEFAULT_MODEL.exists():
        return _DEFAULT_MODEL
    return Path("yolov8n.pt")          # let ultralytics resolve or download it


class PersonDetector:
    """Wraps one ultralytics model. Create it once and reuse it: loading the
    weights takes a second or two and tracker state lives inside the model.
    """

    def __init__(self, confidence=CONFIDENCE, image_size=IMAGE_SIZE, track=True):
        from ultralytics import YOLO

        self.confidence = confidence
        self.image_size = image_size
        self.track = track
        self.model = YOLO(str(model_path()))

    def find(self, bgr):
        """People in this frame, biggest box first.

        Biggest first because the nearest person is the one a demo is about,
        and because it makes "the only person in the frame" unambiguous.
        """
        if bgr is None:
            return []
        if self.track:
            # persist=True keeps ByteTrack state across calls, which is what
            # gives stable ids. It assumes frames arrive in order, as they do here.
            results = self.model.track(bgr, persist=True, verbose=False,
                                       conf=self.confidence, imgsz=self.image_size,
                                       classes=[PERSON_CLASS], tracker="bytetrack.yaml")
        else:
            results = self.model.predict(bgr, verbose=False, conf=self.confidence,
                                         imgsz=self.image_size, classes=[PERSON_CLASS])
        people = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            ids = boxes.id
            for index in range(len(boxes)):
                track_id = None
                if ids is not None:
                    track_id = int(ids[index].item())
                people.append(Person(boxes.xyxy[index].tolist(),
                                     boxes.conf[index].item(), track_id))
        people.sort(key=lambda person: person.area, reverse=True)
        return people


def person_with_id(people, track_id):
    """The person carrying this track id, or None."""
    if track_id is None:
        return None
    for person in people:
        if person.track_id == track_id:
            return person
    return None


def aim_box(person):
    """The part of a person's box to aim the head at: the top sixth.

    Never the box centre, which is somebody's stomach. When a face has been
    found this is only the fallback; the face box is better.
    """
    x1, y1, x2, y2 = person.box
    height = max(1, y2 - y1)
    return (x1, y1, x2, y1 + height // 6)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python detect_person/detect_person.py IMAGE")
        raise SystemExit(2)

    import cv2

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("could not read %s" % sys.argv[1])
        raise SystemExit(1)

    print("model: %s" % model_path())
    detector = PersonDetector(track=False)       # tracking needs a video, not one frame
    found = detector.find(image)
    print("%d person(s) in %dx%d" % (len(found), image.shape[1], image.shape[0]))
    for person in found:
        print("  %r  aim_box=%s" % (person, aim_box(person)))
