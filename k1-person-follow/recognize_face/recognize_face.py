"""
Face detection and embedding — OpenCV YuNet + SFace.

Two ONNX files loaded by the OpenCV that is already installed: no new pip
package, no internet at run time, nothing to install on the robot. InsightFace
is more accurate but pulls in onnxruntime and about 300 MB; dlib is painful to
build on Jetson hardware.

An embedding is a list of numbers that acts like a fingerprint for a face. Two
pictures of the same person give embeddings that point in nearly the same
direction, which is what cosine() measures.

Models go in recognize_face/models/:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
Copy them from the earlier booster-k1-person-recognition project, or fetch them
from the opencv_zoo repository on a machine with internet.

Standalone:
    python recognize_face/recognize_face.py photo.jpg
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

MODELS_DIR = Path(os.environ.get(
    "K1_FACE_MODELS", str(Path(__file__).resolve().parent / "models")))
DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

DETECT_SCORE = 0.80         # YuNet confidence for "this is a face"
DETECT_NMS = 0.30
DETECT_TOP_K = 50

# YuNet runs at this width, and it sets how far away a face can still be found.
# Measured on a 1280x720 frame (a face is about 0.16 m across, fx = 644):
#
#   detect_width   smallest face found   usable range   cosine at that range
#        640             39 px               2.6 m            0.95
#        960             29 px               3.6 m            0.83
#       1280             23 px               4.5 m            0.71
#
#   cost: 10 ms / 21 ms / 36 ms per frame
#
# All three stay well clear of the 0.40 match threshold even at their limit, so
# a wider detection buys range without costing accuracy. 960 is the default:
# 2.6 m is uncomfortably close for a demo, and 21 ms is affordable when the
# camera delivers a frame every 60-200 ms. Override with K1_FACE_DETECT_WIDTH.
DETECT_WIDTH = int(os.environ.get("K1_FACE_DETECT_WIDTH", "960"))


class Face:
    """One detected face: box (x1, y1, x2, y2) in pixels, plus its score."""

    __slots__ = ("box", "score", "landmarks")

    def __init__(self, box, score, landmarks=None):
        self.box = tuple(int(round(value)) for value in box)
        self.score = float(score)
        self.landmarks = landmarks

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

    def __repr__(self):
        return "Face(box=%s, score=%.2f)" % (self.box, self.score)


def cosine(a, b) -> float:
    """Cosine similarity of two embeddings: 1.0 identical, 0.0 unrelated.

    Clamped to [-1, 1]: in float32 an embedding compared with itself can come
    out as 1.0000001, which is harmless arithmetic but looks like a bug in a
    printed score and would break any caller that assumes the range.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b) / denominator)))


def face_in_box(faces: Sequence[Face], person_box) -> Optional[Face]:
    """The largest face whose centre falls inside person_box, or None.

    Used to decide which face belongs to the person being followed, so a face
    in the background cannot be credited to them.
    """
    x1, y1, x2, y2 = person_box
    inside = [face for face in faces
              if x1 <= face.centre[0] <= x2 and y1 <= face.centre[1] <= y2]
    if not inside:
        return None
    return max(inside, key=lambda face: face.area)


def models_present():
    """(ok, message) — whether both ONNX files are where they are expected."""
    missing = [path.name for path in (DETECTOR_MODEL, RECOGNIZER_MODEL)
               if not path.exists()]
    if missing:
        return False, ("Missing face model(s) in %s: %s\n"
                       "    Copy them from booster-k1-person-recognition/recognize_face/models/"
                       % (MODELS_DIR, ", ".join(missing)))
    return True, "ok"


class FaceRecognizer:
    """YuNet for finding faces, SFace for turning one into an embedding."""

    def __init__(self, detector_model=None, recognizer_model=None,
                 score=DETECT_SCORE, detect_width=DETECT_WIDTH):
        import cv2

        detector_model = Path(detector_model or DETECTOR_MODEL)
        recognizer_model = Path(recognizer_model or RECOGNIZER_MODEL)
        ok, message = models_present()
        if not ok and (not detector_model.exists() or not recognizer_model.exists()):
            raise FileNotFoundError(message)

        self._cv2 = cv2
        self.detect_width = int(detect_width)
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (self.detect_width, self.detect_width),
            score, DETECT_NMS, DETECT_TOP_K)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    def detect(self, bgr) -> List[Face]:
        """Faces in this frame, largest first, in the frame's own pixel coordinates."""
        if bgr is None:
            return []
        cv2 = self._cv2
        height, width = bgr.shape[:2]
        scale = 1.0
        working = bgr
        if width > self.detect_width:
            scale = self.detect_width / float(width)
            working = cv2.resize(bgr, (self.detect_width, int(round(height * scale))))
        work_height, work_width = working.shape[:2]
        self.detector.setInputSize((work_width, work_height))
        _, raw = self.detector.detect(working)
        faces = []
        if raw is None:
            return faces
        for row in raw:
            x, y, box_width, box_height = row[:4]
            # YuNet gives x, y, w, h. Undo the resize so boxes refer to the frame
            # the caller passed in, not the shrunken copy.
            faces.append(Face(
                (x / scale, y / scale, (x + box_width) / scale, (y + box_height) / scale),
                row[-1], landmarks=row[4:14].copy()))
        faces.sort(key=lambda face: face.area, reverse=True)
        return faces

    def embed(self, bgr, face: Face) -> Optional[np.ndarray]:
        """The embedding for one detected face, or None if it cannot be aligned.

        Alignment needs the landmarks YuNet produced at its own scale, so this
        re-detects on a crop around the face rather than trusting rescaled points.
        """
        height, width = bgr.shape[:2]
        x1, y1, x2, y2 = face.box
        pad = max(16, (x2 - x1) // 4)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        crop_height, crop_width = crop.shape[:2]
        self.detector.setInputSize((crop_width, crop_height))
        _, raw = self.detector.detect(crop)
        if raw is None or len(raw) == 0:
            return None
        best = max(raw, key=lambda row: row[2] * row[3])
        aligned = self.recognizer.alignCrop(crop, best)
        return self.recognizer.feature(aligned).copy()

    def detect_and_embed(self, bgr) -> List[Tuple[Face, np.ndarray]]:
        """Every face in the frame paired with its embedding."""
        pairs = []
        for face in self.detect(bgr):
            embedding = self.embed(bgr, face)
            if embedding is not None:
                pairs.append((face, embedding))
        return pairs

    def largest_face_embedding(self, bgr) -> Optional[Tuple[Face, np.ndarray]]:
        """The biggest face in the frame and its embedding, or None.

        This is the one to use when exactly one person is expected: the nearest
        face is the biggest.
        """
        faces = self.detect(bgr)
        if not faces:
            return None
        face = faces[0]
        embedding = self.embed(bgr, face)
        if embedding is None:
            return None
        return face, embedding


def draw_faces(bgr, faces, labels=None):
    """Boxes and optional labels drawn onto a COPY of the frame."""
    import cv2

    canvas = bgr.copy()
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = face.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 120), 2)
        if labels and index < len(labels) and labels[index]:
            cv2.putText(canvas, str(labels[index]), (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120), 2, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    import sys

    ok, message = models_present()
    print("models dir: %s" % MODELS_DIR)
    print(message)
    if not ok:
        raise SystemExit(1)

    if len(sys.argv) < 2:
        print("\nusage: python recognize_face/recognize_face.py IMAGE")
        raise SystemExit(2)

    import cv2

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("could not read %s" % sys.argv[1])
        raise SystemExit(1)

    recognizer = FaceRecognizer()
    pairs = recognizer.detect_and_embed(image)
    print("\n%d face(s) with embeddings in %dx%d"
          % (len(pairs), image.shape[1], image.shape[0]))
    for face, embedding in pairs:
        print("  %r  embedding length %d" % (face, embedding.size))
