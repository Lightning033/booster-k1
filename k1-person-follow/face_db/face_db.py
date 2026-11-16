"""
The face database — names and embeddings on local disk, reloaded every run.

Layout, one folder per person under face_db/people/:
    people/harsh/meta.json        name, slug, when, how many samples
    people/harsh/embeddings.npy   (N, 128) float32, one row per enrolled sample
    people/harsh/sample_01.jpg    the frame each sample came from

PRIVACY: people/ holds photographs of real people and the embeddings derived
from them. That is biometric data. It is in .gitignore and must stay out of any
repository, shared drive or backup. Ask each person before enrolling them, tell
them what it is for, and remove them afterwards:
    python face_db/face_db.py remove harsh

Matching is a cosine similarity with two gates:
  threshold  the best score must clear MATCH_THRESHOLD
  margin     it must also beat the runner-up by MIN_MARGIN
The margin gate is what stops a near-tie being resolved into a confident wrong
name. Greeting a stranger by someone else's name is worse than waiting a second.

Use:
    python face_db/face_db.py list
    python face_db/face_db.py enroll photo.jpg --name "Harsh"
    python face_db/face_db.py whois photo.jpg
    python face_db/face_db.py remove harsh
"""

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PEOPLE_DIR = Path(os.environ.get(
    "K1_FACE_DB", str(Path(__file__).resolve().parent / "people")))

# SFace's published operating point is 0.363. 0.40 was the first choice here and
# it turned out to be far too permissive in a real room: a visitor scored 0.41
# against an enrolled colleague and was named as them.
#
# The reason that could happen at all is the single-person case. The margin gate
# below compares the best match against the RUNNER-UP, and with one person
# enrolled there is no runner-up, so the margin is measured against zero and
# passes trivially. The threshold is then the only thing standing between any
# face and that person's name — there is no "none of the above".
#
# Measured on this project's own models (see face_db/calibrate.py, which will
# measure it again on your own enrolments):
#
#   same person, clean frontal        1.00
#   same person, shrunk to 140 px     0.76
#   same person, motion-blurred       0.71
#   same person, head tilted 28 deg   0.71
#   a different person                0.10 - 0.16
#
# True matches stay above 0.70 even badly degraded, so 0.55 keeps a wide cushion
# under them while rejecting the 0.41 that caused the misnaming. Override with
# K1_FACE_THRESHOLD once calibrate.py has told you what your own faces score.
MATCH_THRESHOLD = float(os.environ.get("K1_FACE_THRESHOLD", "0.55"))
MIN_MARGIN = float(os.environ.get("K1_FACE_MARGIN", "0.05"))
MAX_SAMPLES = 12            # per person; more adds little and slows matching


@dataclass
class Match:
    """A resolved identity. score is the winner, runner_up the next best."""
    slug: str
    name: str
    score: float
    runner_up: float = 0.0
    person_id: str = ""

    @property
    def margin(self):
        return self.score - self.runner_up


class Person:
    """One enrolled person and every embedding stored for them."""

    def __init__(self, slug, name, embeddings, folder=None, person_id=""):
        self.slug = slug
        self.name = name
        # A short stable label (P001, P002, ...) assigned when the person is
        # first enrolled and never reused, so a person can be referred to in
        # logs and on the overlay without depending on the spelling of a name.
        self.person_id = person_id or ""
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        self.folder = folder

    def best_score(self, embedding) -> float:
        """Highest cosine similarity between embedding and any stored sample."""
        from recognize_face.recognize_face import cosine

        if self.embeddings.size == 0:
            return 0.0
        return max(cosine(embedding, row) for row in self.embeddings)

    def __repr__(self):
        return "Person(%s %r, %d sample(s))" % (self.person_id or "-", self.name,
                                                len(self.embeddings))

    @property
    def label(self):
        """How to show this person: 'P001 Harsh', or just the name if unlabelled."""
        return ("%s %s" % (self.person_id, self.name)).strip()


def slugify(name: str) -> str:
    """A safe folder name from a person's name: 'Harsh P' -> 'harsh-p'."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return slug or "person"


class FaceDB:
    """Every enrolled person, loaded from disk."""

    def __init__(self, people: Dict[str, Person], root: Path = PEOPLE_DIR):
        self.people = people
        self.root = Path(root)

    @classmethod
    def load(cls, root: Path = PEOPLE_DIR) -> "FaceDB":
        """Read every person folder under root. A missing root is simply empty."""
        root = Path(root)
        people = {}
        if root.exists():
            for folder in sorted(root.iterdir()):
                person = cls._load_person(folder)
                if person is not None:
                    people[person.slug] = person
        return cls(people, root)

    @staticmethod
    def _load_person(folder: Path) -> Optional[Person]:
        """One person folder, or None if it is not one / is unreadable."""
        if not folder.is_dir():
            return None
        meta_path = folder / "meta.json"
        embeddings_path = folder / "embeddings.npy"
        if not meta_path.exists() or not embeddings_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            embeddings = np.load(embeddings_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # One corrupt folder must not stop a demo; say so and skip it.
            print("face_db: skipping %s (%s)" % (folder.name, exc))
            return None
        return Person(meta.get("slug", folder.name), meta.get("name", folder.name),
                      embeddings, folder=folder,
                      person_id=meta.get("person_id", ""))

    def __len__(self):
        return len(self.people)

    def names(self) -> List[str]:
        return [person.name for person in self.people.values()]

    def match(self, embedding, threshold=MATCH_THRESHOLD,
              margin=MIN_MARGIN) -> Optional[Match]:
        """Who this embedding belongs to, or None if nobody clears both gates."""
        if not self.people or embedding is None:
            return None
        scored = sorted(((person.best_score(embedding), person)
                         for person in self.people.values()),
                        key=lambda pair: pair[0], reverse=True)
        best_score, best_person = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < threshold:
            return None
        if best_score - runner_up < margin:
            return None                 # too close to call; wait for a better look
        return Match(best_person.slug, best_person.name, best_score, runner_up,
                     person_id=best_person.person_id)

    def enroll_embedding(self, embedding, name, frame=None) -> Person:
        """Add one embedding under name, creating or extending that person.

        frame, when given, is saved beside the embedding as the picture this
        sample came from — useful for checking later why a match misbehaved.
        """
        if embedding is None:
            raise ValueError("no embedding to enroll")
        slug = slugify(name)
        folder = self.root / slug
        folder.mkdir(parents=True, exist_ok=True)
        # An existing person keeps the id they were given; a new one gets the
        # next free number. Ids are never reused, so removing someone does not
        # hand their id to the next person enrolled.
        existing_person = self.people.get(slug)
        person_id = (existing_person.person_id if existing_person is not None
                     else "") or self.next_person_id()

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        existing = self.people.get(slug)
        if existing is not None and existing.embeddings.size:
            if existing.embeddings.shape[1] != embedding.shape[1]:
                raise ValueError(
                    "embedding length %d does not match the %d already stored for %r"
                    % (embedding.shape[1], existing.embeddings.shape[1], name))
            stacked = np.vstack([existing.embeddings, embedding])[-MAX_SAMPLES:]
        else:
            stacked = embedding

        np.save(folder / "embeddings.npy", stacked)
        (folder / "meta.json").write_text(json.dumps({
            "slug": slug,
            "person_id": person_id,
            "name": str(name),
            "samples": int(len(stacked)),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

        if frame is not None:
            import cv2

            index = len(stacked)
            cv2.imwrite(str(folder / ("sample_%02d.jpg" % index)), frame)

        person = Person(slug, str(name), stacked, folder=folder,
                        person_id=person_id)
        self.people[slug] = person
        return person

    def next_person_id(self) -> str:
        """The next free Pnnn, one past the highest already in use."""
        highest = 0
        for person in self.people.values():
            text = (person.person_id or "").strip().upper()
            if text.startswith("P") and text[1:].isdigit():
                highest = max(highest, int(text[1:]))
        return "P%03d" % (highest + 1)

    def enroll_from_frame(self, frame, name, recognizer) -> Person:
        """Find the biggest face in this frame and enroll it under name.

        This is the live path the follow loop uses: enrolling from the robot's
        own camera, in the demo room, at demo distance. A phone photo is a
        different lens, distance and light, and every one of those pushes the
        stored embedding away from what the robot will actually see.
        """
        found = recognizer.largest_face_embedding(frame)
        if found is None:
            raise ValueError("no usable face in this frame")
        _, embedding = found
        return self.enroll_embedding(embedding, name, frame=frame)

    def enroll(self, image_path, name=None, recognizer=None) -> Person:
        """Enroll from an image FILE. Weaker than enroll_from_frame; see above."""
        import cv2

        from recognize_face.recognize_face import FaceRecognizer

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError("could not read image %s" % image_path)
        recognizer = recognizer or FaceRecognizer()
        return self.enroll_from_frame(frame, name or image_path.stem, recognizer)

    def remove(self, slug: str) -> bool:
        """Delete a person and their stored photos. True if there was one."""
        slug = slugify(slug)
        person = self.people.pop(slug, None)
        folder = self.root / slug
        if folder.exists():
            shutil.rmtree(folder)
        return person is not None or folder.exists()


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def _cmd_list(_args):
    database = FaceDB.load()
    if not len(database):
        print("Nobody enrolled yet. The follow loop will offer to enroll "
              "the first person it sees.")
        return
    print("%d enrolled, in %s:" % (len(database), database.root))
    for person in database.people.values():
        print("  %-20s %-24s %d sample(s)"
              % (person.slug, person.name, len(person.embeddings)))


def _cmd_enroll(args):
    from recognize_face.recognize_face import FaceRecognizer

    database = FaceDB.load()
    name = args.name
    if not name:
        name = input("Name for %s: " % args.image).strip()
    if not name:
        print("no name given; nothing enrolled")
        raise SystemExit(1)
    person = database.enroll(args.image, name=name, recognizer=FaceRecognizer())
    print("enrolled %r as %s (%d sample(s))"
          % (person.name, person.slug, len(person.embeddings)))


def _cmd_whois(args):
    import cv2

    from recognize_face.recognize_face import FaceRecognizer

    database = FaceDB.load()
    if not len(database):
        print("nobody enrolled")
        raise SystemExit(1)
    frame = cv2.imread(args.image)
    if frame is None:
        print("could not read %s" % args.image)
        raise SystemExit(1)
    found = FaceRecognizer().largest_face_embedding(frame)
    if found is None:
        print("no usable face in that image")
        raise SystemExit(1)
    _, embedding = found
    print("scores against everyone enrolled:")
    for person in database.people.values():
        print("  %-20s %.3f" % (person.name, person.best_score(embedding)))
    match = database.match(embedding)
    if match is None:
        print("\nverdict: no confident match "
              "(needs score >= %.2f and a margin >= %.2f)"
              % (MATCH_THRESHOLD, MIN_MARGIN))
    else:
        print("\nverdict: %s  (score %.3f, margin %.3f)"
              % (match.name, match.score, match.margin))


def _cmd_remove(args):
    database = FaceDB.load()
    if database.remove(args.slug):
        print("removed %s" % slugify(args.slug))
    else:
        print("no such person: %s" % args.slug)
        raise SystemExit(1)


def main():
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(description="Booster K1 face database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="who is enrolled").set_defaults(
        function=_cmd_list)

    enroll_parser = subparsers.add_parser("enroll", help="enroll from an image file")
    enroll_parser.add_argument("image")
    enroll_parser.add_argument("--name", help="asked for if omitted")
    enroll_parser.set_defaults(function=_cmd_enroll)

    whois_parser = subparsers.add_parser("whois", help="score an image against everyone")
    whois_parser.add_argument("image")
    whois_parser.set_defaults(function=_cmd_whois)

    remove_parser = subparsers.add_parser("remove", help="delete a person and their photos")
    remove_parser.add_argument("slug")
    remove_parser.set_defaults(function=_cmd_remove)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
