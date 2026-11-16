"""Measure what YOUR enrolled faces actually score, and pick a threshold from it.

Why this exists. A visitor was named as an enrolled colleague off a score of
0.41, with the threshold at 0.40. Any threshold is a guess until it is measured
against the faces it has to separate, in the room and at the distance it will
work in, so this measures it.

Two numbers decide everything:

  * the LOWEST score between two samples of the same person — true matches must
    stay above the threshold, so this is the floor it must sit under,
  * the HIGHEST score between samples of different people — impostors must stay
    below it, so this is the ceiling it must sit above.

A healthy database has a wide gap between them. A gap that is narrow or
inverted means recognition cannot be made reliable by moving the threshold, and
the enrolments themselves need redoing: more samples, closer, better lit.

Every score comes from the samples already stored in face_db/people/, so this
needs no robot and no camera.

    python face_db/calibrate.py
    python face_db/calibrate.py --pairs        # also list the worst pairs
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

# Running a script inside face_db/ puts that folder first on sys.path, where it
# shadows the package of the same name: "face_db" then resolves to face_db.py
# and "face_db.face_db" cannot be found. Drop the script's own folder, then add
# the project root.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [entry for entry in sys.path if entry not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from face_db.face_db import MATCH_THRESHOLD, FaceDB
from recognize_face.recognize_face import cosine


def within_person(person):
    """Every pairing of one person's own samples."""
    rows = list(person.embeddings)
    return [(cosine(a, b), i, j)
            for (i, a), (j, b) in combinations(list(enumerate(rows)), 2)]


def between_people(one, two):
    """Every pairing across two different people."""
    return [(cosine(a, b), i, j)
            for i, a in enumerate(one.embeddings)
            for j, b in enumerate(two.embeddings)]


def describe(values):
    if not values:
        return "no pairs"
    array = np.array(values, dtype="float64")
    return ("n=%-4d min %.3f  median %.3f  max %.3f"
            % (len(array), array.min(), np.median(array), array.max()))


def main():
    parser = argparse.ArgumentParser(
        description="Measure the score distributions of the enrolled faces")
    parser.add_argument("--pairs", action="store_true",
                        help="list the worst same-person and best cross-person pairs")
    args = parser.parse_args()

    database = FaceDB.load()
    people = list(database.people.values())
    print("database: %s" % database.root)
    print("enrolled: %d person(s)\n" % len(people))
    if not people:
        raise SystemExit("Nobody is enrolled yet, so there is nothing to measure.")

    for person in people:
        print("  %-22s %d sample(s)" % (person.label, len(person.embeddings)))
    print()

    same, same_detail = [], []
    for person in people:
        pairs = within_person(person)
        same += [score for score, _, _ in pairs]
        same_detail += [(score, person.name, i, j) for score, i, j in pairs]

    cross, cross_detail = [], []
    for one, two in combinations(people, 2):
        pairs = between_people(one, two)
        cross += [score for score, _, _ in pairs]
        cross_detail += [(score, one.name, two.name, i, j) for score, i, j in pairs]

    print("SAME person, sample against sample")
    print("  %s" % describe(same))
    if len(people) < 2:
        print("\nDIFFERENT people")
        print("  nothing to compare: only one person is enrolled.")
        print("  This is the dangerous case. With a single person on file there")
        print("  is no runner-up, so the margin gate passes on anything and the")
        print("  threshold alone decides. Enrol a second person and run this")
        print("  again — that is what turns the margin gate back on.")
    else:
        print("\nDIFFERENT people, sample against sample")
        print("  %s" % describe(cross))

    print("\n--- verdict ---")
    floor = min(same) if same else None
    ceiling = max(cross) if cross else None
    print("  current threshold: %.2f" % MATCH_THRESHOLD)
    if floor is not None:
        print("  lowest same-person score:  %.3f  (the threshold must be under this)"
              % floor)
    if ceiling is not None:
        print("  highest cross-person score: %.3f  (the threshold must be over this)"
              % ceiling)

    if floor is not None and ceiling is not None:
        gap = floor - ceiling
        if gap <= 0:
            print("\n  THE TWO OVERLAP by %.3f. No threshold separates these" % -gap)
            print("  faces: the same setting that accepts the real person also")
            print("  accepts somebody else. Re-enrol with more and better samples")
            print("  before trusting recognition — closer, facing the camera, and")
            print("  several views each.")
        else:
            # Deliberately NOT the midpoint. The midpoint best separates the
            # samples already on file, but the faces that cause harm are the
            # ones not in the database at all — a visitor, a passer-by — and
            # nothing here measures those. So the suggestion sits just under the
            # true-match floor, keeping the widest margin on the side where a
            # mistake puts somebody else's name on a person.
            suggested = round(max(min(floor - 0.15, 0.80),
                                  ceiling + 0.10, 0.40), 2)
            print("\n  gap between them: %.3f" % gap)
            print("  suggested threshold: %.2f" % suggested)
            print("  (just under the same-person floor, not midway: the risky")
            print("   faces are the ones NOT enrolled, and none are measured here)")
            if gap < 0.15:
                print("\n  That gap is narrow. It will hold in this lighting and may")
                print("  not in another room; add samples per person to widen it.")
            if suggested > MATCH_THRESHOLD + 0.02:
                print("\n  Raise it:  export K1_FACE_THRESHOLD=%.2f" % suggested)
            elif suggested < MATCH_THRESHOLD - 0.02:
                print("\n  The current %.2f is stricter than these faces need."
                      % MATCH_THRESHOLD)
                print("  Only lower it to %.2f if real people are being MISSED;"
                      % suggested)
                print("  a stricter threshold costs a second, a loose one costs a name.")
            else:
                print("\n  The current %.2f fits this data. Nothing to change."
                      % MATCH_THRESHOLD)
    elif floor is not None:
        print("\n  With one person enrolled the only measurable number is the")
        print("  same-person floor. Keep the threshold well under %.3f and enrol"
              % floor)
        print("  a second person to measure the other side.")

    if args.pairs:
        print("\n--- worst same-person pairs (closest to being rejected) ---")
        for score, name, i, j in sorted(same_detail)[:8]:
            print("  %.3f  %s  sample %d vs %d" % (score, name, i, j))
        if cross_detail:
            print("\n--- best cross-person pairs (closest to a false name) ---")
            for score, a, b, i, j in sorted(cross_detail, reverse=True)[:8]:
                print("  %.3f  %s sample %d  vs  %s sample %d" % (score, a, i, b, j))


if __name__ == "__main__":
    main()
