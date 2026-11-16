"""The face database: enrolling, matching, the margin gate, and bad data."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from support import PROJECT_ROOT, unit_embedding  # noqa: F401

from face_db.face_db import (
    MATCH_THRESHOLD, MAX_SAMPLES, MIN_MARGIN, FaceDB, slugify)


class TestSlugify(unittest.TestCase):
    def test_names_become_safe_folder_names(self):
        self.assertEqual(slugify("Harsh"), "harsh")
        self.assertEqual(slugify("Harsh P"), "harsh-p")
        self.assertEqual(slugify("  Dedeepya  K  "), "dedeepya-k")
        self.assertEqual(slugify("O'Neill"), "o-neill")

    def test_a_name_with_no_usable_characters_still_gives_a_folder(self):
        self.assertEqual(slugify("!!!"), "person")
        self.assertEqual(slugify(""), "person")

    def test_path_separators_cannot_escape_the_folder(self):
        # A name is user input; it must not be able to write outside people/.
        self.assertNotIn("/", slugify("../../etc/passwd"))
        self.assertNotIn("..", slugify("../../etc/passwd"))


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="k1_face_db_"))
        self.database = FaceDB({}, root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestEnrolling(DatabaseTestCase):
    def test_enrolling_writes_the_expected_files(self):
        self.database.enroll_embedding(unit_embedding(1), "Harsh")
        folder = self.root / "harsh"
        self.assertTrue((folder / "meta.json").exists())
        self.assertTrue((folder / "embeddings.npy").exists())
        meta = json.loads((folder / "meta.json").read_text())
        self.assertEqual(meta["name"], "Harsh")
        self.assertEqual(meta["slug"], "harsh")
        self.assertEqual(meta["samples"], 1)

    def test_it_survives_a_restart(self):
        """The whole point: enrol once, recognised on every later run."""
        self.database.enroll_embedding(unit_embedding(1), "Harsh")
        reloaded = FaceDB.load(root=self.root)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded.names(), ["Harsh"])

    def test_a_second_sample_is_added_not_replaced(self):
        self.database.enroll_embedding(unit_embedding(1), "Harsh")
        person = self.database.enroll_embedding(unit_embedding(2), "Harsh")
        self.assertEqual(len(person.embeddings), 2)
        self.assertEqual(len(FaceDB.load(root=self.root)), 1)

    def test_samples_are_capped(self):
        for seed in range(MAX_SAMPLES + 6):
            person = self.database.enroll_embedding(unit_embedding(seed), "Harsh")
        self.assertEqual(len(person.embeddings), MAX_SAMPLES)

    def test_enrolling_nothing_is_an_error(self):
        with self.assertRaises(ValueError):
            self.database.enroll_embedding(None, "Harsh")

    def test_mismatched_embedding_length_is_rejected(self):
        self.database.enroll_embedding(unit_embedding(1, length=128), "Harsh")
        with self.assertRaises(ValueError):
            self.database.enroll_embedding(unit_embedding(2, length=64), "Harsh")


class TestMatching(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.harsh = unit_embedding(11)
        self.aditya = unit_embedding(22)
        self.database.enroll_embedding(self.harsh, "Harsh")
        self.database.enroll_embedding(self.aditya, "Aditya")

    def test_recognises_an_enrolled_person(self):
        match = self.database.match(self.harsh)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Harsh")
        self.assertGreater(match.score, MATCH_THRESHOLD)

    def test_a_stranger_is_not_matched(self):
        self.assertIsNone(self.database.match(unit_embedding(999)))

    def test_an_empty_database_matches_nobody(self):
        self.assertIsNone(FaceDB({}, root=self.root).match(self.harsh))

    def test_none_embedding_matches_nobody(self):
        self.assertIsNone(self.database.match(None))

    def test_below_threshold_is_rejected(self):
        # A noticeably different look at the same person: close enough to be the
        # best candidate, not close enough for a strict threshold.
        blurred = self.harsh + 0.9 * unit_embedding(77)
        blurred = blurred / np.linalg.norm(blurred)
        score = self.database.people["harsh"].best_score(blurred)
        self.assertLess(score, 0.95, "test fixture should not be a near-identical face")
        self.assertIsNone(self.database.match(blurred, threshold=0.95))
        self.assertIsNotNone(self.database.match(blurred, threshold=0.4, margin=0.0))

    def test_an_identical_embedding_scores_one(self):
        match = self.database.match(self.harsh)
        self.assertLessEqual(match.score, 1.0, "cosine must not exceed 1.0")
        self.assertAlmostEqual(match.score, 1.0, places=5)

    def test_the_margin_gate_refuses_a_near_tie(self):
        """Greeting a stranger by someone else's name is the worst outcome.

        With two enrolled people who look nearly identical to the model, the
        best score may clear the threshold while being barely ahead of the
        runner-up. That must not resolve into a confident name.
        """
        database = FaceDB({}, root=self.root)
        base = unit_embedding(31)
        nearly = base + 0.01 * unit_embedding(32)
        nearly = nearly / np.linalg.norm(nearly)
        database.enroll_embedding(base, "Twin One")
        database.enroll_embedding(nearly, "Twin Two")

        match = database.match(base)
        self.assertIsNone(match, "a near tie must not produce a name")

        # Relaxing only the margin lets it through, which proves the margin gate
        # is what refused it rather than the threshold.
        relaxed = database.match(base, margin=0.0)
        self.assertIsNotNone(relaxed)

    def test_margin_is_reported(self):
        match = self.database.match(self.harsh)
        self.assertAlmostEqual(match.margin, match.score - match.runner_up, places=6)
        self.assertGreaterEqual(match.margin, MIN_MARGIN)


class TestRemoving(DatabaseTestCase):
    def test_removing_deletes_the_folder_and_photos(self):
        self.database.enroll_embedding(unit_embedding(1), "Harsh")
        folder = self.root / "harsh"
        self.assertTrue(folder.exists())
        self.assertTrue(self.database.remove("harsh"))
        self.assertFalse(folder.exists())
        self.assertEqual(len(self.database), 0)

    def test_removing_accepts_the_display_name(self):
        self.database.enroll_embedding(unit_embedding(1), "Harsh P")
        self.assertTrue(self.database.remove("Harsh P"))
        self.assertEqual(len(self.database), 0)

    def test_removing_somebody_absent_reports_false(self):
        self.assertFalse(self.database.remove("nobody"))


class TestBadData(DatabaseTestCase):
    def test_a_missing_root_loads_as_empty(self):
        database = FaceDB.load(root=self.root / "does-not-exist")
        self.assertEqual(len(database), 0)

    def test_a_folder_without_embeddings_is_skipped(self):
        stray = self.root / "stray"
        stray.mkdir()
        (stray / "meta.json").write_text('{"name": "Stray"}')
        self.assertEqual(len(FaceDB.load(root=self.root)), 0)

    def test_corrupt_data_is_skipped_not_fatal(self):
        """One bad folder must not stop a demo."""
        self.database.enroll_embedding(unit_embedding(1), "Harsh")
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "meta.json").write_text("{not json")
        (broken / "embeddings.npy").write_bytes(b"not an npy file")

        database = FaceDB.load(root=self.root)
        self.assertEqual(len(database), 1)
        self.assertEqual(database.names(), ["Harsh"])

    def test_a_loose_file_in_people_is_ignored(self):
        (self.root / "notes.txt").write_text("hello")
        self.assertEqual(len(FaceDB.load(root=self.root)), 0)


if __name__ == "__main__":
    unittest.main()
