import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

try:
    from gui_postprocessing import (  # noqa: E402
        _auto_match_sync_pairs,
        _sync_prefix_match_score,
        _sync_tokenize_stem,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - GUI deps (PySide6) may be absent
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"gui_postprocessing import failed: {_IMPORT_ERROR}")
class SyncFilenameMatchTests(unittest.TestCase):
    def test_tokenize_strips_noise_keeps_ids(self):
        toks = _sync_tokenize_stem(
            "A_028527_social_interaction_dLight_38_BLA_prep_crop_325_96_1533_1097_w_900_aspect_preprocessed.mp4"
        )
        # Noise words (crop/aspect/prep/preprocessed/w) are dropped...
        self.assertNotIn("preprocessed", toks)
        self.assertNotIn("crop", toks)
        # ...but numeric IDs (animal id 028527) are kept as key discriminators.
        self.assertEqual(toks[:4], ["a", "028527", "social", "interaction"])
        self.assertIn("028527", toks)

    def test_ain_suffix_stripped(self):
        toks = _sync_tokenize_stem("A_028527_social_interaction_0001_AIN01.csv")
        self.assertEqual(toks, ["a", "028527", "social", "interaction", "0001"])

    def test_prefix_score_counts_leading_overlap(self):
        a = _sync_tokenize_stem("A_028527_social_interaction_0001_AIN01.csv")
        b = _sync_tokenize_stem(
            "A_028527_social_interaction_dLight_38_BLA_prep_crop_325_96_1533_1097_w_900_aspect_preprocessed.mp4"
        )
        prefix, shared = _sync_prefix_match_score(a, b)
        self.assertEqual(prefix, 4)  # a, 028527, social, interaction
        self.assertGreaterEqual(shared, 4)

    def test_example_pair_matches(self):
        photometry = ["A_028527_social_interaction_0001_AIN01"]
        reference = [
            "A_028527_social_interaction_dLight_38_BLA_prep_crop_325_96_1533_1097_w_900_aspect_preprocessed"
        ]
        pairs = _auto_match_sync_pairs(photometry, reference)
        self.assertEqual(pairs, {photometry[0]: reference[0]})

    def test_greedy_one_to_one_no_reuse(self):
        photometry = [
            "A_028527_social_interaction_0001_AIN01",
            "A_031000_social_interaction_0001_AIN01",
            "A_044444_open_field_0001_AIN01",
        ]
        reference = [
            "A_044444_open_field_dLight_preprocessed",
            "A_028527_social_interaction_dLight_preprocessed",
            "A_031000_social_interaction_dLight_preprocessed",
        ]
        pairs = _auto_match_sync_pairs(photometry, reference)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(set(pairs.values())), 3)  # each reference used once
        self.assertIn("028527", pairs["A_028527_social_interaction_0001_AIN01"])
        self.assertIn("044444", pairs["A_044444_open_field_0001_AIN01"])

    def test_no_match_when_unrelated(self):
        pairs = _auto_match_sync_pairs(["mouse_A_session1"], ["completely_different_thing"])
        self.assertEqual(pairs, {})


if __name__ == "__main__":
    unittest.main()
