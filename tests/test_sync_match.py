import os
import json
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

try:
    from gui_postprocessing import (  # noqa: E402
        PostProcessingPanel,
        _LED_SYNC_COLUMN,
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

    def test_led_sync_column_is_preferred_over_raw(self):
        columns = {
            "LED raw": [12.0, 13.0, 80.0],
            _LED_SYNC_COLUMN: [0.0, 0.0, 1.0],
        }
        picked = PostProcessingPanel._pick_sync_column(None, columns)
        self.assertEqual(picked, _LED_SYNC_COLUMN)

    def test_led_roi_is_bounded_to_video_frame(self):
        self.assertEqual(
            PostProcessingPanel._sync_bound_roi_tuple((708, 114, 5, 14), (644, 900)),
            (708, 114, 5, 14),
        )
        self.assertEqual(
            PostProcessingPanel._sync_bound_roi_tuple((999, 999, 5, 5), (644, 900)),
            (899, 643, 1, 1),
        )

    def test_led_roi_config_can_migrate_from_led_sync_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "session.mp4")
            open(video_path, "wb").close()
            csv_path = os.path.join(tmp, "session_LED_sync.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("# roi_x: 708\n")
                f.write("# roi_y: 114\n")
                f.write("# roi_w: 5\n")
                f.write("# roi_h: 14\n")
                f.write("# channel: Grayscale\n")
                f.write("# reduce: mean\n")
                f.write("# signal_format: binary\n")
                f.write("# start_frame: 0\n")
                f.write("# end_frame: 74244\n")
                f.write("time,LED signal,LED raw\n")
                f.write("0,0,63\n")
            config = PostProcessingPanel._sync_read_led_sync_csv_config(None, video_path)
            self.assertEqual(config["roi"], [708, 114, 5, 14])
            self.assertEqual(config["end_frame"], 74244)

    def test_led_roi_config_can_migrate_from_standalone_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "session.mp4")
            open(video_path, "wb").close()
            cache_dir = os.path.join(tmp, ".sig_cache")
            os.makedirs(cache_dir)
            meta_path = os.path.join(cache_dir, "abc123.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "video_path": video_path,
                        "roi": [706, 113, 5, 14],
                        "channel": "gray",
                        "start_frame": 0,
                        "end_frame": 74244,
                        "threshold": 128.0,
                    },
                    f,
                )
            config = PostProcessingPanel._sync_read_vbe_cache_config(PostProcessingPanel, video_path)
            self.assertEqual(config["roi"], [706, 113, 5, 14])
            self.assertEqual(config["channel"], "Grayscale")
            self.assertEqual(config["source"], "video_barcode_extractor_cache")
            self.assertEqual(config["threshold"], 128.0)


if __name__ == "__main__":
    unittest.main()
