"""Regression checks for clock and behavior/trajectory table inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyBer.behavior_import import detect_onset_table, detect_time_column, infer_table, read_behavior_csv


class BehaviorImportTests(unittest.TestCase):
    def test_pykaboo_prefers_software_clock_without_rebasing(self):
        """Camera and software clocks stay selectable with their original offsets."""
        df = pd.DataFrame({"timestamp_camera": [50, 51, 52], "timestamp_software": [100, 101, 102],
                           "frame_id": [0, 1, 2], "live_detection_available": [1, 1, 1],
                           "behavior_rearing": [0, 1, -1], "mouse_1_center_x": [10, 11, -1],
                           "mouse_1_center_y": [20, 21, -1], "mouse_1_class_id": [0, 0, -1],
                           "mouse_1_confidence": [.8, .9, -1], "mouse_2_center_x": [-1]*3,
                           "mouse_2_center_y": [-1]*3, "mouse_2_class_id": [-1]*3})
        original = df.copy(deep=True)
        result = infer_table(df)
        self.assertEqual(result["time_column"], "timestamp_software")
        self.assertEqual(detect_time_column(df), "timestamp_software")
        np.testing.assert_equal(result["time"], [100, 101, 102])
        np.testing.assert_equal(result["time_candidates"]["timestamp_camera"], [50, 51, 52])
        self.assertEqual(set(result["behaviors"]), {"behavior_rearing"})
        np.testing.assert_equal(result["behaviors"]["behavior_rearing"], [0, 1, np.nan])
        self.assertEqual(result["default_coordinate_pair"], ("mouse_1_center_x", "mouse_1_center_y"))
        np.testing.assert_equal(result["trajectory"]["mouse_1_center_x"], [10, 11, np.nan])
        self.assertNotIn("mouse_2_center_x", result["trajectory"])
        pd.testing.assert_frame_equal(df, original)
        selected = infer_table(df, time_column="timestamp_camera")
        self.assertEqual(selected["auto_time_column"], "timestamp_software")
        np.testing.assert_equal(selected["time"], [50, 51, 52])

    def test_generic_binary_coordinates_remain_spatial(self):
        """Small tracks with coordinates zero or one are not behavior flags."""
        result = infer_table(pd.DataFrame({"time": [10, 11], "x": [0, 1], "y": [1, 0],
                                          "groom": ["false", "true"], "speed": [.3, .5]}))
        self.assertEqual(result["default_coordinate_pair"], ("x", "y"))
        self.assertEqual(set(result["behaviors"]), {"groom"})
        self.assertEqual(set(result["trajectory"]), {"x", "y", "speed"})

    def test_ethovision_and_invalid_clock_fallback(self):
        """Invalid aliases cannot hide an available valid acquisition clock."""
        df = pd.DataFrame({"time": [0, 0, 0], "Trial time": [5, 6, 7],
                           "X center": [1, 2, 3], "Y center": [4, 5, 6], "Grooming": [0, 1, 0]})
        result = infer_table(df)
        self.assertEqual(result["time_column"], "Trial time")
        self.assertEqual(result["default_coordinate_pair"], ("X center", "Y center"))
        with self.assertRaises(ValueError):
            infer_table(df, time_column="time")
        self.assertIsNone(infer_table(pd.DataFrame({"groom": [0, 1]}))["time_column"])

    def test_coordinate_validity_and_manual_override(self):
        """Detection confidence and keypoint visibility mask only imported copies."""
        df = pd.DataFrame({"time": [0, 1, 2], "mouse_1_center_x": [2, 3, 4],
                           "mouse_1_center_y": [4, 5, 6], "mouse_1_detected": [1, 0, 1],
                           "mouse_1_kp_1_x": [20, 30, 40], "mouse_1_kp_1_y": [40, 50, 60],
                           "mouse_1_kp_1_likelihood": [1, 1, 0], "latitude": [1, 2, 3],
                           "longitude": [3, 4, 5], "custom_clock": [10, 11, 12], "flag": [1, 0, 1]})
        result = infer_table(df, time_column="custom_clock", x_column="longitude", y_column="latitude", behavior_columns=["flag"])
        self.assertEqual(result["default_coordinate_pair"], ("longitude", "latitude"))
        np.testing.assert_equal(result["trajectory"]["mouse_1_center_x"], [2, np.nan, 4])
        np.testing.assert_equal(result["trajectory"]["mouse_1_kp_1_x"], [20, np.nan, np.nan])
        with self.assertRaises(ValueError):
            infer_table(df, x_column="longitude")
        with self.assertRaises(ValueError):
            infer_table(df, behavior_columns=["latitude"])

    def test_legacy_vector_states_and_explicit_precedence(self):
        """Old serialized states remain usable without duplicating direct columns."""
        result = infer_table(pd.DataFrame({"timestamp_software": [0, 1, 2],
                                          "behavior_state_vector": ["groom=0|rear=1", "groom=1|rear=0", "none"],
                                          "behavior_rear": [0, 0, 1]}))
        np.testing.assert_equal(result["behaviors"]["behavior_groom"], [0, 1, np.nan])
        np.testing.assert_equal(result["behaviors"]["behavior_rear"], [0, 0, 1])

    def test_delimiters_and_boolean_import(self):
        """CSV, semicolon and tab inputs use the same inference without renaming."""
        with tempfile.TemporaryDirectory() as directory:
            for separator in [",", ";", "\t"]:
                path = Path(directory) / "behavior.csv"
                path.write_text(separator.join(["time", "rear"]) + "\n" + separator.join(["0", "True"]) + "\n" + separator.join(["1", "False"]), encoding="utf-8-sig")
                result = infer_table(read_behavior_csv(path))
                np.testing.assert_equal(result["behaviors"]["rear"], [1, 0])

    def test_elapsed_time_text_keeps_offset_in_seconds(self):
        """Ethovision elapsed-duration strings are clocks, not synthetic indices."""
        df = pd.DataFrame({"Trial time": ["01:02:03.5", "01:02:04.5", "01:02:05.5"], "rear": [0, 1, 0]})
        self.assertEqual(detect_time_column(df), "Trial time")
        np.testing.assert_equal(infer_table(df)["time"], [3723.5, 3724.5, 3725.5])
        df["Trial time"] = ["02:03", "02:04", "02:05"]
        np.testing.assert_equal(infer_table(df)["time"], [123, 124, 125])


class OnsetTableDetectionTests(unittest.TestCase):
    """Columns of chronological onset times are recognized; sampled tables are not."""

    def test_tab_and_comma_onset_files_with_ragged_columns(self):
        """Ragged columns keep their own counts; values are returned as written."""
        rows = ["injection{0}CD1_entry{0}death", "5.6{0}325{0}400", "15.6{0}340{0}", "315{0}{0}"]
        with tempfile.TemporaryDirectory() as directory:
            for separator in ("\t", ",", ";"):
                path = Path(directory) / "onsets.csv"
                path.write_text("\n".join(row.format(separator) for row in rows), encoding="utf-8")
                found = detect_onset_table(read_behavior_csv(path))
                self.assertIsNotNone(found, separator)
                np.testing.assert_equal(found["onsets"]["injection"], [5.6, 15.6, 315])
                np.testing.assert_equal(found["onsets"]["CD1_entry"], [325, 340])
                np.testing.assert_equal(found["onsets"]["death"], [400])
                self.assertEqual(found["report"]["onset_counts"], {"injection": 3, "CD1_entry": 2, "death": 1})

    def test_elapsed_clock_text_index_column_and_empty_behavior(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1], "shock": ["00:01:05.5", "00:02:00"],
                           "escape": [np.nan, np.nan]})
        found = detect_onset_table(df)
        np.testing.assert_equal(found["onsets"]["shock"], [65.5, 120])
        self.assertEqual(found["onsets"]["escape"].size, 0)
        self.assertEqual(found["report"]["ignored_columns"], ["Unnamed: 0"])

    def test_times_in_any_order_are_sorted(self):
        """Times listed out of order (as typed by hand) are accepted and sorted."""
        found = detect_onset_table(pd.DataFrame({"injection": [5.6, 15.6, 315.], "death": [200., 210., 60.]}))
        np.testing.assert_equal(found["onsets"]["death"], [60, 200, 210])
        found = detect_onset_table(pd.DataFrame({"injection": [245., 454., 300.]}))
        np.testing.assert_equal(found["onsets"]["injection"], [245, 300, 454])

    def test_sampled_tables_keep_existing_parser(self):
        """Clocks, 0/1 states, gaps, negative values, frame counters and text are rejected."""
        rejected = {
            "clock": pd.DataFrame({"time": [0., 1., 2.], "x": [1., 2., 3.]}),
            "binary": pd.DataFrame({"groom": [0, 0, 1], "rear": [0, 1, 1]}),
            "interior gap": pd.DataFrame({"a": [1., np.nan, 3.]}),
            "negative": pd.DataFrame({"a": [1., -2., 3.]}),
            "frame counter": pd.DataFrame({"frame": [0, 1, 2], "t_ms": [0, 33, 66]}),
            "text": pd.DataFrame({"a": ["start", "stop"]}),
            "booleans": pd.DataFrame({"a": [False, True]}),
            "unnamed list": pd.DataFrame({"Unnamed: 0": [4., 9.], "a": [1., 2.]}),
            "empty": pd.DataFrame({"a": [np.nan, np.nan]}),
        }
        for name, df in rejected.items():
            with self.subTest(name):
                self.assertIsNone(detect_onset_table(df))


if __name__ == "__main__":
    unittest.main()
