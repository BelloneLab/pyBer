"""Regression checks for clock and behavior/trajectory table inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyBer.behavior_import import detect_time_column, infer_table, read_behavior_csv


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


if __name__ == "__main__":
    unittest.main()
