"""Verify inferred tables remain usable through GUI selection and project saves."""
from pathlib import Path
import tempfile
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import test_postprocessing_empty_state as gui_fixture
from gui_postprocessing import _load_behavior_csv, _behavior_table_info


class BehaviorImportGuiTests(unittest.TestCase):
    """Use isolated preferences and real widgets without touching user settings."""

    setUpClass = classmethod(gui_fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = gui_fixture.PostprocessingEmptyStateTests.setUp
    tearDown = gui_fixture.PostprocessingEmptyStateTests.tearDown

    def test_clock_switch_spatial_selection_and_project_roundtrip(self):
        """Changing clocks updates derived events and survives an HDF5 roundtrip."""
        table = pd.DataFrame({
            "timestamp_software": [100., 101., 102., 103.],
            "timestamp_camera": [50., 51., 52., 53.],
            "behavior_rearing": [0, 1, 1, 0],
            "mouse_1_center_x": [10., 11., 12., -1.],
            "mouse_1_center_y": [20., 21., 22., -1.],
            "mouse_2_center_x": [30., 31., 32., 33.],
            "mouse_2_center_y": [40., 41., 42., 43.],
            "live_detection_available": [1, 1, 1, 0],
        })
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "recording_metadata.csv"
            table.to_csv(csv_path, index=False)
            info = _load_behavior_csv(str(csv_path))
            panel = self.panel
            panel._behavior_sources = {"recording": info}
            panel._update_behavior_time_panel()
            panel._refresh_spatial_columns()
            self.assertEqual(panel.combo_spatial_x.currentText(), "mouse_1_center_x")
            self.assertEqual(panel.combo_spatial_y.currentText(), "mouse_1_center_y")
            self.assertEqual(set(info["behaviors"]), {"behavior_rearing"})
            np.testing.assert_equal(info["time"], [100, 101, 102, 103])
            self.assertTrue(np.isnan(info["trajectory"]["mouse_1_center_x"][-1]))
            info["event_behaviors"] = {"position": {
                "variable": "mouse_1_center_x", "rule": "> 10.5 and < 11.5", "align": "Align to onset",
            }}
            panel.combo_behavior_clock.setCurrentText("timestamp_camera")
            np.testing.assert_equal(info["time"], [50, 51, 52, 53])
            np.testing.assert_equal(info["event_behaviors"]["position"]["on"], [51])
            panel.combo_spatial_x.setCurrentText("mouse_2_center_x")
            panel.combo_spatial_y.setCurrentText("mouse_2_center_y")
            panel._refresh_spatial_columns()
            self.assertEqual(panel.combo_spatial_x.currentText(), "mouse_2_center_x")
            self.assertEqual(panel.combo_spatial_y.currentText(), "mouse_2_center_y")
            project_path = str(Path(directory) / "roundtrip.h5")
            panel._save_project_h5(project_path)
            saved = panel._load_project_h5(project_path)
            loaded = saved["behavior_sources"]["recording"]
            np.testing.assert_equal(loaded["time"], info["time"])
            for clock, values in info["time_candidates"].items():
                np.testing.assert_equal(loaded["time_candidates"][clock], values)
            self.assertEqual(loaded["auto_time_column"], "timestamp_software")
            self.assertEqual(loaded["import_report"], json.loads(json.dumps(info["import_report"])))
            panel._behavior_sources = {"recording": loaded}
            panel.combo_behavior_clock.setCurrentText("Auto")
            np.testing.assert_equal(loaded["time"], [100, 101, 102, 103])
            np.testing.assert_equal(loaded["event_behaviors"]["position"]["on"], [101])

    def test_generic_time_generation_and_explicit_event_timestamps(self):
        """FPS fallback and explicit event lists keep their existing semantics."""
        info = _behavior_table_info(pd.DataFrame({"groom": [0, 1, 0]}), "binary_columns", 2.)
        self.assertTrue(info["needs_generated_time"])
        np.testing.assert_equal(info["time"], [0, .5, 1])
        info = _behavior_table_info(pd.DataFrame({"timestamp_software": [100, 101, 102],
                                                "groom": [12., np.nan, 15.]}), "timestamp_columns", 30.)
        self.assertEqual(set(info["behaviors"]), {"groom"})
        np.testing.assert_equal(info["behaviors"]["groom"], [12, 15])

    def test_metadata_batch_matches_recordings_before_load_order(self):
        """Reversed metadata import order cannot swap the recording association."""
        panel = self.panel
        first = SimpleNamespace(path="29539_1_baseline.csv")
        second = SimpleNamespace(path="29539_2_baseline.csv")
        panel._processed = [first, second]
        one, two = {"recording": 1}, {"recording": 2}
        panel._behavior_sources = {"29539_2_baseline_metadata": two,
                                   "29539_1_baseline_metadata": one}
        self.assertIs(panel._match_behavior_source(first), one)
        self.assertIs(panel._match_behavior_source(second), two)
        panel._processed = []


if __name__ == "__main__":
    unittest.main()
