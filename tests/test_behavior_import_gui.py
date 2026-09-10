"""Verify inferred tables remain usable through GUI selection and project saves."""
from pathlib import Path
import tempfile
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import test_postprocessing_empty_state as gui_fixture
from gui_postprocessing import _load_behavior_csv, _behavior_table_info
from analysis_core import ProcessedTrial
from PySide6 import QtWidgets


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

    def test_project_open_restores_embedded_behavior_without_source_files(self):
        """Both project Open and signal/drop Open restore the complete snapshot."""
        panel = self.panel
        with tempfile.TemporaryDirectory() as directory:
            time = np.arange(0., 60., .1)
            source_path = Path(directory) / "recording_metadata.csv"
            pd.DataFrame({
                "timestamp_software": time,
                "timestamp_camera": time + .25,
                "rearing": ((time % 10 >= 4) & (time % 10 < 6)).astype(int),
                "mouse_center_x": np.sin(time), "mouse_center_y": np.cos(time),
            }).to_csv(source_path, index=False)
            panel._processed = [ProcessedTrial(
                path=str(Path(directory) / "recording.csv"), channel_id="AIN01",
                time=time, raw_signal=np.sin(time), raw_reference=np.cos(time),
                output=np.sin(time), output_label="dFF")]
            panel._load_behavior_paths([str(source_path)], replace=True)
            panel._refresh_behavior_list()
            panel.combo_behavior_clock.setCurrentText("timestamp_camera")
            panel.combo_behavior_name.setCurrentText("rearing")
            panel.spin_b0.setValue(-3.)
            panel.spin_b1.setValue(-1.)
            expected = panel._behavior_sources["recording_metadata"]
            project = str(Path(directory) / "snapshot.h5")
            panel._save_project_h5(project)
            # Existing linked files must not cause a reload prompt or override.
            with patch.object(QtWidgets.QMessageBox, "question") as question:
                self.assertTrue(panel._load_project_from_path(project))
                question.assert_not_called()
            source_path.unlink()
            for route in ("project", "signal", "drop"):
                with self.subTest(route=route):
                    panel._behavior_sources = {}
                    panel._processed = []
                    panel.combo_behavior_clock.setCurrentText("Auto")
                    with patch.object(QtWidgets.QMessageBox, "question") as question:
                        if route == "project":
                            self.assertTrue(panel._load_project_from_path(project))
                        elif route == "signal":
                            with patch.object(QtWidgets.QFileDialog, "getOpenFileNames", return_value=([project], "")):
                                panel.btn_load_processed_single.click()
                        else:
                            panel._on_preprocessed_files_dropped([project])
                        question.assert_not_called()
                    restored = panel._behavior_sources["recording_metadata"]
                    self.assertEqual(len(panel._processed), 1)
                    self.assertEqual(panel.combo_behavior_name.currentText(), "rearing")
                    self.assertEqual(panel.combo_behavior_clock.currentText(), "timestamp_camera")
                    self.assertIn("1 file(s) loaded", panel.lbl_beh.text())
                    np.testing.assert_equal(restored["time"], expected["time"])
                    np.testing.assert_equal(restored["behaviors"]["rearing"], expected["behaviors"]["rearing"])
                    for name, values in expected["trajectory"].items():
                        np.testing.assert_equal(restored["trajectory"][name], values)
                    for name, values in expected["time_candidates"].items():
                        np.testing.assert_equal(restored["time_candidates"][name], values)
                    self.assertGreater(np.asarray(panel._last_events).size, 0)

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
