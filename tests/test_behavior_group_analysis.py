"""Individual trials and animal-level groups follow the original pyBer pipeline."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySide6 import QtTest, QtWidgets

from analysis_core import ProcessedTrial
from behavior_zone_compare import group_recordings
from postprocessing_core import compute_psth_matrix
import test_postprocessing_empty_state as fixture


class BehaviorGroupTests(unittest.TestCase):
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def load_group(self):
        panel = self.panel
        time = np.arange(0., 50., .1)
        for index, (name, value, starts) in enumerate((
            ("mouse_a", 2., [10.]), ("mouse_b", 8., [10., 20., 30.]),
            ("mouse_no_bouts", 100., []), ("mouse_missing_signal", np.nan, [10.]))):
            signal = np.full(time.size, value)
            path = name + ".csv"
            panel._processed.append(ProcessedTrial(
                path=path, channel_id="AIN01", time=time.copy(), raw_signal=signal.copy(),
                raw_reference=np.zeros(time.size), output=signal, output_label="dFF"))
            on = np.asarray(starts, float)
            panel._behavior_sources[name] = {
                "kind": "binary_columns", "behaviors": {}, "time": np.array([]),
                "event_behaviors": {"attack": {"on": on, "off": on + 2., "dur": np.full(on.size, 2.)}},
                "import_report": {"format": "mamir", "identity": str(index + 1),
                                  "paired_path": path, "paired_index": index}}
        panel.tab_sources.setCurrentIndex(1)
        panel._refresh_behavior_list()
        panel._set_event_category("behavior")
        panel.combo_psth_normalization.setCurrentIndex(panel.combo_psth_normalization.findData("none"))
        panel.spin_resample.setValue(10.)
        panel.cb_filter_events.setChecked(False)
        panel._compute_psth()
        return panel.behavior_zone_panel

    def test_one_animal_all_events_and_group_equal_animal_weights(self):
        view = self.load_group()
        panel = self.panel
        panel.combo_individual_file.setCurrentText("mouse_b")
        panel._compute_psth()
        view._compute()
        self.assertEqual(panel._last_mat.shape[0], 3)
        np.testing.assert_allclose(panel._last_mat, 8.)
        self.assertEqual(len(view._last_rows), 1)
        self.assertEqual(view._last_rows[0]["events"], 3)
        self.assertEqual(view._last_rows[0]["during"], 8.)

        panel.tab_visual_mode.setCurrentIndex(1)
        panel._compute_psth()
        view._compute()
        self.assertEqual(panel._last_psth_display_level, "animals")
        self.assertEqual(panel._last_mat.shape[0], 2)
        np.testing.assert_allclose(panel._last_mat.mean(axis=0), 5.)
        row = view._last_group_rows[0]
        self.assertEqual((row["recordings"], row["events"], row["rejected"]), (2, 4, 1))
        self.assertEqual(row["during"], 5.)  # Not the trial-weighted value 6.5.
        self.assertAlmostEqual(row["during_sem"], 3.)
        self.assertEqual(row["recording_ids"], ["mouse_a", "mouse_b"])
        self.assertEqual(row["recordings_excluded"], 2)

    def test_main_group_trial_option_keeps_trials_and_individual_remains_available(self):
        self.load_group()
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(1)
        panel.cb_group_keep_trials.setChecked(True)
        panel._compute_psth()
        self.assertEqual(panel._last_psth_display_level, "trials")
        self.assertEqual(panel._last_mat.shape[0], 4)
        np.testing.assert_allclose(panel._last_mat.mean(axis=0), 6.5)
        panel.cb_exclude_low_event_animals.setChecked(True)
        panel.spin_min_events_per_animal.setValue(2)
        panel._compute_psth()
        self.assertEqual(panel._last_mat.shape[0], 3)
        np.testing.assert_allclose(panel._last_mat, 8.)
        self.assertIn("mouse_a", panel._psth_excluded_files)
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.combo_individual_file.setCurrentText("mouse_a")
        panel._compute_psth()
        self.assertEqual(panel._last_mat.shape[0], 1)
        np.testing.assert_allclose(panel._last_mat, 2.)

    def test_group_baseline_normalization_rejects_flat_animal_and_matches_core(self):
        self.load_group()
        panel = self.panel
        panel._processed[1].output = np.sin(panel._processed[1].time) + .1 * panel._processed[1].time
        panel.tab_visual_mode.setCurrentIndex(1)
        panel.combo_psth_normalization.setCurrentIndex(panel.combo_psth_normalization.findData("zscore"))
        panel._compute_psth()
        self.assertEqual(panel._group_labels, ["mouse_b"])
        proc = panel._processed[1]
        _time, matrix = compute_psth_matrix(
            proc.time, proc.output, [10., 20., 30.],
            (-panel.spin_pre.value(), panel.spin_post.value()),
            (panel.spin_b0.value(), panel.spin_b1.value()), panel.spin_resample.value(), normalization="zscore")
        np.testing.assert_allclose(panel._last_mat[0], matrix.mean(axis=0), atol=1e-12)
        # Existing subtraction mode can still use flat baselines.
        panel.combo_psth_normalization.setCurrentIndex(panel.combo_psth_normalization.findData("subtract"))
        panel._compute_psth()
        self.assertEqual(panel._group_labels, ["mouse_a", "mouse_b"])
        np.testing.assert_allclose(panel._last_mat[0], 0.)

    def test_explicit_pairing_survives_reorder_removal_and_project_save(self):
        self.load_group()
        panel = self.panel
        expected = {proc.path: panel._match_behavior_source(proc) for proc in panel._processed}
        panel._processed.reverse()
        for proc in panel._processed:
            self.assertIs(panel._match_behavior_source(proc), expected[proc.path])
        panel._processed = panel._processed[-2:]
        for proc in panel._processed:
            self.assertIs(panel._match_behavior_source(proc), expected[proc.path])
        with tempfile.TemporaryDirectory() as directory:
            project = str(Path(directory) / "group.h5")
            panel._save_project_h5(project)
            restored = panel._load_project_h5(project)
            panel._behavior_sources = restored["behavior_sources"]
            for proc in panel._processed:
                info = panel._match_behavior_source(proc)
                self.assertEqual(info["import_report"]["paired_path"], proc.path)

    def test_duplicate_recordings_cannot_be_counted_twice(self):
        view = self.load_group()
        panel = self.panel
        panel._processed.append(copy.deepcopy(panel._processed[1]))
        panel.tab_visual_mode.setCurrentIndex(1)
        view._compute()
        self.assertIn("unique", view.status.text())
        self.assertFalse(view.export_button.isEnabled())
        self.assertEqual(view.table.rowCount(), 0)

    def test_group_export_contains_counts_sem_windows_and_units(self):
        view = self.load_group()
        self.panel.tab_visual_mode.setCurrentIndex(1)
        view._compute()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "group.csv")
            with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(path, "")):
                view._export()
            row = pd.read_csv(path).iloc[0]
            self.assertEqual(row["scope"], "group")
            self.assertEqual((row["recordings"], row["events"]), (2, 4))
            self.assertEqual((row["during"], row["during_sem"]), (5., 3.))
            self.assertEqual(row["before_window_s"], view.before.value())
            self.assertEqual(row["signal_units"], "dFF")
            self.assertEqual(json.loads(row["recording_ids"]), ["mouse_a", "mouse_b"])

    def test_group_missing_behavior_uses_its_own_n_and_scope_changes_do_not_export_stale_rows(self):
        view = self.load_group()
        panel = self.panel
        # This animal has no annotations for attack, not a zero-valued response.
        panel._behavior_sources["mouse_b"]["event_behaviors"] = {}
        panel.tab_visual_mode.setCurrentIndex(1)
        view._compute()
        self.assertEqual(view._last_group_rows[0]["recordings"], 1)
        self.assertEqual(view._last_group_rows[0]["during"], 2.)
        panel.tab_visual_mode.setCurrentIndex(0)
        self.assertFalse(view.export_button.isEnabled())
        self.assertEqual(view._last_rows, [])
        panel.combo_individual_file.setCurrentText("mouse_a")
        QtTest.QTest.qWait(400)
        self.assertEqual(view._last_rows[0]["file_id"], "mouse_a")
        self.assertEqual(view._last_export_context["scope"], "individual")

    def test_partial_sync_is_reported_in_status_and_export_context(self):
        view = self.load_group()
        panel = self.panel
        panel._processed[0].sync_aligned_time = panel._processed[0].time.copy()
        panel.tab_visual_mode.setCurrentIndex(1)
        panel.cb_sync_use_aligned.setChecked(True)
        view._compute()
        self.assertEqual(view._last_export_context["photometry_clock"], "mixed")
        clocks = json.loads(view._last_export_context["recording_clocks"])
        self.assertEqual(clocks["mouse_a"], "sync_aligned")
        self.assertEqual(clocks["mouse_b"], "original")
        self.assertIn("original times are used", view.status.text())

    def test_sync_clock_change_invalidates_and_updates_comparison(self):
        view = self.load_group()
        panel = self.panel
        for proc in panel._processed:
            proc.sync_aligned_time = proc.time + 100.
        panel.tab_visual_mode.setCurrentIndex(1)
        view._compute()
        self.assertEqual(view._last_group_rows[0]["recordings"], 2)
        panel.cb_sync_use_aligned.setChecked(True)
        self.assertEqual(view._last_rows, [])
        QtTest.QTest.qWait(400)
        self.assertEqual(view._last_group_rows[0]["recordings"], 0)
        panel.cb_sync_use_aligned.setChecked(False)
        QtTest.QTest.qWait(400)
        self.assertEqual(view._last_group_rows[0]["recordings"], 2)


class GroupSummarySafetyTests(unittest.TestCase):
    def row(self, name, count=1, value=2.):
        return dict(file_id=name, label="attack", events=count, rejected=0, before=value,
                    during=value, after=value, during_minus_before=0., after_minus_before=0.)

    def test_duplicate_and_nonfinite_contributors(self):
        with self.assertRaisesRegex(ValueError, "Duplicate recording"):
            group_recordings([self.row("a"), self.row("a")])
        result = group_recordings([self.row("a"), self.row("b", value=np.nan), self.row("c", count=0)])[0]
        self.assertEqual(result["recordings"], 1)
        self.assertEqual(result["recordings_excluded"], 2)
        self.assertEqual(result["during"], 2.)
        self.assertTrue(np.isnan(result["during_sem"]))


if __name__ == "__main__":
    unittest.main()
