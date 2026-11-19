"""Exercise long marker-aligned windows through the dashboard and saved project."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import test_postprocessing_empty_state as fixture
from PySide6 import QtCore, QtTest, QtWidgets
from postprocessing_core import compute_psth_matrix
from test_postprocessing_export_pipeline import _export_to_folder


class LongWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def enter(self, field, text):
        field.selectAll()
        QtTest.QTest.keyClicks(field, text)
        QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)

    def test_unit_entry_inline_controls_and_old_settings(self):
        p = self.panel
        p.combo_window_unit.setCurrentText("Minutes")
        self.enter(p.spin_post, "30")
        self.assertEqual(p.spin_post.value(), 1800)
        self.assertEqual(p.behavior_psth_bar.post.value(), 1800)
        p.combo_window_unit.setCurrentText("Hours")
        self.assertAlmostEqual(float(p.spin_post.cleanText()), .5)
        self.enter(p.behavior_psth_bar.post, "1.25")
        self.assertEqual(p.spin_post.value(), 4500)
        self.enter(p.spin_metric_pre0, "-0.5")
        self.assertEqual(p.spin_metric_pre0.value(), -1800)
        p.spin_pre.setValue(7.21)
        p.combo_window_unit.setCurrentText("Minutes")
        self.assertEqual(p.spin_pre.value(), 7.21)
        self.assertEqual(p._collect_settings()["window_post"], 4500)
        p._apply_settings({"window_pre": 1.25, "window_post": 5.25})
        self.assertEqual(p.combo_window_unit.currentData(), "s")
        self.assertEqual((p.spin_pre.value(), p.spin_post.value()), (1.25, 5.25))

    def recording(self, name, last, marker):
        time = np.arange(last + 1, dtype=float)
        signal = np.sin(time / 30)
        return fixture.ProcessedTrial(path=name, channel_id="AIN01", time=time,
                                      raw_signal=signal, raw_reference=signal, output=signal,
                                      output_label="dFF", injection_seconds=marker)

    def test_group_marker_coverage_export_and_project(self):
        p = self.panel
        long = self.recording("long.csv", 2400, 600)
        short = self.recording("short.csv", 900, 300)
        short.output[305:311] = np.nan
        p._processed = [long, short, self.recording("control.csv", 3000, None)]
        p.tab_sources.setCurrentIndex(1)
        p._update_file_lists()
        p.psth_align_source.setCurrentText("Event marker")
        p.combo_psth_normalization.setCurrentIndex(p.combo_psth_normalization.findData("none"))
        p.tab_visual_mode.setCurrentIndex(1)
        p.combo_window_unit.setCurrentText("Minutes")
        p.spin_resample.setValue(1)
        p.btn_available_window.click()
        self.assertEqual((p.spin_pre.value(), p.spin_post.value()), (600, 1800))
        self.assertIn("2,401", p.lbl_psth_bins.text())
        p.spin_b0.setValue(-600)
        p.spin_metric_post1.setValue(1500)
        p._compute_psth()
        np.testing.assert_equal(p._last_tvec, np.arange(-600, 1801))
        coverage = p._coverage_counts
        self.assertEqual(p._group_labels, ["long", "short"])
        np.testing.assert_equal(coverage[[0, 300, 600, 607, 1200, 1201, -1]], [1, 2, 2, 1, 2, 1, 1])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _export_to_folder(p, tmp, avg=True)
            avg = np.genfromtxt(next(folder.glob("*_avg_psth.csv")), delimiter=",", names=True)
            np.testing.assert_equal(avg["n_recordings"], coverage)
            with h5py.File(next(folder.glob("*_avg_psth.h5"))) as f:
                np.testing.assert_equal(f["n_recordings"][:], coverage)
            project = str(folder / "project.h5")
            p._save_project_h5(project)
            p.combo_window_unit.setCurrentText("Seconds")
            p.spin_post.setValue(5)
            self.assertTrue(p._load_project_from_path(project))
            self.assertEqual(p.combo_window_unit.currentData(), "min")
            self.assertEqual(p.spin_post.value(), 1800)
            self.assertEqual(p.spin_metric_post1.value(), 1500)
            self.assertEqual(p.spin_b0.value(), -600)
            self.assertEqual([r.injection_seconds for r in p._processed], [600, 300, None])
        p.tab_visual_mode.setCurrentIndex(0)
        p._compute_psth()
        p.combo_individual_file.setCurrentText("short")
        p.btn_available_window.click()
        self.assertEqual((p.spin_pre.value(), p.spin_post.value()), (300, 600))
        self.assertIsNone(p._coverage_counts)

    def test_group_trial_counts_and_recording_counts(self):
        p = self.panel
        p._processed = [self.recording("one.csv", 10, None), self.recording("two.csv", 10, None)]
        p.tab_sources.setCurrentIndex(1)
        p.tab_visual_mode.setCurrentIndex(1)
        p.cb_exclude_low_event_animals.setChecked(False)
        p.cb_group_keep_trials.setChecked(True)
        p.combo_psth_normalization.setCurrentIndex(p.combo_psth_normalization.findData("none"))
        p._get_events_for_proc = lambda r: (
            np.array([1., 3.] if r is p._processed[0] else [2.]), np.array([]))
        p._filter_events = lambda ev, dur: (ev, dur)
        p.spin_resample.setValue(1)
        p.btn_available_window.click()
        p._compute_psth()
        self.assertEqual(p._last_mat.shape[0], 3)
        zero = np.flatnonzero(p._last_tvec == 0)[0]
        self.assertEqual(p._coverage_counts[zero], 2)
        self.assertEqual(p._coverage_counts[0], 1)
        self.assertEqual(p._coverage_counts[-1], 1)

    def test_marker_at_start_and_recording_without_marker(self):
        p = self.panel
        proc = self.recording("at_start.csv", 600, 0)
        p._processed = [proc]
        p.psth_align_source.setCurrentText("Event marker")
        p.spin_resample.setValue(.05)
        p.btn_available_window.click()
        self.assertEqual((p.spin_pre.value(), p.spin_post.value()), (0, 600))
        self.assertEqual(p.lbl_psth_bins.text(), "31 bins per event")
        proc.injection_seconds = None
        p.btn_available_window.click()
        self.assertEqual((p.spin_pre.value(), p.spin_post.value()), (0, 600))

    def test_large_request_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "Lower Resample"):
            compute_psth_matrix(np.arange(10.), np.arange(10.), [2, 3],
                                (-36000, 36000), (-1, 0), 1000)
