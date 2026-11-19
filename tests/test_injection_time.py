import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import test_postprocessing_empty_state as fixture
from PySide6 import QtCore, QtWidgets
from test_postprocessing_export_pipeline import _export_to_folder
from injection_time import injection_times, period_summary


class InjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def recording(self, name):
        time = np.arange(10., 121., .1)
        signal = np.sin(time)
        return fixture.ProcessedTrial(path=name, channel_id="AIN01", time=time, raw_signal=signal,
                                      raw_reference=signal, output=signal, output_label="dFF")

    def test_photometry_only_group_and_project_roundtrip(self):
        p = self.panel
        first, second, control = [self.recording(f"{n}.csv") for n in "abc"]
        p._processed = [first, second, control]
        p.tab_sources.setCurrentIndex(1)
        p._update_file_lists()
        p.combo_align.setCurrentText("Injection")
        self.assertTrue(p.cb_injection.isChecked())
        p.recordings_section.detach()
        self.app.processEvents()
        def row_y(widget, row):
            return widget.viewport().mapToGlobal(widget.visualItemRect(widget.item(row)).topLeft()).y()
        self.assertEqual(row_y(p.list_injection, 1), row_y(p.list_preprocessed, 1))
        def enter_time():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QtWidgets.QComboBox).setCurrentIndex(1)
            dialog.findChild(QtWidgets.QLineEdit).setText("00:00:30")
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        QtCore.QTimer.singleShot(0, enter_time)
        p._edit_injection(p.list_injection.item(0))
        second.injection_seconds = 50.
        p._compute_psth()
        np.testing.assert_allclose(p._last_events, [40, 60])
        self.assertEqual(p._group_labels, ["a", "b"])
        self.assertFalse(p._behavior_sources)
        p._processed.reverse()
        p._update_file_lists()
        self.assertEqual(p.list_injection.item(2).text(), "00:00:30")
        with tempfile.TemporaryDirectory() as tmp:
            p.tab_visual_mode.setCurrentIndex(1)
            p._compute_psth()
            _export_to_folder(p, tmp, pub_figure=True, pub_content="Heatmap + Avg PSTH")
            manifest = json.loads(next(Path(tmp).glob("*_manifest.json")).read_text())
            self.assertEqual(manifest["status"], "complete")
            records = {row["file"]: row for row in manifest["injections"]}
            self.assertIsNone(records["c.csv"]["entered_seconds"])
            self.assertEqual(records["a.csv"]["entered_seconds"], 30.)
            self.assertEqual(records["a.csv"]["photometry_seconds"], 40.)
            self.assertTrue(next(Path(tmp).glob("*_publication_figure.pdf")).stat().st_size > 100)
            path = str(Path(tmp) / "test.h5")
            p._save_project_h5(path)
            self.assertTrue(p._load_project_from_path(path))
            self.assertEqual([r.injection_seconds for r in p._processed], [None, 50., 30.])
            self.assertEqual(p.combo_align.currentText(), "Injection")
        control.injection_seconds = 0.
        self.assertEqual(injection_times(control)[0], 10.)

    def test_video_clock_with_distinct_offsets_and_missing_sync(self):
        p = self.panel
        p.combo_align.setCurrentText("Injection")
        for offset in (-4., 3.):
            proc = self.recording("test")
            proc.injection_clock = "video"
            proc.injection_seconds = 60.
            self.assertFalse(p._get_events_for_proc(proc)[0].size)
            proc.sync_aligned_time = proc.time + offset
            p.cb_sync_use_aligned.setChecked(False)
            np.testing.assert_allclose(p._get_events_for_proc(proc)[0], [60. - offset])
            p.cb_sync_use_aligned.setChecked(True)
            np.testing.assert_allclose(p._get_events_for_proc(proc)[0], [60.])
            proc.injection_seconds = 190.
            self.assertFalse(p._get_events_for_proc(proc)[0].size)

    def test_behavior_periods_and_crossing_bout(self):
        p = self.panel
        proc = self.recording("test")
        proc.injection_seconds = 50.
        proc.sync_aligned_time = proc.time + 5.
        p.cb_injection.setChecked(True)
        p.cb_sync_use_aligned.setChecked(False)
        p._processed = [proc]
        p._behavior_sources = {"test": {"event_behaviors": {"groom": {
            "on": [63., 65., 80.], "off": [68., 66., 82.], "dur": [5., 1., 2.]}}}}
        p.combo_behavior_name.addItem("groom")
        p.combo_behavior_name.setCurrentText("groom")
        p.combo_injection_period.setCurrentIndex(1)
        np.testing.assert_allclose(p._get_events_for_proc(proc)[0], [58.])
        p.combo_injection_period.setCurrentIndex(2)
        np.testing.assert_allclose(p._get_events_for_proc(proc)[0], [60., 75.])
        rows = period_summary([58., 60., 75.], [5., 1., 2.], [(10., 70.), (75., 80.)], 60.)
        self.assertEqual(rows, [("Before", 1, 50., 1.2, 2.), ("After", 2, 15., 8., 5.)])
