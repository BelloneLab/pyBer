"""Baseline suggestions stay optional, scoped and safe across asynchronous closure."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets
from analysis_core import ProcessedTrial
from baseline_advisor_dialog import BaselineAdvisorDialog, _ACTIVE_JOBS
from gui_postprocessing import PostProcessingPanel


def _report(status="recommended"):
    """A transparent fixture tests presentation independently of numerical quality."""
    return {
        "status": status, "window": [-8.0, -1.0] if status == "recommended" else None,
        "summary": "Synthetic presentation fixture.", "reasons": [],
        "candidates": [{"start": -8.0, "end": -1.0, "score": 0.8,
                        "eligible": status == "recommended",
                        "diagnostics": {"training": {"coverage": 1.0},
                                        "validation": {"coverage": 0.95}}}],
        "recordings": [{"label": "recording_a", "training_events": 20, "validation_events": 10}],
        "current": {"start": -1.0, "end": 0, "eligible": False}, "config": {},
    }


class BaselineAdvisorGuiTests(unittest.TestCase):
    """Use isolated settings so checking analysis controls never touches a project."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.resources = ExitStack()
        temporary = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="pyber-baseline-ui-"))

        class IsolatedSettings(QtCore.QSettings):
            def __init__(self, *_args, **_kwargs):
                super().__init__(str(Path(temporary) / "preferences.ini"), QtCore.QSettings.Format.IniFormat)

        self.resources.enter_context(patch.object(QtCore, "QSettings", IsolatedSettings))
        self.resources.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed"))
        self.resources.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path",
                                                 return_value=str(Path(temporary) / "autosave.h5")))
        self.panel = PostProcessingPanel()
        self.app.processEvents()

    def tearDown(self):
        self.panel._project_dirty = False
        self.panel._psth_timer.stop()
        self.panel.close()
        self.panel.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.resources.close()

    @staticmethod
    def _recording(name):
        t = np.arange(0, 120, 0.1)
        return ProcessedTrial(path=f"{name}.csv", channel_id="AIN01", time=t,
                              raw_signal=np.sin(t), raw_reference=np.cos(t),
                              output=np.sin(t), output_label="dFF")

    def test_snapshot_preserves_removed_bouts_and_selects_individual(self):
        """Filtering targets must never erase the excluded preceding events."""
        panel = self.panel
        panel._processed = [self._recording("a"), self._recording("b")]
        panel.combo_individual_file.addItems(["a", "b"])
        panel.combo_individual_file.setCurrentText("b")
        panel.spin_event_start.setValue(2)
        panel.spin_group_window.setValue(0)
        raw_events = np.array([10., 30., 50.])
        durations = np.array([2., 3., 4.])
        panel.combo_behavior_align.setCurrentText("Align to onset")
        with patch.object(panel, "_get_events_for_proc", return_value=(raw_events, durations)), \
                patch.object(panel, "_proc_time", side_effect=lambda proc: proc.time + 5):
            recordings, scope = panel._baseline_advisor_recordings()
        self.assertEqual([recording.label for recording in recordings], ["b"])
        self.assertIn("Individual", scope)
        np.testing.assert_array_equal(recordings[0].events, [30, 50])
        np.testing.assert_array_equal(recordings[0].exclusion_intervals, [[10, 12], [30, 33], [50, 54]])
        self.assertEqual(recordings[0].time[0], 5)
        recordings[0].signal[:] = 17
        self.assertFalse(np.all(panel._processed[1].output == 17))

    def test_group_and_offset_alignment_keep_complete_bouts(self):
        panel = self.panel
        panel._processed = [self._recording("a"), self._recording("b")]
        panel.tab_sources.setCurrentIndex(1)
        panel.tab_visual_mode.setCurrentIndex(1)
        panel.combo_behavior_align.setCurrentText("Align to offset")
        panel.cb_filter_events.setChecked(False)
        with patch.object(panel, "_get_events_for_proc", return_value=(np.array([10., 30.]), np.array([2., np.nan]))):
            recordings, scope = panel._baseline_advisor_recordings()
        self.assertEqual(len(recordings), 2)
        self.assertIn("Group", scope)
        np.testing.assert_array_equal(recordings[0].exclusion_intervals, [[8, 10], [30, 30]])

    def test_apply_is_one_undoable_change_and_keeps_normalization(self):
        """Two spin boxes change together; redraw never sees a mixed baseline."""
        panel = self.panel
        panel._reset_history_snapshot()
        before = (panel.spin_b0.value(), panel.spin_b1.value())
        normalization = panel._psth_normalization()
        observed = []
        with patch.object(panel, "_schedule_psth", side_effect=lambda: observed.append((panel.spin_b0.value(), panel.spin_b1.value()))):
            self.assertTrue(panel._apply_recommended_baseline((-8, -1)))
        self.assertEqual(observed, [(-8, -1)])
        self.assertEqual(len(panel._history_undo), 1)
        self.assertEqual(panel._psth_normalization(), normalization)
        with patch.object(panel, "_compute_psth"), patch.object(panel, "_update_trace_preview"):
            panel._undo_post_action()
        self.assertEqual((panel.spin_b0.value(), panel.spin_b1.value()), before)
        self.assertFalse(panel._apply_recommended_baseline((0, -1)))
        self.assertFalse(panel._apply_recommended_baseline((-80, -1)))

    def test_transition_exclusions_include_nonmatching_component_bouts(self):
        """Failed transition matches can still contaminate a neighboring baseline."""
        panel = self.panel
        panel._processed = [self._recording("a")]
        panel.combo_behavior_align.setCurrentText("Transition A->B")
        panel.combo_behavior_from.addItem("A")
        panel.combo_behavior_to.addItem("B")
        panel.combo_behavior_from.setCurrentText("A")
        panel.combo_behavior_to.setCurrentText("B")

        def behavior_events(_source, name):
            starts = np.array([5., 20.]) if name == "A" else np.array([10., 30.])
            return starts, starts + 2, np.full(2, 2.)

        with patch.object(panel, "_get_events_for_proc", return_value=(np.array([30.]), np.array([2.]))), \
                patch.object(panel, "_match_behavior_source", return_value={"source": "fixture"}), \
                patch.object(panel, "_extract_behavior_events", side_effect=behavior_events):
            recordings, _ = panel._baseline_advisor_recordings()
        intervals = {tuple(interval) for interval in recordings[0].exclusion_intervals}
        self.assertEqual(intervals, {(5, 7), (10, 12), (20, 22), (30, 32)})

    def test_result_needs_explicit_apply_and_changed_settings_invalidate_it(self):
        dialog = BaselineAdvisorDialog([], current_window=(-1, 0), scope="Synthetic fixture", auto_start=False)
        dialog._receive_result(_report())
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Rejected)
        dialog.lookback.setValue(20)
        self.assertIsNone(dialog.proposed_window)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertFalse(dialog.export_button.isEnabled())
        dialog._receive_result(_report("unavailable"))
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog._apply()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Rejected)
        dialog._receive_result(_report())
        dialog._apply()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        dialog.deleteLater()

    def test_closing_during_computation_keeps_worker_and_gui_safe(self):
        """Closing discards callbacks, while Qt owns the worker until completion."""
        entered, release = threading.Event(), threading.Event()

        def estimate(*_args, **_kwargs):
            entered.set()
            release.wait(3)
            return _report()

        dialog = BaselineAdvisorDialog([], current_window=(-1, 0), scope="Synthetic fixture", auto_start=False)
        with patch("baseline_advisor_dialog.recommend_baseline", side_effect=estimate):
            dialog._start()
            deadline = time.monotonic() + 3
            while not entered.is_set() and time.monotonic() < deadline:
                self.app.processEvents()
                QtTest.QTest.qWait(10)
            self.assertTrue(entered.is_set())
            self.assertTrue(_ACTIVE_JOBS)
            dialog.reject()
            dialog.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            release.set()
            deadline = time.monotonic() + 3
            while _ACTIVE_JOBS and time.monotonic() < deadline:
                self.app.processEvents()
                QtTest.QTest.qWait(10)
        self.assertFalse(_ACTIVE_JOBS)


if __name__ == "__main__":
    unittest.main()
