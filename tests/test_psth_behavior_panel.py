"""Exercise cached behavior charts, keyboard controls, and recording scope."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtTest
from analysis_core import ProcessedTrial
from gui_postprocessing import PostProcessingPanel


class BehaviorPanelTests(unittest.TestCase):
    """Build the real panel with isolated settings and small known recordings."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.resources = ExitStack()
        folder = self.resources.enter_context(tempfile.TemporaryDirectory())
        settings_type = QtCore.QSettings

        class IsolatedSettings(settings_type):
            def __init__(self, *_args, **_kwargs):
                super().__init__(str(Path(folder) / "settings.ini"), settings_type.Format.IniFormat)

        self.resources.enter_context(patch.object(QtCore, "QSettings", IsolatedSettings))
        self.resources.enter_context(patch.object(PostProcessingPanel, "_restore_project_autosave_if_needed"))
        self.resources.enter_context(patch.object(PostProcessingPanel, "_autosave_project_cache_path", return_value=str(Path(folder) / "autosave.h5")))
        self.panel = PostProcessingPanel()
        p = self.panel
        p._is_restoring_settings = True
        t = np.arange(0, 101, dtype=float)
        p._processed = [ProcessedTrial(path=f"{name}.csv", channel_id="AIN01", time=t,
                                      raw_signal=t, raw_reference=t, output=np.sin(t), output_label="dFF")
                        for name in ("mouse1", "mouse2")]
        p.combo_individual_file.addItems(["mouse1", "mouse2"])
        p.combo_individual_file.setCurrentText("mouse1")
        p._per_file_event_rows = {
            "mouse1": [{"file_id": "mouse1", "event_time_sec": 10., "duration_sec": 2.},
                       {"file_id": "mouse1", "event_time_sec": 20., "duration_sec": 4.}],
            "mouse2": [{"file_id": "mouse2", "event_time_sec": 30., "duration_sec": 8.}],
        }
        p._last_event_rows = sum(p._per_file_event_rows.values(), [])
        p._is_restoring_settings = False
        p._psth_timer.stop()
        p._refresh_psth_duration_view()

    def tearDown(self):
        self.panel._project_dirty = False
        self.panel._psth_timer.stop()
        self.panel.close()
        self.panel.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.resources.close()

    def test_metric_and_keyboard_bin_redraw_without_psth_compute(self):
        p = self.panel
        with patch.object(p, "_compute_psth") as compute:
            p.combo_psth_behavior_metric.setCurrentIndex(1)
            self.assertTrue(p.spin_psth_behavior_bin.isEnabled())
            p.spin_psth_behavior_bin.lineEdit().selectAll()
            QtTest.QTest.keyClicks(p.spin_psth_behavior_bin.lineEdit(), "10")
            QtTest.QTest.keyClick(p.spin_psth_behavior_bin.lineEdit(), QtCore.Qt.Key.Key_Return)
            self.assertEqual(p.spin_psth_behavior_bin.value(), 10)
            summary = p._current_psth_behavior_summary()
            self.assertEqual(len(summary["values"]), 10)
            self.assertTrue(p.plot_dur.property("hasPlotData"))
            self.assertIn("frequency", p.plot_dur.getPlotItem().titleLabel.text.lower())
            compute.assert_not_called()

    def test_all_metrics_and_exact_median_annotation(self):
        p = self.panel
        self.assertEqual(p.combo_psth_behavior_metric.count(), 8)
        reference = next(item for item in p.plot_dur.items() if isinstance(item, pg.InfiniteLine))
        self.assertEqual(reference.value(), 3)
        annotation = next(item for item in p.plot_dur.items() if isinstance(item, pg.TextItem))
        self.assertIn("Median 3 s", annotation.toPlainText())
        with patch.object(p, "_compute_psth") as compute:
            for code in ("occupancy", "cumulative_count", "onset_interval", "duration_time"):
                p.combo_psth_behavior_metric.setCurrentIndex(p.combo_psth_behavior_metric.findData(code))
                self.assertTrue(p._current_psth_behavior_summary()["has_data"])
                self.assertTrue(p.plot_dur.property("hasPlotData"))
                self.assertIn("Median" if code != "cumulative_count" else "median", next(
                    item for item in p.plot_dur.items() if isinstance(item, pg.TextItem)).toPlainText())
            compute.assert_not_called()

    def test_individual_group_and_offset_reconstruction(self):
        p = self.panel
        self.assertEqual(p._current_psth_behavior_summary()["file_ids"], ["mouse1"])
        with QtCore.QSignalBlocker(p.combo_individual_file):
            p.combo_individual_file.setCurrentText("mouse2")
        self.assertEqual(p._current_psth_behavior_summary()["file_ids"], ["mouse2"])
        p._psth_behavior_offset_aligned = True
        record = p._psth_behavior_summary_recordings()[0]
        np.testing.assert_array_equal(record["onsets"], [22])
        np.testing.assert_array_equal(record["offsets"], [30])
        with QtCore.QSignalBlocker(p.tab_visual_mode):
            p.tab_visual_mode.setCurrentIndex(1)
        self.assertEqual(len(p._current_psth_behavior_summary()["file_ids"]), 2)

    def test_cumulative_and_histogram_geometry_and_theme(self):
        p = self.panel
        p.combo_psth_behavior_metric.setCurrentIndex(3)
        p.spin_psth_behavior_bin.setValue(10)
        summary = p._current_psth_behavior_summary()
        self.assertEqual(summary["values"][-1], 6)
        for theme in ("Paper", "Midnight", "Sand"):
            p._set_plot_preset(theme)
            self.assertIn("Cumulative", p.plot_dur.getPlotItem().titleLabel.text)
            self.assertTrue(p.plot_dur.property("hasPlotData"))
        p.combo_psth_behavior_metric.setCurrentIndex(0)
        p.cb_psth_behavior_auto_bins.setChecked(False)
        p.spin_psth_behavior_bin.setValue(1)
        summary = p._current_psth_behavior_summary()
        bars = next(item for item in p.plot_dur.items() if isinstance(item, pg.BarGraphItem))
        np.testing.assert_allclose(bars.opts["x"], (summary["edges"][:-1] + summary["edges"][1:]) / 2)
        self.assertTrue(np.all(bars.opts["width"] < np.diff(summary["edges"])))

    def test_settings_roundtrip_and_missing_bout_ends(self):
        p = self.panel
        p.combo_psth_behavior_metric.setCurrentIndex(2)
        p.cb_psth_behavior_auto_bins.setChecked(False)
        p.spin_psth_behavior_bin.setValue(.125)
        settings = p._collect_settings()
        p.combo_psth_behavior_metric.setCurrentIndex(0)
        p._is_restoring_settings = True
        p._apply_settings(settings)
        p._is_restoring_settings = False
        self.assertEqual(p.combo_psth_behavior_metric.currentData(), "ibi")
        self.assertEqual(p.spin_psth_behavior_bin.value(), .125)
        self.assertFalse(p.cb_psth_behavior_auto_bins.isChecked())
        for row in p._per_file_event_rows["mouse1"]:
            row["duration_sec"] = np.nan
        p._refresh_psth_duration_view()
        self.assertFalse(p.plot_dur.property("hasPlotData"))
        p.combo_psth_behavior_metric.setCurrentIndex(1)
        self.assertTrue(p.plot_dur.property("hasPlotData"))

    def test_excessive_bins_clear_stale_chart_and_recover(self):
        p = self.panel
        p.combo_psth_behavior_metric.setCurrentIndex(1)
        notifications = QtTest.QSignalSpy(p.statusUpdate)
        p.spin_psth_behavior_bin.setValue(.001)
        self.assertFalse(p.plot_dur.property("hasPlotData"))
        self.assertTrue(p.plot_dur.toolTip())
        self.assertGreater(notifications.count(), 0)
        p.spin_psth_behavior_bin.setValue(10)
        self.assertTrue(p.plot_dur.property("hasPlotData"))

    def test_summary_controls_undo_redo_and_default_reset(self):
        p = self.panel
        p._reset_history_snapshot()
        p.combo_psth_behavior_metric.setCurrentIndex(3)
        self.assertTrue(p._history_undo)
        with patch.object(p, "_compute_psth"), patch.object(p, "_update_trace_preview"):
            p._undo_post_action()
            self.assertEqual(p.combo_psth_behavior_metric.currentData(), "duration")
            p._redo_post_action()
            self.assertEqual(p.combo_psth_behavior_metric.currentData(), "cumulative")
        p._is_restoring_settings = True
        p._apply_settings(p._default_settings_payload())
        p._is_restoring_settings = False
        self.assertEqual(p.combo_psth_behavior_metric.currentData(), "duration")
        self.assertEqual(p.spin_psth_behavior_bin.value(), 30)
        self.assertTrue(p.cb_psth_behavior_auto_bins.isChecked())


if __name__ == "__main__":
    unittest.main()
