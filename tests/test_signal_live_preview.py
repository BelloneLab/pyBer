"""Exercise debounced detection through the real Qt event loop and panel controls."""

import time
import unittest
from unittest.mock import patch

import numpy as np
import pyqtgraph as pg
from PySide6 import QtTest

# Reuse the isolated settings fixture without discovering its tests twice.
import test_postprocessing_empty_state as fixture


class SignalLivePreviewTests(unittest.TestCase):
    """Live previews must reflect current settings without explicit detection."""

    @classmethod
    def setUpClass(cls):
        """Keep one Qt application for the suite's offscreen widgets."""
        cls.app = fixture.QtWidgets.QApplication.instance() or fixture.QtWidgets.QApplication([])

    def setUp(self):
        """Create a panel with isolated preferences and no project autosave."""
        fixture.PostprocessingEmptyStateTests.setUp(self)

    def tearDown(self):
        """Release timers and widgets before the next isolated case."""
        fixture.PostprocessingEmptyStateTests.tearDown(self)

    def load_traces(self, count=1):
        """Load two known peaks per file, with an optional silent last file."""
        clock = np.linspace(0, 10, 1001)
        records = []
        for index in range(count):
            signal = np.random.default_rng(12 + index).normal(0, 0.002, clock.size)
            if index == 0 or index < count - 1:
                signal[200], signal[600] = 1.0, 2.0
            records.append(fixture.ProcessedTrial(
                path=f"recording_{index}.csv", channel_id="AIN01", time=clock.copy(),
                raw_signal=signal.copy(), raw_reference=np.zeros_like(signal),
                output=signal.copy(), output_label="dFF",
            ))
        self.panel.receive_current_processed(records)
        self.panel._psth_timer.stop()
        self.panel.combo_signal_source.setCurrentText("Use processed output trace (loaded file)")
        self.panel.cb_peak_auto_mad.setChecked(False)
        self.panel.cb_peak_noise_gate.setChecked(False)
        self.panel.spin_peak_prominence.setValue(0.5)
        self.panel.spin_peak_smooth.setValue(0)
        self.panel.combo_peak_baseline.setCurrentText("Use trace as-is")
        self.panel.cb_peak_noise_overlay.setChecked(True)
        return records

    def wait_for_preview(self, previous=None):
        """Pump Qt with a bounded deadline until a fresh result is published."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = self.panel.last_signal_events
            if (isinstance(result, dict) and result is not previous
                    and not result.get("settings_changed", False)
                    and self.panel._signal_live_job is None
                    and not self.panel._signal_preview_timer.isActive()):
                return result
            QtTest.QTest.qWait(10)
        self.fail("Automatic detection did not publish current settings within five seconds")

    def test_opening_events_generates_first_preview(self):
        """The first visit computes peaks even without a previous manual run."""
        self.load_traces()
        self.assertIsNone(self.panel.last_signal_events)
        self.panel._section_buttons["signal"].setChecked(True)
        result = self.wait_for_preview()
        self.assertEqual(result["peak_times_sec"].size, 2)
        self.assertTrue(self.panel._signal_preview_timer.isSingleShot())

    def test_live_preview_preserves_pending_psth_update(self):
        """Refreshing peaks must not cancel an independently scheduled PSTH."""
        self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        self.panel._psth_pending = True
        self.panel._psth_timer.start(10000)
        self.panel._signal_preview_timer.stop()
        self.panel._start_signal_preview()
        self.assertTrue(self.panel._psth_pending)
        self.assertTrue(self.panel._psth_timer.isActive())

    def test_loading_while_events_active_generates_preview(self):
        """Opening Events before loading a recording also starts detection."""
        self.panel._section_buttons["signal"].setChecked(True)
        self.load_traces()
        self.assertEqual(self.wait_for_preview()["peak_times_sec"].size, 2)

    def test_rapid_edits_coalesce_into_one_detection_pass(self):
        """Rapid control changes should run only the final requested settings."""
        self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        previous = self.wait_for_preview()
        with patch.object(self.panel, "_preprocess_signal_for_peaks",
                          wraps=self.panel._preprocess_signal_for_peaks) as preprocess:
            for prominence in (0.7, 1.2, 0.9, 1.5):
                self.panel.spin_peak_prominence.setValue(prominence)
            result = self.wait_for_preview(previous)
        self.assertEqual(preprocess.call_count, 1)
        self.assertEqual(result["peak_times_sec"].size, 1)
        self.assertAlmostEqual(result["file_summaries"][0]["used_prominence"], 1.5)

    def test_threshold_edit_refreshes_markers_and_noise_band(self):
        """Changing prominence updates the analysis and visible overlays together."""
        self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        previous = self.wait_for_preview()
        self.assertEqual(self.panel.curve_peak_markers.xData.size, 2)
        old_items = list(self.panel._signal_noise_items)
        self.panel.plot_trace.setXRange(1.0, 7.0, padding=0)
        self.panel.plot_trace.setYRange(-0.1, 2.2, padding=0)
        zoom = self.panel.plot_trace.viewRange()
        self.panel.spin_peak_prominence.setValue(1.5)
        result = self.wait_for_preview(previous)
        self.assertEqual(result["peak_times_sec"].size, 1)
        np.testing.assert_allclose(self.panel.curve_peak_markers.xData, [6.0])
        self.assertTrue(any(isinstance(item, pg.FillBetweenItem)
                            for item in self.panel._signal_noise_items))
        self.assertFalse(any(item in self.panel._signal_noise_items for item in old_items))
        self.assertIn("1.5", self.panel.lbl_peak_threshold.text())
        np.testing.assert_allclose(self.panel.plot_trace.viewRange(), zoom)

    def test_all_files_preview_retains_silent_file_without_dialog(self):
        """Live batches retain zero counts and match the explicit detection output."""
        self.load_traces(count=3)
        self.panel.combo_signal_scope.setCurrentText("All files")
        with patch.object(fixture.QtWidgets, "QProgressDialog") as progress:
            self.panel._section_buttons["signal"].setChecked(True)
            live = self.wait_for_preview()
        progress.assert_not_called()
        self.assertEqual([row["number_of_peaks"] for row in live["file_summaries"]], [2, 2, 0])
        self.panel._run_signal_event_detection(self.panel._resolve_signal_detection_targets())
        manual = self.panel.last_signal_events
        for key in ("peak_times_sec", "peak_heights", "inter_peak_intervals_sec"):
            np.testing.assert_allclose(live[key], manual[key])
        self.assertEqual([row["number_of_peaks"] for row in manual["file_summaries"]], [2, 2, 0])

    def test_superseded_batch_never_publishes_stale_parameters(self):
        """Editing between files discards the unfinished batch before publication."""
        self.load_traces(count=3)
        self.panel.combo_signal_scope.setCurrentText("All files")
        self.panel._section_buttons["signal"].setChecked(True)
        previous = self.wait_for_preview()
        self.panel.spin_peak_prominence.setValue(0.75)
        self.panel._signal_preview_timer.stop()
        self.panel._start_signal_preview()
        self.panel._signal_preview_step_timer.stop()
        self.panel._advance_signal_preview()
        self.panel._signal_preview_step_timer.stop()
        old_job = self.panel._signal_live_job
        self.assertIsNotNone(old_job)
        self.assertIs(self.panel.last_signal_events, previous)
        self.panel.spin_peak_prominence.setValue(1.5)
        self.assertIsNot(self.panel._signal_live_job, old_job)
        result = self.wait_for_preview(previous)
        self.assertEqual([row["number_of_peaks"] for row in result["file_summaries"]], [1, 1, 0])
        for summary in result["file_summaries"]:
            self.assertAlmostEqual(summary["used_prominence"], 1.5)

    def test_selected_file_switch_recomputes_preview(self):
        """Changing the selected recording replaces its peaks with the new file's."""
        self.load_traces(count=2)
        self.panel.combo_signal_scope.setCurrentText("Selected file")
        self.panel.combo_signal_file.setCurrentIndex(0)
        self.panel._section_buttons["signal"].setChecked(True)
        previous = self.wait_for_preview()
        self.assertEqual(previous["peak_times_sec"].size, 2)
        self.panel.combo_signal_file.setCurrentIndex(1)
        result = self.wait_for_preview(previous)
        self.assertEqual(result["peak_times_sec"].size, 0)
        self.assertEqual(result["file_summaries"][0]["file_id"], "recording_1")

    def test_psth_input_source_switch_recomputes_preview(self):
        """Selecting a prepared PSTH input automatically replaces loaded-file results."""
        records = self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        previous = self.wait_for_preview()
        # Supply the same internal input arrays that a completed PSTH owns.
        self.panel._last_tvec = records[0].time.copy()
        signal = np.zeros_like(self.panel._last_tvec)
        signal[400] = 3.0
        self.panel._last_mat = np.stack([signal, signal])
        self.panel.combo_signal_source.setCurrentText("Use PSTH input trace")
        self.panel._psth_timer.stop()
        result = self.wait_for_preview(previous)
        np.testing.assert_allclose(result["peak_times_sec"], [4.0])
        self.assertEqual(result["file_summaries"][0]["file_id"], "psth_trace")

    def test_clearing_recordings_cancels_pending_preview(self):
        """An old recording must not reappear after its source has been cleared."""
        self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        self.panel.receive_current_processed([])
        QtTest.QTest.qWait(400)
        self.assertIsNone(self.panel._signal_live_job)
        self.assertFalse(self.panel._signal_preview_timer.isActive())
        self.assertIsNone(self.panel.last_signal_events)

    def test_closing_panel_cancels_pending_preview(self):
        """Closing the panel prevents delayed analysis of disposed UI state."""
        self.load_traces()
        self.panel._section_buttons["signal"].setChecked(True)
        self.panel._project_dirty = False
        self.panel.close()
        self.assertFalse(self.panel._signal_preview_timer.isActive())
        self.assertFalse(self.panel._signal_preview_step_timer.isActive())
        self.assertIsNone(self.panel._signal_live_job)


if __name__ == "__main__":
    unittest.main()
