"""Regression checks for adjustable peak detection and independent batch records."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pyqtgraph as pg

# Import the fixture module, not its TestCase, to avoid duplicate discovery.
import test_postprocessing_empty_state as fixture


class SignalEventsWorkflowTests(unittest.TestCase):
    """Exercise the public panel workflow with deterministic synthetic recordings."""

    @classmethod
    def setUpClass(cls):
        cls.app = fixture.QtWidgets.QApplication.instance() or fixture.QtWidgets.QApplication([])

    def setUp(self):
        fixture.PostprocessingEmptyStateTests.setUp(self)

    def tearDown(self):
        fixture.PostprocessingEmptyStateTests.tearDown(self)

    def load_traces(self, traces):
        """Install test-only signals through the normal preprocessing handoff."""
        records = [
            fixture.ProcessedTrial(
                path=name + ".csv", channel_id="AIN01", time=time,
                raw_signal=signal.copy(), raw_reference=np.zeros_like(signal),
                output=signal.copy(), output_label="dFF",
            )
            for name, time, signal in traces
        ]
        self.panel.receive_current_processed(records)
        self.panel._psth_timer.stop()
        self.panel.cb_peak_auto_mad.setChecked(False)
        self.panel.spin_peak_prominence.setValue(0.5)
        self.panel.spin_peak_smooth.setValue(0)
        self.panel.combo_peak_baseline.setCurrentText("Use trace as-is")
        return records

    def test_batch_keeps_quiet_file_and_rates_use_recorded_exposure(self):
        """A silent recording contributes exposure, never artificial intervals."""
        time = np.linspace(0, 10, 1001)
        first = np.zeros_like(time)
        first[[200, 600]] = 2
        second = np.zeros_like(time)
        second[[300, 700]] = 2
        self.load_traces([("first", time, first), ("second", time, second),
                          ("quiet", time, np.zeros_like(time))])
        self.panel.combo_signal_scope.setCurrentText("All files")
        self.panel._detect_signal_events()
        result = self.panel.last_signal_events
        summaries = {row["file_id"]: row for row in result["file_summaries"]}
        self.assertEqual(set(summaries), {"first", "second", "quiet"})
        self.assertEqual(summaries["quiet"]["number_of_peaks"], 0)
        self.assertEqual(summaries["quiet"]["peak_frequency_per_min"], 0)
        np.testing.assert_allclose(np.sort(result["inter_peak_intervals_sec"]), [4, 4])
        exposure = sum(row["observed_duration_s"] for row in summaries.values())
        self.assertAlmostEqual(exposure, 30, delta=0.05)
        self.assertAlmostEqual(result["derived_metrics"]["peak_frequency_per_min"], 4 * 60 / exposure)

    def test_zero_peak_result_survives_session_roundtrip(self):
        """A completed silent-file analysis is a result and must remain inspectable."""
        time = np.linspace(0, 10, 1001)
        self.load_traces([("quiet", time, np.zeros_like(time))])
        self.panel._detect_signal_events()
        result = self.panel.last_signal_events
        self.assertIsInstance(result, dict)
        self.assertEqual(result["peak_times_sec"].size, 0)
        self.assertEqual(result["file_summaries"][0]["number_of_peaks"], 0)
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "results.h5", "w") as handle:
                self.panel._save_signal_events_h5(handle)
                restored = self.panel._load_signal_events_h5(handle)
        restored_summary = restored["file_summaries"][0]
        self.assertEqual(restored_summary["file_id"], "quiet")
        self.assertEqual(restored_summary["number_of_peaks"], 0)
        self.assertEqual(restored_summary["status"], result["file_summaries"][0]["status"])
        self.assertAlmostEqual(restored_summary["observed_duration_s"],
                               result["file_summaries"][0]["observed_duration_s"])
        np.testing.assert_array_equal(restored["inter_peak_intervals_sec"], [])

    def test_cut_time_is_excluded_from_rate_and_interval_metrics(self):
        """A removed time interval contributes neither exposure nor an interval."""
        time = np.r_[np.linspace(0, 4, 401), np.linspace(7, 10, 301)]
        signal = np.zeros_like(time)
        signal[[200, 501]] = 2
        self.load_traces([("cut", time, signal)])
        self.panel._detect_signal_events()
        result = self.panel.last_signal_events
        summary = result["file_summaries"][0]
        self.assertEqual(summary["number_of_peaks"], 2)
        self.assertAlmostEqual(summary["observed_duration_s"], 7, delta=0.03)
        self.assertEqual(result["inter_peak_intervals_sec"].size, 0)
        self.assertAlmostEqual(summary["peak_frequency_per_min"],
                               120 / summary["observed_duration_s"])

    def test_export_includes_silent_recording_summary(self):
        """Export zero counts explicitly so downstream comparisons retain quiet files."""
        time = np.linspace(0, 10, 1001)
        self.load_traces([("quiet", time, np.zeros_like(time))])
        self.panel._detect_signal_events()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(fixture.QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory):
                self.panel._export_signal_events_csv()
            with (Path(directory) / "quiet_peaks.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("peak_time_sec", reader.fieldnames)
                self.assertEqual(list(reader), [])
            with (Path(directory) / "quiet_peaks_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["file_id"], "quiet")
            self.assertEqual(float(rows[0]["number_of_peaks"]), 0)
            self.assertEqual(float(rows[0]["peak_frequency_per_min"]), 0)

    def test_baseline_window_controls_auto_noise_without_normalization(self):
        """The chosen quiet window affects detection independently of amplitude units."""
        time = np.linspace(0, 40, 8001)
        rng = np.random.default_rng(901)
        signal = rng.normal(size=time.size) * np.where(time <= 10, 0.01, 0.3)
        self.load_traces([("variable_noise", time, signal)])
        self.panel.cb_peak_norm_prominence.setChecked(False)
        self.panel.cb_peak_auto_mad.setChecked(True)
        source = self.panel.combo_signal_baseline_source
        source.setCurrentIndex(source.findData("whole"))
        self.panel._detect_signal_events()
        whole = self.panel.last_signal_events["file_summaries"][0]["noise_sigma"]
        source.setCurrentIndex(source.findData("window"))
        self.panel.spin_signal_baseline_start.setValue(0)
        self.panel.spin_signal_baseline_end.setValue(10)
        self.panel._detect_signal_events()
        quiet = self.panel.last_signal_events["file_summaries"][0]["noise_sigma"]
        self.assertGreater(whole, 5 * quiet)

    def test_auto_settings_are_fresh_and_editable(self):
        """Copy auto for the selected recording even before a detection run exists."""
        time = np.linspace(0, 20, 4001)
        signal = np.random.default_rng(82).normal(0, 0.03, time.size)
        self.load_traces([("noise", time, signal)])
        self.panel.cb_peak_auto_mad.setChecked(True)
        self.panel._detect_signal_events()
        expected = self.panel.last_signal_events["file_summaries"][0]["used_prominence"]
        self.panel.last_signal_events = None
        self.panel._use_auto_peak_settings()
        self.assertFalse(self.panel.cb_peak_auto_mad.isChecked())
        self.assertTrue(self.panel.spin_peak_prominence.isEnabled())
        self.assertAlmostEqual(self.panel.spin_peak_prominence.value(), expected, delta=0.0001)

    def test_noise_fill_respects_cuts_and_current_file(self):
        """Each shaded band stays on one recorded segment and one source file."""
        time = np.r_[np.linspace(0, 4, 401), np.linspace(7, 10, 301)]
        signal = 5 + np.random.default_rng(11).normal(0, 0.03, time.size)
        self.load_traces([("cut", time, signal)])
        self.panel.combo_peak_baseline.setCurrentText("Detrend with rolling median")
        self.panel.cb_peak_noise_overlay.setChecked(True)
        self.panel._detect_signal_events()
        fills = [item for item in self.panel._signal_noise_items if isinstance(item, pg.FillBetweenItem)]
        self.assertGreaterEqual(len(fills), 2)
        for fill in fills:
            for curve in fill.curves:
                x, y = curve.getData()
                finite = np.isfinite(x) & np.isfinite(y)
                self.assertFalse(np.any(x[finite] < 4.1) and np.any(x[finite] > 6.9))
                self.assertGreater(float(np.median(y[finite])), 4)
        self.panel._refresh_signal_overlay()
        for item in self.panel._signal_noise_items:
            self.panel.plot_trace.removeItem(item)
        self.panel._signal_noise_items = []
        self.panel._draw_signal_noise_overlay("other_file")
        self.assertEqual(self.panel._signal_noise_items, [])

    def test_legacy_pooled_scope_restores_as_all_files(self):
        """Old saved projects retain their batch scope after clearer labeling."""
        self.panel._apply_settings({"signal_scope": "Pooled", "signal_auto_mad": True,
                                    "signal_mad_multiplier": 3.25})
        self.assertEqual(self.panel.combo_signal_scope.currentText(), "All files")
        self.assertTrue(self.panel.cb_peak_auto_mad.isChecked())
        self.assertAlmostEqual(self.panel.spin_peak_mad_multiplier.value(), 3.25)

    def test_noise_gate_and_quiet_window_settings_roundtrip(self):
        """Saved detection preferences retain the exact noise reference and gate."""
        panel = self.panel
        panel.cb_peak_noise_gate.setChecked(True)
        panel.combo_signal_baseline_source.setCurrentIndex(panel.combo_signal_baseline_source.findData("window"))
        panel.spin_signal_baseline_start.setValue(12.5)
        panel.spin_signal_baseline_end.setValue(48.25)
        panel.spin_signal_baseline_pad.setValue(1.75)
        settings = panel._collect_settings()
        panel.cb_peak_noise_gate.setChecked(False)
        panel.combo_signal_baseline_source.setCurrentIndex(panel.combo_signal_baseline_source.findData("whole"))
        panel.spin_signal_baseline_start.setValue(0)
        panel.spin_signal_baseline_end.setValue(0)
        panel.spin_signal_baseline_pad.setValue(0)
        panel._apply_settings(settings)
        self.assertTrue(panel.cb_peak_noise_gate.isChecked())
        self.assertEqual(panel.combo_signal_baseline_source.currentData(), "window")
        self.assertAlmostEqual(panel.spin_signal_baseline_start.value(), 12.5)
        self.assertAlmostEqual(panel.spin_signal_baseline_end.value(), 48.25)
        self.assertAlmostEqual(panel.spin_signal_baseline_pad.value(), 1.75)
        settings["signal_noise_gate"] = False
        panel._apply_settings(settings)
        self.assertFalse(panel.cb_peak_noise_gate.isChecked())

    def test_cancel_after_first_file_retains_finished_results(self):
        """Cancellation preserves the completed recording and labels pending ones."""
        class CancelAfterFirst:
            """Emulate user cancellation when the second file is about to begin."""

            def setLabelText(self, text):
                self.label = text

            def setValue(self, value):
                self.value = value

            def wasCanceled(self):
                return self.value >= 1

        time = np.linspace(0, 10, 1001)
        signal = np.zeros_like(time)
        signal[500] = 2
        self.load_traces([("first", time, signal), ("second", time, signal)])
        self.panel.combo_signal_scope.setCurrentText("All files")
        self.panel._run_signal_event_detection(self.panel._resolve_signal_detection_targets(), CancelAfterFirst())
        result = self.panel.last_signal_events
        self.assertEqual(result["file_summaries"][0]["number_of_peaks"], 1)
        self.assertEqual(result["file_summaries"][1]["status"], "cancelled")
        self.assertEqual(result["file_summaries"][1]["observed_duration_s"], 0)
        self.assertEqual(result["file_ids"], ["first"])

    def test_malformed_recording_does_not_abort_next_file(self):
        """Each recording has its own failure boundary during batch execution."""
        time = np.linspace(0, 10, 1001)
        signal = np.zeros_like(time)
        signal[500] = 2
        self.load_traces([("good", time, signal)])
        targets = [("bad", time, signal[:-3]), ("good", time, signal)]
        self.panel._run_signal_event_detection(targets)
        summaries = self.panel.last_signal_events["file_summaries"]
        self.assertEqual(summaries[0]["status"], "failed")
        self.assertEqual(summaries[1]["number_of_peaks"], 1)
        self.assertEqual(self.panel.last_signal_events["file_ids"], ["good"])

    def test_session_roundtrip_preserves_noise_overlay_arrays(self):
        """Reopened analysis keeps the same threshold shading and per-file metrics."""
        time = np.r_[np.linspace(0, 4, 401), np.linspace(7, 10, 301)]
        signal = 5 + np.random.default_rng(514).normal(0, 0.03, time.size)
        self.load_traces([("cut", time, signal)])
        self.panel.cb_peak_auto_mad.setChecked(True)
        self.panel._detect_signal_events()
        original = self.panel.last_signal_events
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "results.h5", "w") as handle:
                self.panel._save_signal_events_h5(handle)
                restored = self.panel._load_signal_events_h5(handle)
        for field in ("time", "detection_trace", "baseline_trace", "height_trace"):
            np.testing.assert_allclose(restored["noise_overlay_by_file"]["cut"][field],
                                       original["noise_overlay_by_file"]["cut"][field], equal_nan=True)
        np.testing.assert_allclose(restored["inter_peak_intervals_sec"], original["inter_peak_intervals_sec"])
        for field in ("number_of_peaks", "observed_duration_s", "noise_sigma", "used_prominence"):
            self.assertEqual(restored["file_summaries"][0][field], original["file_summaries"][0][field])

    def test_duplicate_names_keep_distinct_batch_results_and_preview(self):
        """Identical basenames in different directories must retain their own signal."""
        time = np.linspace(0, 10, 1001)
        rng = np.random.default_rng(317)
        first = rng.normal(0, 0.01, time.size)
        second = 5 + rng.normal(0, 0.05, time.size)
        self.load_traces([("folder_a/recording", time, first), ("folder_b/recording", time, second)])
        self.panel.combo_signal_scope.setCurrentText("All files")
        self.panel.cb_peak_auto_mad.setChecked(True)
        self.panel._detect_signal_events()
        result = self.panel.last_signal_events
        ids = [row["file_id"] for row in result["file_summaries"]]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(set(result["noise_overlay_by_file"]), set(ids))
        self.assertGreater(result["file_summaries"][1]["noise_sigma"],
                           3 * result["file_summaries"][0]["noise_sigma"])
        self.panel.combo_signal_file.setCurrentIndex(1)
        self.app.processEvents()
        actual_time, actual_signal = self.panel.curve_trace.getData()
        np.testing.assert_allclose(actual_time, time)
        np.testing.assert_allclose(actual_signal, second)


if __name__ == "__main__":
    unittest.main()
